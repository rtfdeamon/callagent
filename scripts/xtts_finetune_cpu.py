#!/usr/bin/env python3
"""XTTS v2 fine-tune on Andreev's voice (CPU mode, 20 cores).

Trains only the GPT encoder (speaker adaptation) — freezes decoder.
Uses 200 best segments (~35 min) from the Andreev dataset.
Expected time: 3-6 hours on 20-core CPU.
"""
import os
import sys
import json
import wave
import time
import shutil
import random

os.environ["COQUI_TOS_AGREED"] = "1"

import torch
# Patch torch.load for XTTS compatibility
_orig = torch.load
def _patch(*a, **k):
    k["weights_only"] = False
    return _orig(*a, **k)
torch.load = _patch

# Use all CPU cores
torch.set_num_threads(16)

SEG_DIR = "/app/data/andreev_dataset/segments"
META_FILE = "/app/data/andreev_dataset/metadata.csv"
MODEL_DIR = "/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2"
OUT_DIR = "/app/data/xtts_finetuned"
MAX_MINUTES = 35
EPOCHS = 5  # Start small, check quality
BATCH_SIZE = 2
LR = 1e-5

def select_best_segments():
    """Select best segments for training."""
    lines = open(META_FILE, encoding="utf-8").readlines()
    entries = []
    for line in lines:
        parts = line.strip().split("|", 1)
        if len(parts) != 2:
            continue
        seg_file, text = parts
        path = os.path.join(SEG_DIR, seg_file)
        try:
            with wave.open(path, "rb") as w:
                dur = w.getnframes() / w.getframerate()
            if 3.0 <= dur <= 12.0 and len(text) > 15:
                entries.append((seg_file, text, dur))
        except:
            pass

    # Prefer 5-10s segments
    entries.sort(key=lambda x: abs(x[2] - 7.0))

    selected = []
    total = 0
    for seg, text, dur in entries:
        if total + dur > MAX_MINUTES * 60:
            break
        selected.append((seg, text, dur))
        total += dur

    random.shuffle(selected)
    print(f"Selected {len(selected)} segments ({total/60:.1f} min)", flush=True)
    return selected


def prepare_dataset(selected):
    """Create training dataset in XTTS format."""
    train_dir = os.path.join(OUT_DIR, "dataset", "wavs")
    os.makedirs(train_dir, exist_ok=True)

    split = int(len(selected) * 0.9)
    train_set = selected[:split]
    eval_set = selected[split:]

    for subset_name, subset in [("train", train_set), ("eval", eval_set)]:
        meta_path = os.path.join(OUT_DIR, "dataset", f"metadata_{subset_name}.csv")
        with open(meta_path, "w", encoding="utf-8") as f:
            for seg, text, dur in subset:
                src = os.path.join(SEG_DIR, seg)
                dst = os.path.join(train_dir, seg)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                f.write(f"wavs/{seg}|{text}\n")
        print(f"  {subset_name}: {len(subset)} segments", flush=True)

    return len(train_set), len(eval_set)


def run_finetune():
    """Fine-tune XTTS GPT encoder."""
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    print("Loading base model...", flush=True)
    config = XttsConfig()
    config.load_json(os.path.join(MODEL_DIR, "config.json"))

    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=MODEL_DIR, eval=False)
    print("Base model loaded", flush=True)

    # Freeze everything except GPT
    model.init_for_training()

    # Count trainable params
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params/1e6:.0f}M total, {trainable/1e6:.0f}M trainable ({trainable/total_params*100:.1f}%)", flush=True)

    # Training loop
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01
    )

    # Load training data
    dataset_dir = os.path.join(OUT_DIR, "dataset")
    train_meta = open(os.path.join(dataset_dir, "metadata_train.csv")).readlines()

    print(f"\nStarting training: {EPOCHS} epochs, {len(train_meta)} samples, lr={LR}", flush=True)

    metrics = {
        "epochs": EPOCHS,
        "samples": len(train_meta),
        "lr": LR,
        "losses": [],
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        n_batches = 0
        t0 = time.time()

        random.shuffle(train_meta)

        for i, line in enumerate(train_meta):
            parts = line.strip().split("|", 1)
            if len(parts) != 2:
                continue
            wav_file, text = parts
            wav_path = os.path.join(dataset_dir, wav_file)

            try:
                import torchaudio
                waveform, sr = torchaudio.load(wav_path)
                if sr != 22050:
                    waveform = torchaudio.functional.resample(waveform, sr, 22050)

                # Get conditioning from the same audio (self-supervised)
                gpt_cond, spk_emb = model.get_conditioning_latents(
                    audio_path=[wav_path],
                    gpt_cond_len=6,
                    gpt_cond_chunk_len=3,
                )

                # Forward pass
                loss_dict = model.forward(
                    text,
                    waveform.squeeze(),
                    gpt_cond,
                    spk_emb,
                    language="ru",
                )

                loss = loss_dict.get("loss", loss_dict.get("gpt_loss", 0))
                if isinstance(loss, torch.Tensor) and loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    epoch_loss += loss.item()
                    n_batches += 1

            except Exception as e:
                if i < 3:
                    print(f"  Sample {i} error: {e}", flush=True)
                continue

            if (i + 1) % 50 == 0:
                avg = epoch_loss / max(n_batches, 1)
                print(f"  Epoch {epoch+1}/{EPOCHS} [{i+1}/{len(train_meta)}] loss={avg:.4f}", flush=True)

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        metrics["losses"].append({"epoch": epoch+1, "loss": round(avg_loss, 4), "time_sec": round(elapsed)})
        print(f"Epoch {epoch+1}/{EPOCHS}: loss={avg_loss:.4f} ({elapsed/60:.1f} min, {n_batches} batches)", flush=True)

    # Save fine-tuned model
    ft_dir = os.path.join(OUT_DIR, "model")
    os.makedirs(ft_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ft_dir, "model.pth"))
    shutil.copy(os.path.join(MODEL_DIR, "config.json"), os.path.join(ft_dir, "config.json"))
    shutil.copy(os.path.join(MODEL_DIR, "vocab.json"), os.path.join(ft_dir, "vocab.json"))

    metrics["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    metrics["final_loss"] = round(avg_loss, 4)
    json.dump(metrics, open(os.path.join(OUT_DIR, "training_metrics.json"), "w"), indent=2)

    print(f"\nModel saved to {ft_dir}/", flush=True)
    print(f"Metrics: {json.dumps(metrics, indent=2)}", flush=True)


def main():
    print("=" * 60, flush=True)
    print("  XTTS v2 Fine-Tune: Andreev Voice (CPU)", flush=True)
    print("=" * 60, flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)

    selected = select_best_segments()
    n_train, n_eval = prepare_dataset(selected)
    run_finetune()

    print("\nDone! Next: generate cached phrases with fine-tuned model.", flush=True)


if __name__ == "__main__":
    main()
