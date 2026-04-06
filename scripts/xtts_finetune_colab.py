#!/usr/bin/env python3
"""
XTTS v2 Fine-Tune Script for Andreev Voice
==========================================

Run this on Google Colab (free T4 GPU) or any machine with GPU.

Steps:
1. Upload data/andreev_dataset/ folder to Colab
2. Run this script
3. Download fine-tuned model files back to server

Expected training time: 2-3 hours on T4 GPU.
"""

import os
import json
import random

# === CONFIG ===
DATASET_DIR = "data/andreev_dataset"
SEGMENTS_DIR = os.path.join(DATASET_DIR, "segments")
METADATA_FILE = os.path.join(DATASET_DIR, "metadata.csv")
OUTPUT_DIR = "xtts_finetuned_andreev"
MAX_DATASET_MINUTES = 40  # Use best 40 minutes to avoid overfitting
EPOCHS = 50
BATCH_SIZE = 2
LEARNING_RATE = 5e-6

# === METRICS ===
metrics = {
    "total_segments": 0,
    "selected_segments": 0,
    "selected_duration_min": 0,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "training_loss_history": [],
    "final_loss": None,
    "training_time_hours": None,
}


def prepare_training_data():
    """Select best segments (up to MAX_DATASET_MINUTES)."""
    import wave

    lines = open(METADATA_FILE, encoding="utf-8").readlines()
    metrics["total_segments"] = len(lines)

    # Calculate durations
    entries = []
    for line in lines:
        parts = line.strip().split("|", 1)
        if len(parts) != 2:
            continue
        seg_file, text = parts
        path = os.path.join(SEGMENTS_DIR, seg_file)
        try:
            with wave.open(path, "rb") as w:
                dur = w.getnframes() / w.getframerate()
            # Filter: 3-12 seconds, text length > 10 chars
            if 3 <= dur <= 12 and len(text) > 10:
                entries.append((seg_file, text, dur))
        except:
            pass

    # Sort by duration (prefer 5-10s segments)
    entries.sort(key=lambda x: abs(x[2] - 7.5))  # prefer ~7.5s segments

    # Select up to MAX_DATASET_MINUTES
    selected = []
    total_dur = 0
    for seg_file, text, dur in entries:
        if total_dur + dur > MAX_DATASET_MINUTES * 60:
            break
        selected.append((seg_file, text, dur))
        total_dur += dur

    metrics["selected_segments"] = len(selected)
    metrics["selected_duration_min"] = total_dur / 60

    print(f"Selected {len(selected)} segments ({total_dur/60:.1f} min) from {len(entries)} candidates")

    # Create train/eval split (90/10)
    random.shuffle(selected)
    split = int(len(selected) * 0.9)
    train = selected[:split]
    eval_set = selected[split:]

    # Write LJSpeech-format metadata
    os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "eval"), exist_ok=True)

    for subset, data in [("train", train), ("eval", eval_set)]:
        meta_path = os.path.join(OUTPUT_DIR, subset, "metadata.csv")
        with open(meta_path, "w", encoding="utf-8") as f:
            for seg_file, text, dur in data:
                # Copy or symlink audio file
                src = os.path.join(SEGMENTS_DIR, seg_file)
                dst = os.path.join(OUTPUT_DIR, subset, seg_file)
                if not os.path.exists(dst):
                    import shutil
                    shutil.copy2(src, dst)
                f.write(f"{seg_file}|{text}\n")

        print(f"  {subset}: {len(data)} segments")

    return len(train), len(eval_set)


def run_finetune():
    """Run XTTS v2 fine-tuning."""
    import time
    import torch

    os.environ["COQUI_TOS_AGREED"] = "1"
    _orig = torch.load
    def _patch(*a, **k):
        k["weights_only"] = False
        return _orig(*a, **k)
    torch.load = _patch

    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}")
    if device == "cpu":
        print("WARNING: CPU training will be very slow (10-20 hours). Use GPU!")

    # Load base model config
    config = XttsConfig()
    config.load_json(os.path.join(
        os.path.expanduser("~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2"),
        "config.json"
    ))

    # Training config
    config.training = {
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LEARNING_RATE,
        "output_path": OUTPUT_DIR,
    }

    print(f"Config: epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}")
    print(f"Starting fine-tune...")

    t0 = time.time()

    # Use the official XTTS fine-tuning recipe
    # This trains the GPT encoder (speaker adaptation) while freezing the rest
    from TTS.tts.datasets import load_tts_samples
    from trainer import Trainer, TrainerArgs

    # Load model
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_dir=os.path.expanduser(
            "~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2"
        ),
        eval=False,
    )

    if device == "cuda":
        model.cuda()

    # Load datasets
    train_samples, eval_samples = load_tts_samples(
        {"path": os.path.join(OUTPUT_DIR, "train"), "meta_file_train": "metadata.csv"},
        eval_split=True,
        eval_split_size=0.1,
    )

    # Setup trainer
    trainer_args = TrainerArgs(
        restore_path=None,
        skip_train_epoch=False,
    )

    trainer = Trainer(
        trainer_args,
        config,
        output_path=OUTPUT_DIR,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    # Train
    trainer.fit()

    elapsed_hours = (time.time() - t0) / 3600
    metrics["training_time_hours"] = round(elapsed_hours, 2)
    print(f"\nTraining complete in {elapsed_hours:.1f} hours")
    print(f"Model saved to: {OUTPUT_DIR}/")


def main():
    print("=" * 60)
    print("  XTTS v2 Fine-Tune: Andreev Voice")
    print("=" * 60)

    # Step 1: Prepare data
    n_train, n_eval = prepare_training_data()

    # Step 2: Fine-tune
    print(f"\nStarting fine-tune ({n_train} train / {n_eval} eval segments)...")
    try:
        run_finetune()
    except Exception as e:
        print(f"Fine-tune error: {e}")
        metrics["error"] = str(e)

    # Save metrics
    metrics_path = os.path.join(OUTPUT_DIR, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics: {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
