#!/usr/bin/env python3
"""Fine-tune XTTS GPT on the local Andreev dataset and save a runnable checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import wave
from pathlib import Path

os.environ.setdefault("COQUI_TOS_AGREED", "1")

import torch
from trainer import Trainer, TrainerArgs
from trainer.trainer_utils import get_optimizer

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.models.xtts import XttsAudioConfig


ROOT = Path("/home/dmitriy/work/callagent")
DATASET_DIR = ROOT / "data" / "andreev_dataset"
SEGMENTS_DIR = DATASET_DIR / "segments"
SOURCE_META = DATASET_DIR / "metadata.csv"
BASE_MODEL_DIR = ROOT / "models" / "xtts_v2"
OUTPUT_DIR = ROOT / "data" / "xtts_finetuned_andreev"


class AdaptationGPTTrainer(GPTTrainer):
    def get_optimizer(self):
        trainable_params = [param for param in self.xtts.gpt.parameters() if param.requires_grad]
        return get_optimizer(
            self.config.optimizer,
            self.config.optimizer_params,
            self.config.lr,
            parameters=trainable_params,
        )


def patch_torch_load() -> None:
    original = torch.load

    def patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = patched


BAD_TEXT_PATTERNS = (
    "субтитры",
    "multimedia",
    "tirем",
)

TRAIN_SCOPES = {
    "speaker": (
        "conditioning_encoder",
        "conditioning_perceiver",
    ),
    "light": (
        "conditioning_encoder",
        "conditioning_perceiver",
        "gpt.ln_f",
    ),
    "adaptation": (
        "conditioning_encoder",
        "conditioning_perceiver",
        "mel_head",
        "final_norm",
        "gpt.ln_f",
    ),
}


def is_text_clean(text: str, *, max_chars: int, max_words: int) -> bool:
    normalized = " ".join(text.strip().split())
    if len(normalized) < 18 or len(normalized) > max_chars:
        return False
    if len(normalized.split()) > max_words:
        return False
    if re.search(r"[A-Za-z]", normalized):
        return False
    if normalized.count("...") or "…" in normalized:
        return False
    lowered = normalized.lower()
    if any(pattern in lowered for pattern in BAD_TEXT_PATTERNS):
        return False
    tokens = [token.strip(".,!?-:;\"'()").lower() for token in normalized.split()]
    tokens = [token for token in tokens if token]
    if not tokens:
        return False
    if len(set(tokens)) <= max(2, len(tokens) // 4):
        return False
    return True


def pick_segments(max_minutes: int, seed: int, *, max_chars: int, max_words: int) -> list[dict[str, object]]:
    random.seed(seed)
    selected: list[dict[str, object]] = []
    total_sec = 0.0

    with SOURCE_META.open(encoding="utf-8") as handle:
        for raw_line in handle:
            seg_name, _, text = raw_line.strip().partition("|")
            if not seg_name or not text:
                continue
            wav_path = SEGMENTS_DIR / seg_name
            if not wav_path.exists():
                continue
            try:
                with wave.open(str(wav_path), "rb") as wav_file:
                    duration = wav_file.getnframes() / wav_file.getframerate()
            except wave.Error:
                continue
            if not (3.0 <= duration <= 12.0):
                continue
            cleaned = text.strip()
            if not is_text_clean(cleaned, max_chars=max_chars, max_words=max_words):
                continue
            selected.append(
                {
                    "seg_name": seg_name,
                    "text": cleaned,
                    "duration": duration,
                }
            )

    # Prefer medium-length, speech-rich chunks for quick speaker adaptation.
    selected.sort(key=lambda item: (abs(float(item["duration"]) - 7.0), -len(str(item["text"]))))

    final: list[dict[str, object]] = []
    for item in selected:
        if total_sec + float(item["duration"]) > max_minutes * 60:
            break
        final.append(item)
        total_sec += float(item["duration"])

    random.shuffle(final)
    return final


def write_metadata(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|")
        writer.writerow(["audio_file", "text", "speaker_name"])
        for row in rows:
            writer.writerow([f"wavs/{row['seg_name']}", row["text"], "andreev"])


def prepare_dataset(selected: list[dict[str, object]]) -> dict[str, object]:
    dataset_dir = OUTPUT_DIR / "dataset"
    wavs_dir = dataset_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    for seg_path in wavs_dir.glob("*.wav"):
        seg_path.unlink()

    split_index = max(1, int(len(selected) * 0.9))
    if split_index >= len(selected):
        split_index = max(1, len(selected) - 1)
    train_rows = selected[:split_index]
    eval_rows = selected[split_index:]

    for row in selected:
        shutil.copy2(SEGMENTS_DIR / str(row["seg_name"]), wavs_dir / str(row["seg_name"]))

    write_metadata(train_rows, dataset_dir / "metadata_train.csv")
    write_metadata(eval_rows, dataset_dir / "metadata_eval.csv")

    return {
        "dataset_dir": str(dataset_dir),
        "train_samples": len(train_rows),
        "eval_samples": len(eval_rows),
        "total_minutes": round(sum(float(row["duration"]) for row in selected) / 60, 2),
    }


def freeze_for_adaptation(model: GPTTrainer, train_scope: str) -> dict[str, float]:
    keep_prefixes = TRAIN_SCOPES[train_scope]
    trainable_params = 0
    total_params = 0
    for name, param in model.xtts.gpt.named_parameters():
        total_params += param.numel()
        param.requires_grad = name.startswith(keep_prefixes)
        if param.requires_grad:
            trainable_params += param.numel()
    return {
        "train_scope": train_scope,
        "trainable_millions": round(trainable_params / 1e6, 2),
        "total_millions": round(total_params / 1e6, 2),
    }


def train(
    *,
    output_dir: Path,
    dataset_dir: Path,
    train_scope: str,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    num_loader_workers: int,
    lr: float,
    grad_clip: float,
    save_step: int,
) -> Path:
    run_root = output_dir / "run" / "training"
    run_root.mkdir(parents=True, exist_ok=True)

    model_args = GPTArgs(
        max_conditioning_length=88200,
        min_conditioning_length=44100,
        debug_loading_failures=False,
        max_wav_length=176400,
        max_text_length=220,
        mel_norm_file=str(BASE_MODEL_DIR / "mel_stats.pth"),
        dvae_checkpoint=str(BASE_MODEL_DIR / "dvae.pth"),
        xtts_checkpoint=str(BASE_MODEL_DIR / "model.pth"),
        tokenizer_file=str(BASE_MODEL_DIR / "vocab.json"),
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    config = GPTTrainerConfig(
        epochs=epochs,
        output_path=str(run_root),
        model_args=model_args,
        run_name="andreev_xtts_ft",
        project_name="callagent",
        dashboard_logger="tensorboard",
        logger_uri=None,
        audio=audio_config,
        allow_tf32=True,
        batch_size=batch_size,
        batch_group_size=12,
        eval_batch_size=batch_size,
        num_loader_workers=num_loader_workers,
        eval_split_max_size=128,
        print_step=10,
        plot_step=50,
        log_model_step=save_step,
        grad_clip=grad_clip,
        save_step=save_step,
        save_n_checkpoints=1,
        save_checkpoints=True,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=lr,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000, 150000, 300000], "gamma": 0.5, "last_epoch": -1},
        print_eval=False,
        test_sentences=[],
    )

    config_dataset = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="andreev_ft",
        path=str(dataset_dir),
        meta_file_train="metadata_train.csv",
        meta_file_val="metadata_eval.csv",
        language="ru",
    )

    model = AdaptationGPTTrainer.init_from_config(config)
    freeze_stats = freeze_for_adaptation(model, train_scope)
    print(
        "Trainable GPT params: "
        f"{freeze_stats['trainable_millions']}M / {freeze_stats['total_millions']}M "
        f"(scope={freeze_stats['train_scope']})",
        flush=True,
    )
    train_samples, eval_samples = load_tts_samples(
        [config_dataset],
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=False,
            grad_accum_steps=grad_accum,
        ),
        config,
        output_path=str(run_root),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()
    return Path(trainer.output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune XTTS GPT locally on the Andreev dataset.")
    parser.add_argument("--max-minutes", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-loader-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-step", type=int, default=100)
    parser.add_argument("--max-chars", type=int, default=160)
    parser.add_argument("--max-words", type=int, default=26)
    parser.add_argument("--train-scope", choices=sorted(TRAIN_SCOPES), default="adaptation")
    args = parser.parse_args()

    patch_torch_load()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected = pick_segments(
        max_minutes=args.max_minutes,
        seed=args.seed,
        max_chars=args.max_chars,
        max_words=args.max_words,
    )
    if len(selected) < 8:
        raise RuntimeError(f"Not enough segments selected for training: {len(selected)}")

    dataset_info = prepare_dataset(selected)
    trainer_output = train(
        output_dir=OUTPUT_DIR,
        dataset_dir=Path(str(dataset_info["dataset_dir"])),
        train_scope=args.train_scope,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        num_loader_workers=args.num_loader_workers,
        lr=args.lr,
        grad_clip=args.grad_clip,
        save_step=args.save_step,
    )

    best_model = trainer_output / "best_model.pth"
    checkpoint = trainer_output / "checkpoint_50.pth"
    final_checkpoint = best_model if best_model.exists() else checkpoint
    if not final_checkpoint.exists():
        raise RuntimeError(f"No fine-tuned checkpoint found in {trainer_output}")

    summary = {
        "trainer_output": str(trainer_output),
        "checkpoint_path": str(final_checkpoint),
        "config_path": str(BASE_MODEL_DIR / "config.json"),
        "vocab_path": str(BASE_MODEL_DIR / "vocab.json"),
        "speaker_wav": str(ROOT / "models" / "andreev_voice.wav"),
        "max_minutes": args.max_minutes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "selected_segments": len(selected),
        "train_samples": dataset_info["train_samples"],
        "eval_samples": dataset_info["eval_samples"],
        "dataset_minutes": dataset_info["total_minutes"],
        "train_scope": args.train_scope,
        "lr": args.lr,
        "grad_clip": args.grad_clip,
        "save_step": args.save_step,
        "max_chars": args.max_chars,
        "max_words": args.max_words,
    }
    with (OUTPUT_DIR / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
