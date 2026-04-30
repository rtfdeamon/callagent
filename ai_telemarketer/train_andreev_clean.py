#!/usr/bin/env python3
"""Fine-tune XTTS GPT on the CLEAN Andreev dataset using the proven AdaptationTrainer structure."""

import os
import json
import torch
from pathlib import Path
from trainer import Trainer, TrainerArgs
from trainer.trainer_utils import get_optimizer
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.models.xtts import XttsAudioConfig

# Обязательно для работы c Coqui
os.environ.setdefault("COQUI_TOS_AGREED", "1")

# Пути
ROOT = Path("/home/dmitriy/work/callagent")
DATASET_DIR = ROOT / "data" / "andreev_dataset_clean"
BASE_MODEL_DIR = ROOT / "models" / "xtts_v2"
OUTPUT_DIR = ROOT / "data" / "xtts_finetuned_andreev_v2"

# Используем кастомный тренер для корректной фильтрации обучаемых параметров
class AdaptationGPTTrainer(GPTTrainer):
    def get_optimizer(self):
        trainable_params = [param for param in self.xtts.gpt.parameters() if param.requires_grad]
        return get_optimizer(
            self.config.optimizer,
            self.config.optimizer_params,
            self.config.lr,
            parameters=trainable_params,
        )

def train():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Режим обучения: GPT + Conditioning
    train_scope = (
        "conditioning_encoder",
        "conditioning_perceiver",
        "mel_head",
        "final_norm",
        "gpt.ln_f",
    )

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
        epochs=15,
        output_path=str(OUTPUT_DIR),
        model_args=model_args,
        run_name="andreev_clean_v4",
        project_name="callagent",
        audio=audio_config,
        allow_tf32=True,
        mixed_precision=False, # Выключаем для стабильности (исключаем NaNs)
        batch_size=1,
        batch_group_size=8,
        eval_batch_size=1,
        num_loader_workers=2,
        eval_split_max_size=128,
        print_step=10,
        plot_step=50,
        log_model_step=100,
        save_step=100,
        save_n_checkpoints=1,
        save_checkpoints=True,
        optimizer="AdamW",
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=2e-6,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000, 150000, 300000], "gamma": 0.5},
        test_sentences=[
            {
                "text": "Привет, меня зовут Андреев, я звоню вам из компании Мультимедиа.",
                "speaker_wav": str(ROOT / "models" / "andreev_voice.wav"),
                "language": "ru",
            }
        ],
    )

    config_dataset = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="andreev_clean",
        path=str(DATASET_DIR),
        meta_file_train="metadata.csv",
        language="ru",
    )

    # Инициализация с нашим кастомным классом
    model = AdaptationGPTTrainer.init_from_config(config)
    
    # Замораживаем веса
    for name, param in model.xtts.gpt.named_parameters():
        param.requires_grad = name.startswith(train_scope)

    train_samples, eval_samples = load_tts_samples(
        [config_dataset],
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    trainer = Trainer(
        TrainerArgs(grad_accum_steps=4),
        config,
        output_path=str(OUTPUT_DIR),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    
    trainer.fit()
    print("Training finished successfully!")

if __name__ == "__main__":
    train()
