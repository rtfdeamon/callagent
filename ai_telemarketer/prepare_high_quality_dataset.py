import os
import logging
import torch
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import csv
from faster_whisper import WhisperModel
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetCleaner")

# Пути
ROOT_DIR = Path("/home/dmitriy/work/callagent")
SOURCE_AUDIO_DIR = ROOT_DIR / "data" / "andreev_dataset" / "raw"
REFERENCE_WAV = ROOT_DIR / "models" / "andreev_voice.wav"
BASE_MODEL_DIR = ROOT_DIR / "models" / "xtts_v2"
OUTPUT_DIR = ROOT_DIR / "data" / "andreev_dataset_clean"
METADATA_FILE = OUTPUT_DIR / "metadata.csv"
WAVS_DIR = OUTPUT_DIR / "wavs"

# Параметры фильтрации
SIMILARITY_THRESHOLD = 0.55  # Снизили порог для 8кГц записей
MIN_DURATION = 2.0           # Мин. длительность сегмента в сек
MAX_DURATION = 12.0          # Макс. длительность сегмента в сек
SAMPLE_RATE = 22050          # Частота XTTS (родная 22050)

def load_xtts_for_embeddings():
    """Загружает XTTS только для вычисления эмбеддингов."""
    logger.info("[EMBED] Loading XTTS for speaker verification...")
    config = XttsConfig()
    config.load_json(str(BASE_MODEL_DIR / "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=str(BASE_MODEL_DIR), use_deepspeed=False)
    model.cuda()
    return model

def get_embedding(model, audio_path):
    """Вычисляет speaker_embedding для аудиофайла."""
    with torch.no_grad():
        _, latent = model.get_conditioning_latents(audio_path=[str(audio_path)])
        return latent

def cosine_similarity(emb1, emb2):
    """Вычисляет косинусное сходство между двумя тензорами эмбеддингов."""
    return torch.nn.functional.cosine_similarity(emb1, emb2).item()

def prepare_dataset():
    WAVS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Загружаем модель и считаем эталон
    model = load_xtts_for_embeddings()
    logger.info(f"[EMBED] Computing reference embedding from {REFERENCE_WAV}")
    ref_emb = get_embedding(model, REFERENCE_WAV)
    
    mp3_files = list(SOURCE_AUDIO_DIR.glob("*.mp3"))
    logger.info(f"[DATA] Found {len(mp3_files)} source MP3 files")
    
    valid_segments = []
    seg_counter = 0

    # 2. Обработка файлов: Сегментация -> Фильтрация
    for mp3_path in tqdm(mp3_files, desc="Processing files"):
        try:
            # Загружаем аудио
            y, sr = librosa.load(mp3_path, sr=SAMPLE_RATE)
            
            # Разрезаем по тишине (VAD)
            # Уменьшаем top_db, чтобы захватить и тихую речь
            intervals = librosa.effects.split(y, top_db=35)
            
            for start_idx, end_idx in intervals:
                duration = (end_idx - start_idx) / sr
                
                # Ищем сегменты от 2 до 12 секунд
                if 2.0 <= duration <= MAX_DURATION:
                    temp_seg_path = OUTPUT_DIR / f"temp_seg.wav"
                    chunk = y[start_idx:end_idx]
                    sf.write(temp_seg_path, chunk, sr)
                    
                    chunk_emb = get_embedding(model, temp_seg_path)
                    similarity = cosine_similarity(ref_emb, chunk_emb)
                    
                    if similarity >= 0.55: # Снизили порог для 8кГц записей
                        seg_name = f"andreev_{seg_counter:04d}.wav"
                        final_path = WAVS_DIR / seg_name
                        sf.write(final_path, chunk, sr)
                        
                        valid_segments.append({
                            "path": final_path,
                            "filename": seg_name,
                            "similarity": similarity
                        })
                        seg_counter += 1
                    
                    if temp_seg_path.exists():
                        temp_seg_path.unlink()
        except Exception as e:
            logger.error(f"Error processing {mp3_path}: {e}")

    # 3. Транскрипция (Освобождаем память от XTTS и грузим Whisper)
    del model
    torch.cuda.empty_cache()
    
    logger.info(f"[STT] Filtered {len(valid_segments)} segments. Loading Whisper for transcription...")
    whisper = WhisperModel("medium", device="cuda", compute_type="float16")
    
    rows = []
    for seg in tqdm(valid_segments, desc="Transcribing"):
        segments, _ = whisper.transcribe(str(seg["path"]), language="ru")
        text = " ".join([s.text for s in segments]).strip()
        
        if text:
            rows.append([f"wavs/{seg['filename']}", text, "andreev"])

    # 4. Сохранение метаданных
    with open(METADATA_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["audio_file", "text", "speaker_name"])
        writer.writerows(rows)

    logger.info(f"=== SUCCESS! Created dataset with {len(rows)} high-quality segments. ===")
    logger.info(f"Metadata saved to: {METADATA_FILE}")

if __name__ == "__main__":
    prepare_dataset()
