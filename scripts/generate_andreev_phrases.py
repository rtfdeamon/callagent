#!/usr/bin/env python3
"""Generate TTS phrases with Andreev's cloned voice via XTTS v2."""
import os, time, json, subprocess, wave
os.environ["COQUI_TOS_AGREED"] = "1"

import torch
_orig = torch.load
def _patch(*a, **k):
    k["weights_only"] = False
    return _orig(*a, **k)
torch.load = _patch

from TTS.api import TTS

PHRASES = [
    "Добрый день! Компания Мультимедиа Видеосистемы, меня зовут Алексей.",
    "Мы занимаемся поставкой и монтажом мультимедийного оборудования.",
    "Подскажите, какой у вас объект?",
    "Какое оборудование вас интересует?",
    "Давайте я подготовлю для вас коммерческое предложение.",
    "Мы работаем по сорок четвёртому федеральному закону.",
    "Записал, отправлю вам предложение на почту.",
    "Одну секунду, проверяю информацию.",
    "Спасибо за звонок, всего доброго!",
    "Алло, вы меня слышите?",
]

VOICE = "/app/models/andreev_voice.wav"
OUT_DIR = "/app/data/tts_cache"

print("Loading XTTS v2 on CPU...", flush=True)
t0 = time.time()
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
print(f"Loaded in {time.time()-t0:.0f}s", flush=True)

os.makedirs(OUT_DIR, exist_ok=True)

for i, phrase in enumerate(PHRASES):
    wav_path = f"/tmp/andreev_phrase_{i}.wav"
    print(f"[{i+1}/{len(PHRASES)}] {phrase[:50]}...", flush=True)
    t0 = time.time()
    tts.tts_to_file(text=phrase, speaker_wav=VOICE, language="ru", file_path=wav_path)
    dur = time.time() - t0

    ulaw_path = os.path.join(OUT_DIR, f"phrase_{i:03d}.ulaw")
    meta_path = ulaw_path.replace(".ulaw", ".json")

    subprocess.run(["sox", wav_path, "-r", "8000", "-c", "1", "-e", "mu-law",
                    "-b", "8", "-t", "raw", ulaw_path, "norm", "-3"],
                   capture_output=True)

    size = os.path.getsize(ulaw_path) if os.path.exists(ulaw_path) else 0
    audio_dur = size / 8000
    print(f"  {dur:.0f}s synth -> {audio_dur:.1f}s audio ({size//1024}KB)", flush=True)

    json.dump({"text": phrase, "bytes": size}, open(meta_path, "w"), ensure_ascii=False)

print(f"\nDone! {len(PHRASES)} phrases in {OUT_DIR}", flush=True)
