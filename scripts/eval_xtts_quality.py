#!/usr/bin/env python3
"""Generate test phrases with XTTS and score them via back-transcription."""

from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import torch

ROOT = Path("/home/dmitriy/work/callagent")
AI_TM_DIR = ROOT / "ai_telemarketer"
sys.path.insert(0, str(AI_TM_DIR))

from stt.whisper_service import STTService
from tts.xtts_service import XTTS_Service


DEFAULT_PHRASES = [
    "Здравствуйте. Это первая контрольная фраза после полного переобучения.",
    "Добрый день. Я хотел коротко проверить темп, паузы и естественность интонации.",
    "Смотрите, если удобно, можем обсудить детали буквально за пару минут.",
    "Если вам удобно, я задам пару коротких вопросов и сразу перейду к сути.",
    "Хорошо, тогда уточню только два момента, чтобы не занимать у вас лишнее время.",
]


def normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^a-zа-я0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def score_pair(expected: str, recognized: str) -> dict[str, float]:
    expected_norm = normalize(expected)
    recognized_norm = normalize(recognized)
    expected_tokens = expected_norm.split()
    recognized_tokens = recognized_norm.split()
    char_ratio = SequenceMatcher(None, expected_norm, recognized_norm).ratio()
    word_ratio = SequenceMatcher(None, expected_tokens, recognized_tokens).ratio()
    return {
        "char_ratio": round(char_ratio, 4),
        "word_ratio": round(word_ratio, 4),
        "expected_tokens": len(expected_tokens),
        "recognized_tokens": len(recognized_tokens),
    }


def load_phrases(phrases_file: Path | None) -> list[str]:
    if not phrases_file:
        return list(DEFAULT_PHRASES)
    data = json.loads(phrases_file.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, str) and item.strip() for item in data):
        raise ValueError(f"Expected JSON array of non-empty strings in {phrases_file}")
    return [item.strip() for item in data]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate XTTS output via STT back-transcription.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "models" / "xtts_v2" / "config.json")
    parser.add_argument("--vocab", type=Path, default=ROOT / "models" / "xtts_v2" / "vocab.json")
    parser.add_argument("--speaker-wav", type=Path, default=ROOT / "models" / "andreev_voice.wav")
    parser.add_argument("--label", default="base")
    parser.add_argument("--phrases-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tmp" / "xtts_eval")
    args = parser.parse_args()

    output_dir = args.output_dir / args.label
    output_dir.mkdir(parents=True, exist_ok=True)

    phrases = load_phrases(args.phrases_file)

    tts = XTTS_Service(
        checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
        config_path=str(args.config),
        vocab_path=str(args.vocab),
        speaker_wav=str(args.speaker_wav),
    )
    stt = STTService(model_size="small", device="cpu")

    results: list[dict[str, object]] = []
    for index, phrase in enumerate(phrases, start=1):
        audio_bytes = tts.generate_audio(phrase)
        wav_path = output_dir / f"{index:02d}.wav"
        wav_path.write_bytes(audio_bytes)
        recognized = stt.transcribe(audio_bytes)
        scores = score_pair(phrase, recognized)
        row = {
            "index": index,
            "expected": phrase,
            "recognized": recognized,
            "wav_path": str(wav_path),
            **scores,
        }
        results.append(row)
        print(
            json.dumps(
                row,
                ensure_ascii=False,
            ),
            flush=True,
        )

    avg_char = sum(float(item["char_ratio"]) for item in results) / len(results)
    avg_word = sum(float(item["word_ratio"]) for item in results) / len(results)
    summary = {
        "label": args.label,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "avg_char_ratio": round(avg_char, 4),
        "avg_word_ratio": round(avg_word, 4),
        "results": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), **summary}, ensure_ascii=False, indent=2))

    del tts
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
