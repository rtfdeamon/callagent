"""Pre-cache common TTS phrases at startup for instant playback.

Generates µ-law 8kHz audio for frequent phrases using Silero TTS,
stores in memory dict. On TTS request, checks cache first (fuzzy match).
"""

import logging
import os
import json
import wave
import subprocess
import tempfile
from difflib import SequenceMatcher
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Phrases to pre-generate at startup
PRECACHE_PHRASES = [
    "Добрый день!",
    "Здравствуйте!",
    "Одну секунду, проверяю информацию.",
    "Понял вас.",
    "Да, конечно.",
    "Хороший вопрос.",
    "Спасибо за звонок, всего доброго!",
    "Подскажите, как к вам лучше обращаться?",
    "Какой у вас email для отправки коммерческого предложения?",
    "Я вас услышала, давайте обсудим детали.",
    "Извините, не совсем поняла, можете повторить?",
    "Алло, вы меня слышите?",
    "К сожалению, я вас не слышу. Перезвоните, пожалуйста. До свидания.",
]

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tts_cache")


class TTSCache:
    """In-memory cache of pre-generated TTS audio."""

    def __init__(self, match_threshold: float = 0.85):
        self._cache: Dict[str, bytes] = {}  # text -> µ-law bytes
        self._threshold = match_threshold

    def get(self, text: str) -> Optional[bytes]:
        """Check cache for exact or fuzzy match. Returns µ-law bytes or None."""
        text_lower = text.strip().lower()
        # Exact match first
        for cached_text, audio in self._cache.items():
            if cached_text.lower() == text_lower:
                return audio
        # Fuzzy match
        for cached_text, audio in self._cache.items():
            ratio = SequenceMatcher(None, text_lower, cached_text.lower()).ratio()
            if ratio >= self._threshold:
                logger.debug("TTS cache fuzzy hit: '%s' ~ '%s' (%.2f)", text[:40], cached_text[:40], ratio)
                return audio
        return None

    def put(self, text: str, audio: bytes):
        """Store audio in cache."""
        self._cache[text] = audio

    def size(self) -> int:
        return len(self._cache)

    def load_from_disk(self):
        """Load pre-cached audio from disk."""
        if not os.path.isdir(CACHE_DIR):
            return
        for fname in os.listdir(CACHE_DIR):
            if fname.endswith(".json"):
                meta_path = os.path.join(CACHE_DIR, fname)
                audio_path = meta_path.replace(".json", ".ulaw")
                if os.path.exists(audio_path):
                    try:
                        meta = json.load(open(meta_path))
                        text = meta.get("text", "")
                        with open(audio_path, "rb") as f:
                            self._cache[text] = f.read()
                    except Exception:
                        pass
        if self._cache:
            logger.info("📦 TTS cache loaded: %d phrases from disk", len(self._cache))

    def save_to_disk(self):
        """Save cache to disk for persistence."""
        os.makedirs(CACHE_DIR, exist_ok=True)
        for i, (text, audio) in enumerate(self._cache.items()):
            safe = text[:50].replace(" ", "_").replace("/", "_")
            base = f"phrase_{i:03d}_{safe}"
            try:
                with open(os.path.join(CACHE_DIR, base + ".ulaw"), "wb") as f:
                    f.write(audio)
                with open(os.path.join(CACHE_DIR, base + ".json"), "w") as f:
                    json.dump({"text": text, "bytes": len(audio)}, f, ensure_ascii=False)
            except Exception:
                pass
        logger.info("📦 TTS cache saved: %d phrases to %s", len(self._cache), CACHE_DIR)
