"""Lightweight keyword-based RAG retriever for telephony agent.

Loads compact JSON chunks from data/rag_chunks.json and matches them
against caller transcript using simple keyword overlap scoring.
Designed for low-context LLMs (2048 tokens) — returns at most 2 short
chunks to keep the prompt compact.
"""

import json
import os
import re
from typing import List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)

_CHUNKS: Optional[List[dict]] = None
_CHUNKS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "rag_chunks.json",
)

# Words too common to be useful for matching
_STOP_WORDS = frozenset(
    "а в и к на не но о от по с у я мы вы он она они что как это "
    "да нет так уже ещё бы ли же то есть был было были будет "
    "мне мой моя нам вам вас нас его её их свой этот эта эти "
    "можно нужно надо хочу могу очень тоже также просто только "
    "здравствуйте добрый день алло привет пока спасибо".split()
)

_WORD_RE = re.compile(r"[а-яёa-z0-9-]+", re.IGNORECASE)

# Minimum stem length for fuzzy Russian morphology matching
_MIN_STEM = 4


def _load_chunks() -> List[dict]:
    """Load and cache RAG chunks from JSON file."""
    global _CHUNKS
    if _CHUNKS is not None:
        return _CHUNKS
    try:
        with open(_CHUNKS_PATH, "r", encoding="utf-8") as f:
            _CHUNKS = json.load(f)
        logger.info("RAG chunks loaded", count=len(_CHUNKS), path=_CHUNKS_PATH)
    except Exception:
        logger.warning("Failed to load RAG chunks", path=_CHUNKS_PATH, exc_info=True)
        _CHUNKS = []
    return _CHUNKS


def _tokenize(text: str) -> set:
    """Extract lowercase word tokens, filtering stop words."""
    words = set(_WORD_RE.findall(text.lower()))
    return words - _STOP_WORDS


def _stem_match(keyword: str, words: set) -> bool:
    """Check if keyword root matches any transcript word (poor-man's Russian stemming).

    Russian inflection changes suffixes: музей→музеев, школа→школу, театр→театре.
    Comparing first N characters handles most cases.
    """
    kw = keyword.lower()
    if len(kw) < _MIN_STEM:
        return kw in words  # short words must match exactly
    stem = kw[:_MIN_STEM]
    return any(w.startswith(stem) for w in words if len(w) >= _MIN_STEM)


def retrieve(transcript: str, top_k: int = 2) -> str:
    """Return relevant knowledge text for a caller transcript.

    Args:
        transcript: The caller's speech-to-text transcription.
        top_k: Maximum number of chunks to return.

    Returns:
        A string with relevant knowledge snippets joined by newlines,
        or empty string if nothing matched.
    """
    chunks = _load_chunks()
    if not chunks or not transcript or not transcript.strip():
        return ""

    words = _tokenize(transcript)
    if not words:
        return ""

    transcript_lower = transcript.lower()
    scored: List[Tuple[float, dict]] = []
    for chunk in chunks:
        keywords = chunk.get("keywords", [])
        score = 0.0
        for kw in keywords:
            kw_lower = kw.lower()
            # Exact substring match (highest confidence)
            if kw_lower in transcript_lower:
                score += 2.0
            # Stem-based fuzzy match (handles Russian morphology)
            elif _stem_match(kw_lower, words):
                score += 1.5
            # Partial: any word from a multi-word keyword present
            else:
                kw_words = set(kw_lower.split())
                for kw_word in kw_words:
                    if _stem_match(kw_word, words):
                        score += 1.0 / len(kw_words)
        if score > 0:
            scored.append((score, chunk))

    if not scored:
        return ""

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    parts = []
    for _score, chunk in top:
        parts.append(chunk["text"])

    return "\n".join(parts)
