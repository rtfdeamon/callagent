#!/usr/bin/env python3
"""Full call cycle test: simulate caller → RAG → LLM → TTS → record audio → analyze.

Runs multiple dialogue scenarios end-to-end, records TTS audio,
measures latency and quality, and produces an improvement report.
"""
import asyncio
import base64
import json
import os
import struct
import sys
import time
import wave
from dataclasses import dataclass, field
from typing import List, Optional

import requests
import websockets

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rag import retrieve

# ── Config ──────────────────────────────────────────────────────────────
WS_URL = "ws://127.0.0.1:8765"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen2.5:7b"
LLM_OPTS = {"temperature": 0.3, "num_predict": 60, "repeat_penalty": 1.3, "num_ctx": 2048}
OUTPUT_DIR = "/tmp/call_cycle_test"

SYSTEM_PROMPT = """Ты — Анна, молодая приветливая девушка, менеджер по продажам компании MMVS «Мультимедиа Видеосистемы» (Екатеринбург).
Входящий телефонный звонок. Текст озвучивается женским голосом.
ТВОЙ ХАРАКТЕР: общительная, дружелюбная, немного неформальная. Говори живо и тепло.
КОМПАНИЯ: поставка и монтаж мультимедийного оборудования для музеев, школ, библиотек, ДК.
СКРИПТ: выясни объект, оборудование, тип учреждения, предложи КП, возьми контакт.
ПРАВИЛА: 1-2 коротких предложения. Один вопрос. Только русский. Плоский текст. Цены не называй — предлагай КП."""

# 5 dialogue scenarios
SCENARIOS = [
    {
        "name": "Школа — проекторы",
        "turns": [
            "Здравствуйте, у нас школа, нужны проекторы",
            "Три класса, средняя школа номер пятнадцать в Челябинске",
            "Работаете по сорок четвёртому ФЗ?",
            "Отправьте КП на school15@edu.ru",
            "Спасибо, до свидания",
        ],
    },
    {
        "name": "Музей — интерактив",
        "turns": [
            "Добрый день, это музей истории города",
            "Нам нужны интерактивные экраны и инфокиоски для экспозиции",
            "Бюджет примерно три миллиона, финансирование по нацпроекту",
            "Хорошо, давайте назначим встречу",
            "До свидания",
        ],
    },
    {
        "name": "Библиотека — модельная",
        "turns": [
            "Звоню из библиотеки, хотим стать модельной",
            "Нужна медиа-зона и мини-кинотеатр",
            "Сколько это примерно стоит?",
            "Отправьте предложение на biblio@mail.ru",
        ],
    },
    {
        "name": "Конференц-зал — ВКС",
        "turns": [
            "Нам нужна видеоконференцсвязь для переговорной",
            "Комната на двадцать человек, нужна совместимость с зумом",
            "Какие сроки установки?",
        ],
    },
    {
        "name": "Стадион — табло",
        "turns": [
            "Нужно электронное табло для спортзала",
            "Стадион на пять тысяч мест, нужен медиакуб",
            "А есть LED видеоборты?",
        ],
    },
]


@dataclass
class TurnResult:
    caller_text: str
    agent_text: str
    rag_hit: bool
    rag_preview: str
    tts_latency_ms: float
    tts_bytes: int
    tts_duration_sec: float
    audio_rms: float
    audio_clipping_pct: float
    issues: List[str] = field(default_factory=list)


@dataclass
class ScenarioResult:
    name: str
    turns: List[TurnResult] = field(default_factory=list)
    wav_path: str = ""


def analyze_ulaw(data: bytes) -> dict:
    """Analyze µ-law audio quality."""
    import audioop
    import numpy as np

    if not data:
        return {"rms": 0, "clipping_pct": 0, "duration_sec": 0}
    pcm = np.frombuffer(audioop.ulaw2lin(data, 2), dtype=np.int16).astype(float)
    rms = float(np.sqrt(np.mean(pcm**2)))
    clipping = float(np.sum(np.abs(pcm) > 31000) / len(pcm) * 100)
    duration = len(data) / 8000
    return {"rms": rms, "clipping_pct": clipping, "duration_sec": duration}


async def tts_synthesize(text: str) -> tuple:
    """Send TTS request via WebSocket, return (audio_bytes, latency_ms)."""
    t0 = time.monotonic()
    try:
        async with websockets.connect(WS_URL, ping_interval=None) as ws:
            await ws.send(json.dumps({
                "type": "tts_request",
                "text": text,
                "call_id": "test_cycle",
            }))
            resp = await asyncio.wait_for(ws.recv(), timeout=15)
            latency = (time.monotonic() - t0) * 1000
            if isinstance(resp, bytes):
                return resp, latency
            msg = json.loads(resp)
            if msg.get("audio_data"):
                return base64.b64decode(msg["audio_data"]), latency
            return b"", latency
    except Exception as e:
        return b"", (time.monotonic() - t0) * 1000


def llm_generate(messages: list, rag_text: str = "") -> str:
    """Call Ollama LLM."""
    msgs = list(messages)
    if rag_text:
        base = msgs[0]["content"]
        idx = base.find("\n\n[СПРАВКА]")
        if idx >= 0:
            base = base[:idx]
        msgs[0] = {**msgs[0], "content": f"{base}\n\n[СПРАВКА]\n{rag_text}"}

    try:
        r = requests.post(OLLAMA_URL, json={
            "model": MODEL, "messages": msgs[-10:],
            "stream": False, "options": LLM_OPTS,
        }, timeout=30)
        text = r.json()["message"]["content"].strip()
    except Exception:
        text = ""

    # Sanitize (same as ollama.py)
    import re
    text = re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef\u2e80-\u2eff]+', '', text)
    text = re.sub(r'<\|im_start\|>.*', '', text, flags=re.DOTALL)
    text = re.sub(r'\*{1,3}|#{1,6}\s?|`{1,3}|~{2,3}', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r' {2,}', ' ', text).strip()
    if len(text) > 200:
        text = text[:200]
        for sep in ('.', '!', '?'):
            pos = text.rfind(sep)
            if pos > 60:
                text = text[:pos + 1]
                break
    return text


def save_scenario_wav(turns: List[TurnResult], path: str):
    """Concatenate all TTS audio from a scenario into one WAV file."""
    import audioop
    all_pcm = bytearray()
    for t in turns:
        if t.tts_bytes > 0:
            # We stored raw bytes count but need the actual data
            pass  # audio saved separately per-turn
    # We'll save per-turn instead


async def run_scenario(scenario: dict, scenario_idx: int) -> ScenarioResult:
    """Run one full dialogue scenario."""
    result = ScenarioResult(name=scenario["name"])
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    scenario_audio = bytearray()

    for turn_idx, caller_text in enumerate(scenario["turns"], 1):
        # 1. RAG retrieval
        rag_text = retrieve(caller_text)

        # 2. LLM response
        messages.append({"role": "user", "content": caller_text})
        agent_text = llm_generate(messages, rag_text)
        messages.append({"role": "assistant", "content": agent_text})

        # 3. TTS synthesis
        tts_audio, tts_latency = await tts_synthesize(agent_text) if agent_text else (b"", 0)

        # 4. Analyze audio
        audio_stats = analyze_ulaw(tts_audio)
        scenario_audio.extend(tts_audio)

        # 5. Quality checks
        issues = []
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in agent_text)
        if has_chinese:
            issues.append("CHINESE")
        if len(agent_text) > 200:
            issues.append(f"LONG({len(agent_text)})")
        if not agent_text:
            issues.append("EMPTY")
        if audio_stats["clipping_pct"] > 0.05:
            issues.append(f"CLIP({audio_stats['clipping_pct']:.1f}%)")
        if audio_stats["rms"] < 1000 and tts_audio:
            issues.append(f"QUIET(rms={audio_stats['rms']:.0f})")

        turn = TurnResult(
            caller_text=caller_text,
            agent_text=agent_text,
            rag_hit=bool(rag_text),
            rag_preview=rag_text[:60] + "..." if rag_text and len(rag_text) > 60 else (rag_text or ""),
            tts_latency_ms=tts_latency,
            tts_bytes=len(tts_audio),
            tts_duration_sec=audio_stats["duration_sec"],
            audio_rms=audio_stats["rms"],
            audio_clipping_pct=audio_stats["clipping_pct"],
            issues=issues,
        )
        result.turns.append(turn)

    # Save combined audio
    if scenario_audio:
        import audioop
        wav_path = os.path.join(OUTPUT_DIR, f"scenario_{scenario_idx}_{scenario['name'].replace(' ', '_')}.wav")
        pcm = audioop.ulaw2lin(bytes(scenario_audio), 2)
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(8000)
            wf.writeframes(pcm)
        result.wav_path = wav_path

    return result


async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("FULL CALL CYCLE TEST — 5 scenarios, RAG+LLM+TTS")
    print("=" * 70)

    all_results: List[ScenarioResult] = []
    total_turns = 0
    total_issues = 0
    all_latencies = []
    all_rms = []
    all_clips = []
    all_durations = []

    for idx, scenario in enumerate(SCENARIOS, 1):
        print(f"\n{'─' * 70}")
        print(f"  СЦЕНАРИЙ {idx}: {scenario['name']}")
        print(f"{'─' * 70}")

        result = await run_scenario(scenario, idx)
        all_results.append(result)

        for t in result.turns:
            total_turns += 1
            tag = ",".join(t.issues) if t.issues else "OK"
            if t.issues:
                total_issues += 1
            all_latencies.append(t.tts_latency_ms)
            if t.audio_rms > 0:
                all_rms.append(t.audio_rms)
            all_clips.append(t.audio_clipping_pct)
            all_durations.append(t.tts_duration_sec)

            print(f"  Клиент: {t.caller_text}")
            print(f"  Анна:   {t.agent_text[:100]}{'...' if len(t.agent_text) > 100 else ''}")
            print(f"    RAG={'HIT' if t.rag_hit else 'MISS'} | TTS={t.tts_latency_ms:.0f}ms {t.tts_duration_sec:.1f}s | RMS={t.audio_rms:.0f} | [{tag}]")
            print()

        if result.wav_path:
            print(f"  📁 Audio: {result.wav_path}")

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("ИТОГОВЫЙ АНАЛИЗ")
    print(f"{'=' * 70}")

    avg_lat = sum(all_latencies) / len(all_latencies) if all_latencies else 0
    avg_rms = sum(all_rms) / len(all_rms) if all_rms else 0
    avg_clip = sum(all_clips) / len(all_clips) if all_clips else 0
    avg_dur = sum(all_durations) / len(all_durations) if all_durations else 0
    clean_pct = (total_turns - total_issues) / total_turns * 100 if total_turns else 0

    print(f"\n  Сценариев: {len(SCENARIOS)}")
    print(f"  Ходов: {total_turns}")
    print(f"  Чистых ответов: {total_turns - total_issues}/{total_turns} ({clean_pct:.0f}%)")
    print(f"\n  TTS задержка:  avg={avg_lat:.0f}ms  min={min(all_latencies):.0f}ms  max={max(all_latencies):.0f}ms")
    print(f"  TTS длительность: avg={avg_dur:.1f}s")
    print(f"  Audio RMS:     avg={avg_rms:.0f}  min={min(all_rms):.0f}  max={max(all_rms):.0f}")
    print(f"  Клиппинг:      avg={avg_clip:.2f}%  max={max(all_clips):.2f}%")

    # ── Issue analysis ──────────────────────────────────────────────────
    issue_counts = {}
    for r in all_results:
        for t in r.turns:
            for iss in t.issues:
                key = iss.split("(")[0]
                issue_counts[key] = issue_counts.get(key, 0) + 1

    if issue_counts:
        print(f"\n  Проблемы:")
        for k, v in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"    {k}: {v} раз")

    # ── Conversation quality ────────────────────────────────────────────
    print(f"\n  Качество диалога:")
    for r in all_results:
        texts = [t.agent_text for t in r.turns if t.agent_text]
        avg_len = sum(len(t) for t in texts) / len(texts) if texts else 0
        rag_hits = sum(1 for t in r.turns if t.rag_hit)
        print(f"    {r.name}: {len(r.turns)} ходов, avg_len={avg_len:.0f}, RAG={rag_hits}/{len(r.turns)}")

    # ── Recommendations ─────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ:")
    print(f"{'─' * 70}")

    recs = []
    if avg_clip > 0.05:
        recs.append("  - КЛИППИНГ: снизить gain в sox compand (уменьшить norm до -4)")
    if avg_rms < 3000:
        recs.append("  - ТИХИЙ ЗВУК: увеличить norm в sox до -2")
    if avg_lat > 200:
        recs.append("  - ЗАДЕРЖКА TTS: рассмотреть генерацию на 24kHz вместо 48kHz")
    if issue_counts.get("CHINESE", 0) > 0:
        recs.append("  - КИТАЙСКИЙ: усилить repeat_penalty до 1.5 или сменить модель")
    if issue_counts.get("LONG", 0) > 0:
        recs.append("  - ДЛИННЫЕ ОТВЕТЫ: уменьшить max_tokens до 50")
    if issue_counts.get("EMPTY", 0) > 0:
        recs.append("  - ПУСТЫЕ ОТВЕТЫ: проверить промпт LLM")
    if avg_dur > 5:
        recs.append("  - ДЛИННАЯ РЕЧЬ: увеличить tempo до 1.15")
    if clean_pct == 100 and avg_clip == 0 and avg_lat < 150:
        recs.append("  ✓ Все показатели в норме!")

    for rec in recs:
        print(rec)

    print(f"\n  Аудио файлы: {OUTPUT_DIR}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    asyncio.run(main())
