#!/usr/bin/env python3
"""Test Silero TTS via local_ai_server WebSocket API."""
import asyncio
import json
import sys
import time

import websockets


WS_URL = "ws://127.0.0.1:8765"

TEST_PHRASES = [
    "Добрый день! Компания Мультимедиа Видеосистемы, меня зовут Анна.",
    "Мы предлагаем интерактивные панели и проекторы для школ и музеев.",
    "Давайте я подготовлю для вас коммерческое предложение.",
]


async def test_tts():
    """Send TTS requests and measure latency."""
    print("=" * 60)
    print("Silero TTS Test via WebSocket")
    print("=" * 60)

    try:
        ws = await websockets.connect(WS_URL, ping_interval=None)
    except Exception as e:
        print(f"FAIL: Cannot connect to {WS_URL}: {e}")
        sys.exit(1)

    results = []

    for i, text in enumerate(TEST_PHRASES, 1):
        print(f"\n--- Phrase {i}/{len(TEST_PHRASES)} ---")
        print(f"Text: {text}")

        t0 = time.monotonic()
        await ws.send(json.dumps({
            "type": "tts_request",
            "text": text,
            "call_id": "test_silero_tts",
        }))

        audio_bytes = b""
        try:
            resp = await asyncio.wait_for(ws.recv(), timeout=10)
            elapsed = time.monotonic() - t0

            if isinstance(resp, bytes):
                audio_bytes = resp
            else:
                import base64
                msg = json.loads(resp)
                if msg.get("type") == "tts_response" and msg.get("audio_data"):
                    audio_bytes = base64.b64decode(msg["audio_data"])
                else:
                    print(f"  Unexpected response: {resp[:200]}")
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - t0
            print(f"  TIMEOUT after {elapsed:.2f}s")
            continue

        duration_sec = len(audio_bytes) / 8000 if audio_bytes else 0
        print(f"  Audio: {len(audio_bytes)} bytes, ~{duration_sec:.2f}s playback")
        print(f"  Latency: {elapsed*1000:.0f}ms")

        results.append({
            "phrase": text,
            "audio_bytes": len(audio_bytes),
            "duration_sec": duration_sec,
            "latency_ms": elapsed * 1000,
        })

        # Save first phrase as WAV for manual inspection
        if i == 1 and audio_bytes:
            wav_path = "/tmp/silero_test_output.wav"
            # ulaw 8kHz -> write as raw for inspection
            with open("/tmp/silero_test_output.ulaw", "wb") as f:
                f.write(audio_bytes)
            print(f"  Saved raw ulaw to /tmp/silero_test_output.ulaw")

    await ws.close()

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    if results:
        avg_latency = sum(r["latency_ms"] for r in results) / len(results)
        total_audio = sum(r["duration_sec"] for r in results)
        print(f"  Phrases tested: {len(results)}")
        print(f"  Avg latency:    {avg_latency:.0f}ms")
        print(f"  Total audio:    {total_audio:.1f}s")
        all_ok = all(r["audio_bytes"] > 0 for r in results)
        print(f"  Status:         {'PASS' if all_ok else 'FAIL'}")
        if not all_ok:
            sys.exit(1)
    else:
        print("  No results — FAIL")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(test_tts())
