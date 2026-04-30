#!/usr/bin/env python3
"""
Protocol-level tests for Local AI Server against the current contract in
local_ai_server/main.py and docs/local-ai-server/PROTOCOL.md.

These tests assume the server is reachable at ws://127.0.0.1:8765.
"""

import asyncio
import base64
import json
import os
import sys
import logging
from typing import Optional

import websockets
import pytest
import socket

WS_URL = os.getenv("LOCAL_WS_URL", "ws://127.0.0.1:8765")


def _server_available(url: str) -> bool:
    try:
        host, port = url.replace("ws://", "").replace("wss://", "").split(":")
        with socket.create_connection((host, int(port)), timeout=1.0):
            return True
    except Exception:
        return False

# Mark as integration: requires a running local-ai-server WebSocket
pytestmark = pytest.mark.skipif(
    not _server_available(WS_URL),
    reason="Requires local AI server at 127.0.0.1:8765. Start local_ai_server to enable.",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_tts_roundtrip() -> bool:
    """tts_request → tts_response (JSON с base64 audio_data).

    Контракт: на запрос type=tts_request сервер отвечает одним JSON
    сообщением type=tts_response, где аудио закодировано base64 в
    поле audio_data. Бинарных фреймов в этом режиме не отправляется
    (см. local_ai_server/server.py:3628 и docs/local-ai-server/PROTOCOL.md
    раздел "tts_request").
    """
    async with websockets.connect(WS_URL, max_size=None) as ws:
        req = {
            "type": "tts_request",
            "text": "Hello from protocol test.",
            "call_id": "test-call",
            "request_id": "t1",
        }
        await ws.send(json.dumps(req))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=10.0))
        assert resp["type"] == "tts_response", f"Unexpected type: {resp.get('type')}"
        assert resp["encoding"] == "mulaw"
        assert resp.get("request_id") == "t1"
        assert isinstance(resp.get("audio_data"), str)
        audio_bytes = base64.b64decode(resp["audio_data"])
        logger.info("Received TTS audio bytes (b64-decoded): %s", len(audio_bytes))
        return len(audio_bytes) > 0


async def test_stt_binary_flow() -> bool:
    """STT-режим: проверка формата stt_result.

    Отправляем тишину. Сервер не обязан выдавать is_final=True (нет речи —
    нет окончательного транскрипта), но обязан отдать хотя бы один JSON
    stt_result с корректной формой контракта (поля type/call_id/text).
    """
    async with websockets.connect(WS_URL, max_size=None) as ws:
        await ws.send(json.dumps({"type": "set_mode", "mode": "stt", "call_id": "demo"}))
        # Ожидаем mode_ready, но без assertion — старые версии могут не отвечать
        try:
            ack = await asyncio.wait_for(ws.recv(), timeout=2.0)
            if isinstance(ack, str):
                evt = json.loads(ack)
                if evt.get("type") == "mode_ready":
                    assert evt.get("mode") == "stt"
        except asyncio.TimeoutError:
            pass

        pcm_silence = b"\x00\x00" * 16000
        await ws.send(pcm_silence)

        # Ждём первый stt_result (partial или final). На тишине Vosk обычно
        # выдаёт is_partial=True с пустым текстом — этого достаточно для
        # проверки протокольного контракта.
        for _ in range(3):
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            except asyncio.TimeoutError:
                return False
            if isinstance(msg, (bytes, bytearray)):
                continue
            evt = json.loads(msg)
            if evt.get("type") == "stt_result":
                assert evt.get("call_id") == "demo"
                assert "text" in evt
                assert "is_final" in evt or "is_partial" in evt
                logger.info("STT result OK: %s", evt)
                return True
        return False


async def test_full_audio_frame() -> bool:
    """Full-режим: один JSON аудиокадр должен породить stt_result.

    На тишине LLM и TTS не вызываются (нет распознанного текста), поэтому
    проверяем только что сервер принял кадр и ответил stt_result. Полный
    цикл STT→LLM→TTS требует реальной речи и тестируется в e2e-сценариях.
    """
    async with websockets.connect(WS_URL, max_size=None) as ws:
        pcm = b"\x00\x00" * 16000
        req = {
            "type": "audio",
            "mode": "full",
            "rate": 16000,
            "call_id": "full-test",
            "request_id": "r1",
            "data": base64.b64encode(pcm).decode("utf-8"),
        }
        await ws.send(json.dumps(req))

        for _ in range(5):
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
            except asyncio.TimeoutError:
                return False
            if isinstance(msg, (bytes, bytearray)):
                logger.info("Received early binary frame: %s bytes", len(msg))
                return True
            evt = json.loads(msg)
            if evt.get("type") == "stt_result":
                assert evt.get("call_id") == "full-test"
                logger.info("Full-mode stt_result OK: %s", evt)
                return True
        return False


async def main() -> None:
    ok1 = await test_tts_roundtrip()
    ok2 = await test_stt_binary_flow()
    ok3 = await test_full_audio_frame()
    total = sum([ok1, ok2, ok3])
    print(f"Local AI Server protocol tests passed: {total}/3")
    sys.exit(0 if total == 3 else 1)


if __name__ == "__main__":
    asyncio.run(main())
