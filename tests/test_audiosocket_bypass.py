#!/usr/bin/env python3
"""Bypass test: sends TTS µ-law directly through AudioSocket to Asterisk.

Usage:
  1. Stop ai_engine: docker compose stop ai_engine
  2. Run this script: python3 tests/test_audiosocket_bypass.py
  3. Call extension 777 from Linphone
  4. Listen — if clean, problem is in StreamingPlaybackManager; if crackling, problem is Asterisk/RTP

The script starts a minimal AudioSocket server on port 8090,
accepts one connection, and sends pre-generated TTS audio at 20ms/frame cadence.
"""

import asyncio
import io
import json
import struct
import sys
import time
import uuid
import wave

# AudioSocket TLV constants
TYPE_UUID = 0x01
TYPE_AUDIO = 0x10
TYPE_ERROR = 0xFF
TYPE_TERMINATE = 0x00

FRAME_SIZE = 160  # 160 bytes = 20ms at 8kHz µ-law
FRAME_INTERVAL = 0.020  # 20ms


async def fetch_tts_ulaw(text: str) -> bytes:
    """Get µ-law 8kHz audio from local_ai_server."""
    try:
        import websockets
    except ImportError:
        print("pip install websockets")
        sys.exit(1)

    call_id = f"bypass-{int(time.time())}"
    uri = "ws://127.0.0.1:8765"

    async with websockets.connect(uri, ping_interval=None, ping_timeout=None, max_size=None) as ws:
        await ws.send(json.dumps({"type": "set_mode", "mode": "tts", "call_id": call_id}))

        deadline = time.time() + 5
        while time.time() < deadline:
            msg = await ws.recv()
            if isinstance(msg, str):
                data = json.loads(msg)
                if data.get("type") == "mode_ready":
                    break

        await ws.send(json.dumps({"type": "tts_request", "mode": "tts", "call_id": call_id, "text": text}))

        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                return msg
            data = json.loads(msg)
            if data.get("type") == "tts_response" and data.get("audio_data"):
                import base64
                return base64.b64decode(data["audio_data"])

    return b""


def make_tlv(msg_type: int, payload: bytes) -> bytes:
    return bytes([msg_type]) + len(payload).to_bytes(2, "big") + payload


async def handle_audiosocket_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, ulaw_audio: bytes):
    """Handle one AudioSocket connection: handshake then play audio."""
    peer = writer.get_extra_info("peername")
    print(f"[AS] Connection from {peer}")

    # Read UUID handshake
    header = await reader.readexactly(3)
    msg_type = header[0]
    length = int.from_bytes(header[1:], "big")
    payload = await reader.readexactly(length) if length else b""

    if msg_type != TYPE_UUID:
        print(f"[AS] Expected UUID, got type={msg_type}")
        writer.close()
        return

    uuid_bytes = payload
    try:
        uuid_str = str(uuid.UUID(bytes=uuid_bytes)) if len(uuid_bytes) == 16 else uuid_bytes.decode()
    except Exception:
        uuid_str = uuid_bytes.hex()
    print(f"[AS] UUID: {uuid_str}")

    # Small delay to let bridge setup complete
    await asyncio.sleep(0.5)

    # Send audio frames at steady 20ms cadence
    total_frames = len(ulaw_audio) // FRAME_SIZE
    print(f"[AS] Sending {total_frames} frames ({total_frames * 20}ms) of µ-law audio...")

    # Add 500ms silence prefix (µ-law silence = 0x7F)
    silence_frames = 25  # 500ms
    silence = b"\x7f" * FRAME_SIZE

    next_tick = time.perf_counter()

    # Send silence prefix
    for i in range(silence_frames):
        frame = make_tlv(TYPE_AUDIO, silence)
        writer.write(frame)
        await writer.drain()
        next_tick += FRAME_INTERVAL
        sleep_for = next_tick - time.perf_counter()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)

    # Send actual audio
    frames_sent = 0
    start_time = time.perf_counter()

    for offset in range(0, len(ulaw_audio), FRAME_SIZE):
        chunk = ulaw_audio[offset:offset + FRAME_SIZE]
        if len(chunk) < FRAME_SIZE:
            # Pad last frame with µ-law silence
            chunk = chunk + b"\x7f" * (FRAME_SIZE - len(chunk))

        frame = make_tlv(TYPE_AUDIO, chunk)
        writer.write(frame)
        await writer.drain()
        frames_sent += 1

        next_tick += FRAME_INTERVAL
        sleep_for = next_tick - time.perf_counter()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)

    elapsed = time.perf_counter() - start_time
    expected = frames_sent * FRAME_INTERVAL
    drift_pct = ((elapsed - expected) / expected) * 100 if expected > 0 else 0

    print(f"[AS] Done! {frames_sent} frames in {elapsed:.3f}s (expected {expected:.3f}s, drift {drift_pct:+.1f}%)")

    # Send 1s of silence tail
    for i in range(50):
        writer.write(make_tlv(TYPE_AUDIO, silence))
        await writer.drain()
        next_tick += FRAME_INTERVAL
        sleep_for = next_tick - time.perf_counter()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)

    print("[AS] Silence tail sent. Keeping connection open for inbound audio...")

    # Keep reading inbound audio (discard) to keep the connection alive
    try:
        while True:
            header = await reader.readexactly(3)
            msg_type = header[0]
            length = int.from_bytes(header[1:], "big")
            if length:
                await reader.readexactly(length)
            if msg_type in (TYPE_TERMINATE, TYPE_ERROR):
                print("[AS] Peer terminated")
                break
    except (asyncio.IncompleteReadError, ConnectionResetError):
        print("[AS] Connection closed")
    finally:
        writer.close()


async def main():
    text = "Добрый день! Меня зовут Анна, я менеджер компании Мультимедиа Видеосистемы. Чем могу вам помочь?"

    print(f"[TTS] Generating µ-law audio for: {text[:60]}...")
    ulaw_audio = await fetch_tts_ulaw(text)
    print(f"[TTS] Got {len(ulaw_audio)} bytes ({len(ulaw_audio)/8000:.2f}s)")

    if not ulaw_audio:
        print("[ERROR] No TTS audio received")
        sys.exit(1)

    # Save for reference
    with open("/tmp/bypass_test.ulaw", "wb") as f:
        f.write(ulaw_audio)

    connected = asyncio.Event()

    async def on_connect(reader, writer):
        connected.set()
        await handle_audiosocket_client(reader, writer, ulaw_audio)

    server = await asyncio.start_server(on_connect, "127.0.0.1", 8090)
    print("[AS] Listening on 127.0.0.1:8090")
    print("[AS] Now call extension 777 from Linphone and listen!")
    print("[AS] Press Ctrl+C to stop")

    try:
        await server.serve_forever()
    except KeyboardInterrupt:
        print("\n[AS] Shutting down")
        server.close()


if __name__ == "__main__":
    asyncio.run(main())
