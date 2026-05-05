#!/usr/bin/env python3
"""Minimal AudioSocket server for audio quality isolation testing.

Usage:
  python tests/test_audiosocket_minimal.py

This starts a bare-bones AudioSocket server on port 8091.
When Asterisk connects (via extension 779), it:
1. Reads the UUID handshake
2. Generates clean µ-law sine wave tone (400 Hz, 3 seconds)
3. Sends it frame-by-frame at proper 20ms cadence
4. Closes

Call sip:779@127.0.0.1 from Linphone to test.
If this sounds clean → AudioSocket path works, problem is in the engine.
If this crackles → problem is in Asterisk/SIP/codec config.
"""

import asyncio
import math
import struct
import sys
import time

# AudioSocket TLV types
TYPE_UUID = 0x01
TYPE_AUDIO = 0x10

# µ-law constants
SAMPLE_RATE = 8000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 160 samples
TONE_HZ = 400
DURATION_SEC = 3


def pcm16_to_ulaw(sample: int) -> int:
    """Convert a single PCM16 sample to µ-law byte."""
    BIAS = 0x84
    MAX = 0x7FFF
    CLIP = 32635

    sign = 0
    if sample < 0:
        sign = 0x80
        sample = -sample
    if sample > CLIP:
        sample = CLIP
    sample += BIAS

    exponent = 7
    mask = 0x4000
    while exponent > 0 and not (sample & mask):
        exponent -= 1
        mask >>= 1

    mantissa = (sample >> (exponent + 3)) & 0x0F
    ulaw_byte = ~(sign | (exponent << 4) | mantissa) & 0xFF
    return ulaw_byte


def generate_ulaw_tone(freq_hz: float, duration_sec: float) -> bytes:
    """Generate a pure sine wave tone encoded as µ-law."""
    num_samples = int(SAMPLE_RATE * duration_sec)
    result = bytearray(num_samples)
    amplitude = 16000  # ~50% of max to avoid clipping

    for i in range(num_samples):
        t = i / SAMPLE_RATE
        pcm = int(amplitude * math.sin(2 * math.pi * freq_hz * t))
        result[i] = pcm16_to_ulaw(pcm)

    return bytes(result)


def make_tlv(msg_type: int, payload: bytes) -> bytes:
    """Create AudioSocket TLV frame."""
    return bytes([msg_type]) + len(payload).to_bytes(2, "big") + payload


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    """Handle one AudioSocket connection from Asterisk."""
    peer = writer.get_extra_info("peername")
    print(f"[CONNECT] {peer}")

    # Read UUID handshake
    try:
        header = await asyncio.wait_for(reader.readexactly(3), timeout=5.0)
        msg_type = header[0]
        length = int.from_bytes(header[1:3], "big")
        if length > 0:
            payload = await reader.readexactly(length)
        print(f"[UUID] type=0x{msg_type:02x} len={length}")
    except Exception as e:
        print(f"[ERROR] UUID handshake failed: {e}")
        writer.close()
        return

    # Generate clean µ-law tone
    print(f"[TONE] Generating {TONE_HZ}Hz tone, {DURATION_SEC}s, µ-law 8kHz...")
    ulaw_audio = generate_ulaw_tone(TONE_HZ, DURATION_SEC)
    print(f"[TONE] {len(ulaw_audio)} bytes ({len(ulaw_audio)/SAMPLE_RATE:.2f}s)")

    # Small initial delay for channel setup
    await asyncio.sleep(0.2)

    # Send frames at exact 20ms cadence
    frame_count = 0
    next_tick = time.perf_counter()

    for offset in range(0, len(ulaw_audio), FRAME_SAMPLES):
        chunk = ulaw_audio[offset:offset + FRAME_SAMPLES]
        if len(chunk) < FRAME_SAMPLES:
            # Pad last frame with µ-law silence (0xFF)
            chunk = chunk + b"\xFF" * (FRAME_SAMPLES - len(chunk))

        writer.write(make_tlv(TYPE_AUDIO, chunk))
        await writer.drain()
        frame_count += 1

        next_tick += FRAME_MS / 1000.0
        sleep_time = next_tick - time.perf_counter()
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)

    elapsed = time.perf_counter() - (next_tick - FRAME_MS / 1000.0 * frame_count + FRAME_MS / 1000.0)
    print(f"[SENT] {frame_count} frames in {elapsed:.3f}s")

    # Tail silence (500ms)
    silence_frame = b"\xFF" * FRAME_SAMPLES
    for _ in range(25):
        writer.write(make_tlv(TYPE_AUDIO, silence_frame))
        await writer.drain()
        next_tick += FRAME_MS / 1000.0
        sleep_time = next_tick - time.perf_counter()
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)

    # Wait briefly then close
    await asyncio.sleep(0.5)
    writer.close()
    print(f"[DONE] Connection closed")


async def main():
    port = 8091
    server = await asyncio.start_server(handle_client, "127.0.0.1", port)
    print(f"=== Minimal AudioSocket Test Server ===")
    print(f"Listening on 127.0.0.1:{port}")
    print(f"Call sip:779@127.0.0.1 from Linphone to test")
    print(f"Press Ctrl+C to stop")
    print()

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
