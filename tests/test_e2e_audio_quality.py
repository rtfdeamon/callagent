#!/usr/bin/env python3
"""End-to-end audio quality test bench.

1. Generates TTS µ-law audio from local_ai_server
2. Starts a minimal AudioSocket server on port 8091
3. Originates a Local channel that:
   - Answers, enables MixMonitor (records to WAV)
   - Connects to AudioSocket (receives TTS audio)
4. Waits for playback, then hangs up
5. Analyzes the MixMonitor recording

This tests the full audio path: TTS → AudioSocket → Asterisk → WAV recording
(same path as TTS → AudioSocket → Asterisk → RTP → SIP phone, but recorded)
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
import wave


SUDO_PASS = "mmvsadmin"
RECORDING_PATH = "/tmp/ai_test.wav"
TTS_SERVER_URI = "ws://127.0.0.1:8765"
AUDIOSOCKET_PORT = 8091
FRAME_SIZE = 160  # 160 bytes = 20ms µ-law at 8kHz

# AudioSocket TLV
TYPE_UUID = 0x01
TYPE_AUDIO = 0x10
TYPE_TERMINATE = 0x00
TYPE_ERROR = 0xFF


def asterisk_rx(cmd: str) -> str:
    result = subprocess.run(
        ["sudo", "-S", "asterisk", "-rx", cmd],
        input=f"{SUDO_PASS}\n", capture_output=True, text=True, timeout=10,
    )
    return result.stdout.strip()


async def fetch_tts_ulaw(text: str) -> bytes:
    import websockets
    call_id = f"e2e-{int(time.time())}"
    async with websockets.connect(TTS_SERVER_URI, ping_interval=None, max_size=None) as ws:
        await ws.send(json.dumps({"type": "set_mode", "mode": "tts", "call_id": call_id}))
        deadline = time.time() + 5
        while time.time() < deadline:
            msg = await ws.recv()
            if isinstance(msg, str):
                d = json.loads(msg)
                if d.get("type") == "mode_ready":
                    break
        await ws.send(json.dumps({"type": "tts_request", "mode": "tts", "call_id": call_id, "text": text}))
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                return msg
            d = json.loads(msg)
            if d.get("type") == "tts_response" and d.get("audio_data"):
                import base64
                return base64.b64decode(d["audio_data"])
    return b""


def make_tlv(msg_type: int, payload: bytes) -> bytes:
    return bytes([msg_type]) + len(payload).to_bytes(2, "big") + payload


async def handle_client(reader, writer, ulaw_audio: bytes, done_event: asyncio.Event):
    """Handle one AudioSocket connection from Asterisk.

    app_audiosocket uses the channel's native format (SLIN for Local channels).
    We must decode µ-law → PCM16 (SLIN) before sending.
    SLIN frame: 320 bytes = 160 samples × 2 bytes = 20ms at 8kHz.
    """
    import audioop

    peer = writer.get_extra_info("peername")
    print(f"  [AS] Connection from {peer}")

    # UUID handshake
    header = await reader.readexactly(3)
    length = int.from_bytes(header[1:], "big")
    _ = await reader.readexactly(length) if length else b""
    print(f"  [AS] UUID received ({length} bytes)")

    # Decode µ-law → SLIN (PCM16 LE) for app_audiosocket
    slin_audio = audioop.ulaw2lin(ulaw_audio, 2)
    slin_frame_size = 320  # 160 samples × 2 bytes = 20ms at 8kHz

    await asyncio.sleep(0.3)

    # Silence prefix (300ms) — SLIN silence = zeros
    silence = b"\x00" * slin_frame_size
    next_tick = time.perf_counter()
    for _ in range(15):
        writer.write(make_tlv(TYPE_AUDIO, silence))
        await writer.drain()
        next_tick += 0.020
        s = next_tick - time.perf_counter()
        if s > 0:
            await asyncio.sleep(s)

    # Send SLIN audio at 20ms cadence
    frames_sent = 0
    start = time.perf_counter()
    for offset in range(0, len(slin_audio), slin_frame_size):
        chunk = slin_audio[offset:offset + slin_frame_size]
        if len(chunk) < slin_frame_size:
            chunk += b"\x00" * (slin_frame_size - len(chunk))
        writer.write(make_tlv(TYPE_AUDIO, chunk))
        await writer.drain()
        frames_sent += 1
        next_tick += 0.020
        s = next_tick - time.perf_counter()
        if s > 0:
            await asyncio.sleep(s)

    elapsed = time.perf_counter() - start
    expected = frames_sent * 0.020
    drift = ((elapsed - expected) / expected) * 100 if expected > 0 else 0
    print(f"  [AS] Sent {frames_sent} frames in {elapsed:.3f}s (drift {drift:+.1f}%)")

    # Silence tail (500ms)
    for _ in range(25):
        writer.write(make_tlv(TYPE_AUDIO, silence))
        await writer.drain()
        next_tick += 0.020
        s = next_tick - time.perf_counter()
        if s > 0:
            await asyncio.sleep(s)

    # Read remaining inbound audio for 1s then signal done
    try:
        await asyncio.wait_for(reader.read(32768), timeout=1.0)
    except Exception:
        pass

    writer.close()
    done_event.set()
    print(f"  [AS] Done, connection closed")


def analyze_wav(path: str) -> dict:
    result = {}
    try:
        proc = subprocess.run(["sox", path, "-n", "stat"], capture_output=True, text=True, timeout=10)
        result["sox_stat"] = proc.stderr.strip()
        for line in proc.stderr.splitlines():
            line = line.strip()
            if "RMS" in line and "amplitude" in line:
                result["rms_amplitude"] = float(line.split()[-1])
            elif "Maximum amplitude" in line:
                result["max_amplitude"] = float(line.split()[-1])
            elif "Length (seconds)" in line:
                result["duration_sec"] = float(line.split()[-1])
    except Exception as e:
        result["sox_error"] = str(e)

    try:
        import array, math
        with wave.open(path, "rb") as wf:
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            fr = wf.getframerate()
            nf = wf.getnframes()
            pcm = wf.readframes(nf)
        result["wav_info"] = {"channels": ch, "sample_width": sw, "frame_rate": fr, "frames": nf}

        buf = array.array("h")
        if ch == 2 and sw == 2:
            mono = bytearray()
            for i in range(0, len(pcm), 4):
                mono.extend(pcm[i:i+2])
            buf.frombytes(bytes(mono))
        else:
            buf.frombytes(pcm[:len(pcm) - len(pcm) % 2])

        if len(buf) > 0:
            rms = int(math.sqrt(sum(float(s)**2 for s in buf) / len(buf)))
            result["pcm_rms"] = rms
            result["pcm_peak"] = max(abs(s) for s in buf)

            frame_samples = max(1, fr // 50)
            silence_frames = audio_frames = 0
            for i in range(0, len(buf) - frame_samples, frame_samples):
                frame = buf[i:i + frame_samples]
                frms = math.sqrt(sum(float(s)**2 for s in frame) / len(frame))
                if frms < 100:
                    silence_frames += 1
                else:
                    audio_frames += 1
            total = max(1, silence_frames + audio_frames)
            result["silence_frames"] = silence_frames
            result["audio_frames"] = audio_frames
            result["audio_ratio"] = round(audio_frames / total, 3)

            for i in range(0, len(buf) - frame_samples, frame_samples):
                frame = buf[i:i + frame_samples]
                frms = math.sqrt(sum(float(s)**2 for s in frame) / len(frame))
                if frms > 100:
                    result["first_audio_ms"] = round(i / fr * 1000)
                    break

            big_jumps = 0
            max_delta = 0
            for j in range(1, len(buf)):
                d = abs(buf[j] - buf[j-1])
                if d > max_delta:
                    max_delta = d
                if d > 15000:
                    big_jumps += 1
            result["max_sample_delta"] = max_delta
            result["big_jumps_gt15k"] = big_jumps
    except Exception as e:
        result["analysis_error"] = str(e)
    return result


async def run_test(duration: int):
    print("=" * 60)
    print("E2E Audio Quality Test Bench")
    print("=" * 60)

    # Step 1: Generate TTS audio
    text = "Добрый день! Меня зовут Анна, менеджер компании Мультимедиа Видеосистемы. Чем могу вам помочь?"
    print(f"\n[TTS] Generating audio: {text[:50]}...")
    ulaw_audio = await fetch_tts_ulaw(text)
    if not ulaw_audio:
        print("[ERROR] No TTS audio")
        return
    print(f"[TTS] Got {len(ulaw_audio)} bytes ({len(ulaw_audio)/8000:.2f}s)")

    # Save raw TTS for comparison
    with open("/tmp/e2e_tts_input.ulaw", "wb") as f:
        f.write(ulaw_audio)

    # Step 2: Start AudioSocket server
    done_event = asyncio.Event()

    async def on_connect(reader, writer):
        await handle_client(reader, writer, ulaw_audio, done_event)

    server = await asyncio.start_server(on_connect, "127.0.0.1", AUDIOSOCKET_PORT)
    print(f"\n[AS] AudioSocket server listening on 127.0.0.1:{AUDIOSOCKET_PORT}")

    # Step 3: Clean old recording (may be owned by asterisk user)
    if os.path.exists(RECORDING_PATH):
        subprocess.run(
            ["sudo", "-S", "rm", "-f", RECORDING_PATH],
            input=f"{SUDO_PASS}\n", text=True, timeout=5,
        )

    # Step 4: Originate test call
    print(f"\n[CALL] Originating Local/s@test-ai-record...")
    resp = asterisk_rx("channel originate Local/s@test-ai-record application Wait 120")
    if "error" in resp.lower():
        print(f"[ERROR] {resp}")
        server.close()
        return
    print(f"[CALL] Originated OK")

    # Step 5: Wait for AudioSocket connection and playback
    print(f"[WAIT] Waiting for AudioSocket connection...")
    try:
        await asyncio.wait_for(done_event.wait(), timeout=duration + 10)
        print(f"[CALL] Audio playback complete")
    except asyncio.TimeoutError:
        print(f"[WARN] Timeout — AudioSocket may not have connected")

    # Wait for MixMonitor to flush
    await asyncio.sleep(2)

    # Hangup channels
    print(f"\n[CALL] Hanging up...")
    channels = asterisk_rx("core show channels concise")
    for line in channels.splitlines():
        parts = line.split("!")
        if parts and "Local" in parts[0]:
            asterisk_rx(f"channel request hangup {parts[0]}")

    await asyncio.sleep(1)
    server.close()

    # Step 6: Copy recording to readable location (may be owned by asterisk)
    readable_path = "/tmp/ai_test_copy.wav"
    subprocess.run(
        ["sudo", "-S", "cp", RECORDING_PATH, readable_path],
        input=f"{SUDO_PASS}\n", text=True, timeout=5,
    )
    subprocess.run(
        ["sudo", "-S", "chmod", "644", readable_path],
        input=f"{SUDO_PASS}\n", text=True, timeout=5,
    )

    print("\n" + "=" * 60)
    print("RECORDING ANALYSIS")
    print("=" * 60)

    if not os.path.exists(readable_path) or os.path.getsize(readable_path) < 100:
        print("[ERROR] No recording captured!")
        print("Check: asterisk -rx 'dialplan show s@test-ai-record'")
        return

    r = analyze_wav(readable_path)
    dur = r.get("duration_sec", 0)
    rms = r.get("rms_amplitude", 0)
    ratio = r.get("audio_ratio", 0)
    jumps = r.get("big_jumps_gt15k", 0)

    print(f"\nFile: {readable_path} ({os.path.getsize(readable_path)} bytes)")
    print(f"Duration:        {dur:.2f}s")
    wi = r.get("wav_info", {})
    print(f"Format:          {wi.get('frame_rate')}Hz {wi.get('channels')}ch {wi.get('sample_width','?')*8}bit")
    print(f"RMS amplitude:   {rms}")
    print(f"Max amplitude:   {r.get('max_amplitude', '?')}")
    print(f"PCM RMS:         {r.get('pcm_rms', '?')}")
    print(f"PCM Peak:        {r.get('pcm_peak', '?')}")
    print(f"Audio frames:    {r.get('audio_frames', '?')} (silence: {r.get('silence_frames', '?')})")
    print(f"Audio ratio:     {ratio}")
    print(f"First audio at:  {r.get('first_audio_ms', '?')} ms")
    print(f"Max delta:       {r.get('max_sample_delta', '?')}")
    print(f"Big jumps >15k:  {jumps}")

    # Compare with TTS input
    print("\n" + "-" * 40)
    print("TTS INPUT vs RECORDED OUTPUT")
    print("-" * 40)
    tts_duration = len(ulaw_audio) / 8000
    print(f"TTS input:   {tts_duration:.2f}s ({len(ulaw_audio)} bytes µ-law)")
    print(f"Recorded:    {dur:.2f}s")
    if dur > 0:
        time_ratio = dur / tts_duration
        print(f"Time ratio:  {time_ratio:.2f}x {'(OK)' if 0.8 < time_ratio < 1.5 else '(PROBLEM!)'}")

    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    issues = []
    if ratio < 0.1:
        issues.append("CRITICAL: Almost no audio — AudioSocket→Asterisk path broken")
    elif ratio < 0.3:
        issues.append("WARNING: Low audio ratio — possible silence gaps")
    if isinstance(rms, (int, float)) and rms < 0.005:
        issues.append("CRITICAL: Very low RMS — audio nearly silent")
    if isinstance(dur, (int, float)) and isinstance(jumps, (int, float)) and dur > 0:
        jps = jumps / dur
        if jps > 100:
            issues.append(f"WARNING: High discontinuity rate ({jps:.0f}/sec) — crackling likely")
        elif jps > 30:
            issues.append(f"INFO: Moderate discontinuities ({jps:.0f}/sec) — may sound slightly rough")
    if not issues:
        print("  PASS: Audio chain looks clean")
    else:
        for i in issues:
            print(f"  {i}")

    # Copy to workspace
    workspace_dir = "/home/dmitriy/work/callagent/logs/e2e-test"
    os.makedirs(workspace_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(workspace_dir, f"e2e_{ts}.wav")
    shutil.copy2(readable_path, dest)
    with open(dest.replace(".wav", ".json"), "w") as f:
        json.dump(r, f, indent=2, default=str)
    print(f"\nRecording: {dest}")
    print(f"Listen:    aplay {dest}")
    print(f"Or:        sox {dest} -d")
    print(f"Compare with TTS input: sox -t raw -r 8000 -e mu-law -b 8 -c 1 /tmp/e2e_tts_input.ulaw -d")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=15)
    args = parser.parse_args()
    asyncio.run(run_test(args.duration))


if __name__ == "__main__":
    main()
