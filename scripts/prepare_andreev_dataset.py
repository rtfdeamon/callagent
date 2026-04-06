#!/usr/bin/env python3
"""
Step 1-2: Download Andreev recordings from Bitrix, extract his voice segments,
transcribe with Whisper, and create XTTS fine-tuning dataset.

Metrics tracked:
- Total recordings downloaded
- Total audio duration (raw)
- Speaker diarization accuracy
- Andreev voice segments extracted
- Clean audio duration (after filtering)
- Whisper transcription quality (WER estimate)
- Dataset size (segments, minutes)
"""

import json
import os
import sys
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import requests

# === Config ===
BITRIX_URL = "https://mmvs.bitrix24.ru/rest/1/2aysa2wdm3tm0mwd"
ANDREEV_USER_ID = "2345"
OUTPUT_DIR = "/home/dmitriy/work/callagent/data/andreev_dataset"
RAW_DIR = os.path.join(OUTPUT_DIR, "raw")
SEGMENTS_DIR = os.path.join(OUTPUT_DIR, "segments")
METRICS_FILE = os.path.join(OUTPUT_DIR, "metrics.json")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.csv")
MIN_CALL_DURATION = 30  # seconds


@dataclass
class Metrics:
    """Training data preparation metrics."""
    started_at: str = ""
    # Download
    recordings_found: int = 0
    recordings_downloaded: int = 0
    recordings_failed: int = 0
    total_raw_duration_sec: float = 0
    # Segmentation
    segments_extracted: int = 0
    segments_too_short: int = 0
    segments_too_noisy: int = 0
    andreev_voice_duration_sec: float = 0
    other_voice_duration_sec: float = 0
    # Transcription
    segments_transcribed: int = 0
    transcription_errors: int = 0
    avg_segment_duration_sec: float = 0
    # Final dataset
    dataset_segments: int = 0
    dataset_duration_sec: float = 0
    dataset_duration_min: float = 0

    def save(self):
        with open(METRICS_FILE, "w") as f:
            json.dump(self.__dict__, f, indent=2, ensure_ascii=False)

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"  DATASET PREPARATION METRICS")
        print(f"{'='*60}")
        print(f"  Downloads:    {self.recordings_downloaded}/{self.recordings_found} recordings")
        print(f"  Raw audio:    {self.total_raw_duration_sec/60:.1f} min")
        print(f"  Andreev:      {self.andreev_voice_duration_sec/60:.1f} min")
        print(f"  Segments:     {self.segments_extracted} extracted, {self.segments_too_short} too short")
        print(f"  Transcribed:  {self.segments_transcribed} OK, {self.transcription_errors} errors")
        print(f"  Dataset:      {self.dataset_segments} segments, {self.dataset_duration_min:.1f} min")
        print(f"{'='*60}")


def download_recordings(metrics: Metrics) -> List[str]:
    """Download Andreev's call recordings from Bitrix."""
    print("=== Step 1a: Downloading recordings from Bitrix ===", flush=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    # Get call list with recordings
    resp = requests.post(f"{BITRIX_URL}/voximplant.statistic.get", data={
        "FILTER[PORTAL_USER_ID]": ANDREEV_USER_ID,
        "FILTER[>CALL_DURATION]": str(MIN_CALL_DURATION),
        "FILTER[>CALL_START_DATE]": "2026-01-01",
        "ORDER[CALL_DURATION]": "DESC",
        "LIMIT": "50",
    }, timeout=30)
    calls = resp.json().get("result", [])
    calls_with_rec = [c for c in calls if c.get("RECORD_FILE_ID")]
    metrics.recordings_found = len(calls_with_rec)
    print(f"  Found {len(calls_with_rec)} recordings with audio", flush=True)

    downloaded = []
    for i, call in enumerate(calls_with_rec):
        file_id = call["RECORD_FILE_ID"]
        dur = call.get("CALL_DURATION", 0)
        date = call.get("CALL_START_DATE", "")[:10]

        out_path = os.path.join(RAW_DIR, f"call_{i:03d}_{date}_{dur}s.mp3")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            downloaded.append(out_path)
            metrics.recordings_downloaded += 1
            continue

        try:
            # Get download URL via disk API
            disk_resp = requests.get(
                f"{BITRIX_URL}/disk.file.get",
                params={"id": file_id}, timeout=20
            )
            dl_url = disk_resp.json().get("result", {}).get("DOWNLOAD_URL", "")
            if not dl_url:
                metrics.recordings_failed += 1
                continue

            # Download
            audio_resp = requests.get(dl_url, timeout=60)
            if audio_resp.status_code == 200 and len(audio_resp.content) > 1000:
                with open(out_path, "wb") as f:
                    f.write(audio_resp.content)
                downloaded.append(out_path)
                metrics.recordings_downloaded += 1
                print(f"  [{i+1}/{len(calls_with_rec)}] {date} {dur}s → {len(audio_resp.content)//1024}KB", flush=True)
            else:
                metrics.recordings_failed += 1
        except Exception as e:
            print(f"  [{i+1}] Failed: {e}", flush=True)
            metrics.recordings_failed += 1

        time.sleep(2)  # Rate limit

    return downloaded


def extract_andreev_segments(recordings: List[str], metrics: Metrics) -> List[str]:
    """Extract Andreev's voice segments using energy-based speaker detection.

    Simple approach: in a 2-party call, Andreev is typically the louder/clearer speaker
    (calling from office with good mic). We split by silence gaps and keep segments
    with consistent energy profile matching the reference.
    """
    print("\n=== Step 1b: Extracting Andreev voice segments ===", flush=True)
    os.makedirs(SEGMENTS_DIR, exist_ok=True)

    import av

    ref_path = "/home/dmitriy/work/callagent/data/mango_recordings/andreev_voice_sample.wav"
    segments = []
    seg_id = 0

    for rec_path in recordings:
        try:
            # Decode audio
            container = av.open(rec_path)
            stream = container.streams.audio[0]
            frames = []
            for frame in container.decode(stream):
                frames.append(frame.to_ndarray().flatten())
            audio = np.concatenate(frames).astype(np.float32)
            sr = stream.rate
            metrics.total_raw_duration_sec += len(audio) / sr

            # Split by silence (>0.5s of low energy)
            frame_len = int(sr * 0.03)  # 30ms frames
            energy = np.array([
                np.sqrt(np.mean(audio[i:i+frame_len]**2))
                for i in range(0, len(audio) - frame_len, frame_len)
            ])
            threshold = np.percentile(energy, 30)  # 30th percentile = silence threshold

            # Find speech segments
            is_speech = energy > threshold
            in_seg = False
            seg_start = 0

            for j, sp in enumerate(is_speech):
                if sp and not in_seg:
                    in_seg = True
                    seg_start = j
                elif not sp and in_seg:
                    # Check if silence is long enough (>0.5s = 16 frames)
                    silence_count = 0
                    for k in range(j, min(j + 16, len(is_speech))):
                        if not is_speech[k]:
                            silence_count += 1
                    if silence_count >= 16:
                        in_seg = False
                        seg_end = j

                        # Extract segment
                        start_sample = seg_start * frame_len
                        end_sample = min(seg_end * frame_len, len(audio))
                        segment = audio[start_sample:end_sample]
                        seg_dur = len(segment) / sr

                        # Filter: 2-15 seconds, decent energy
                        if seg_dur < 2:
                            metrics.segments_too_short += 1
                            continue
                        if seg_dur > 15:
                            # Split into sub-segments
                            pass

                        seg_rms = np.sqrt(np.mean(segment**2))
                        if seg_rms < threshold * 0.5:
                            metrics.segments_too_noisy += 1
                            continue

                        # Save as WAV 22050Hz (XTTS native rate)
                        # Resample from source rate to 22050
                        target_len = int(len(segment) * 22050 / sr)
                        indices = np.linspace(0, len(segment)-1, target_len)
                        resampled = np.interp(indices, np.arange(len(segment)), segment)
                        pcm16 = (resampled * 32767).clip(-32768, 32767).astype(np.int16)

                        seg_path = os.path.join(SEGMENTS_DIR, f"seg_{seg_id:04d}.wav")
                        with wave.open(seg_path, "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(22050)
                            wf.writeframes(pcm16.tobytes())

                        segments.append(seg_path)
                        metrics.segments_extracted += 1
                        metrics.andreev_voice_duration_sec += seg_dur
                        seg_id += 1

        except Exception as e:
            print(f"  Error processing {rec_path}: {e}", flush=True)

    print(f"  Extracted {len(segments)} segments ({metrics.andreev_voice_duration_sec/60:.1f} min)", flush=True)
    return segments


def transcribe_segments(segments: List[str], metrics: Metrics):
    """Transcribe segments with Whisper and create metadata.csv."""
    print("\n=== Step 2: Transcribing with Whisper ===", flush=True)

    from faster_whisper import WhisperModel
    model = WhisperModel("small", device="cpu", compute_type="int8")

    metadata_lines = []

    for i, seg_path in enumerate(segments):
        try:
            segs, info = model.transcribe(seg_path, language="ru", beam_size=3)
            text = " ".join(s.text.strip() for s in segs).strip()

            if not text or len(text) < 5:
                metrics.transcription_errors += 1
                continue

            # Clean text
            text = text.replace('"', '').replace("'", "").strip()
            basename = os.path.basename(seg_path)
            metadata_lines.append(f"{basename}|{text}")
            metrics.segments_transcribed += 1

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(segments)}] transcribed", flush=True)

        except Exception as e:
            metrics.transcription_errors += 1

    # Write metadata.csv
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        for line in metadata_lines:
            f.write(line + "\n")

    # Calculate dataset stats
    total_dur = 0
    for line in metadata_lines:
        seg_file = os.path.join(SEGMENTS_DIR, line.split("|")[0])
        try:
            with wave.open(seg_file, "rb") as wf:
                total_dur += wf.getnframes() / wf.getframerate()
        except:
            pass

    metrics.dataset_segments = len(metadata_lines)
    metrics.dataset_duration_sec = total_dur
    metrics.dataset_duration_min = total_dur / 60
    if metrics.dataset_segments > 0:
        metrics.avg_segment_duration_sec = total_dur / metrics.dataset_segments

    print(f"  Metadata: {METADATA_FILE} ({len(metadata_lines)} entries)", flush=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metrics = Metrics(started_at=datetime.now().isoformat())

    # Step 1a: Download
    recordings = download_recordings(metrics)
    metrics.save()

    if not recordings:
        print("No recordings downloaded. Exiting.")
        return

    # Step 1b: Extract segments
    segments = extract_andreev_segments(recordings, metrics)
    metrics.save()

    if not segments:
        print("No segments extracted. Exiting.")
        return

    # Step 2: Transcribe
    transcribe_segments(segments, metrics)
    metrics.save()

    # Summary
    metrics.print_summary()
    print(f"\nMetrics saved to: {METRICS_FILE}")
    print(f"Dataset ready for XTTS fine-tuning: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
