#!/usr/bin/env python3
"""Download call recordings from Mango Office API.

Usage:
    python scripts/download_mango_recordings.py --api-key YOUR_KEY --api-salt YOUR_SALT

Requires Mango Office API credentials from:
    Личный кабинет → Настройки → API → Уникальный код АТС + Ключ
"""

import argparse
import hashlib
import hmac
import json
import os
import time
from datetime import datetime, timedelta

import requests

MANGO_API_BASE = "https://app.mango-office.ru/vpbx"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "mango_recordings")


def sign_request(api_key: str, api_salt: str, json_data: str) -> str:
    """Create HMAC-SHA256 signature for Mango API request."""
    sign_str = api_key + json_data + api_salt
    return hashlib.sha256(sign_str.encode()).hexdigest()


def get_call_history(api_key: str, api_salt: str, from_date: datetime, to_date: datetime, extension: str = ""):
    """Get call history from Mango Office."""
    json_data = json.dumps({
        "date_from": int(from_date.timestamp()),
        "date_to": int(to_date.timestamp()),
        "fields": "records,start,finish,from_number,to_number,duration,direction",
    })

    sign = sign_request(api_key, api_salt, json_data)

    resp = requests.post(
        f"{MANGO_API_BASE}/stats/request",
        data={"vpbx_api_key": api_key, "sign": sign, "json": json_data},
        timeout=30,
    )

    if resp.status_code != 200:
        print(f"Error: {resp.status_code} {resp.text[:200]}")
        return None

    result = resp.json()
    key = result.get("key")
    if not key:
        print(f"No key in response: {result}")
        return None

    # Poll for results
    for _ in range(30):
        time.sleep(2)
        json_data2 = json.dumps({"key": key})
        sign2 = sign_request(api_key, api_salt, json_data2)
        resp2 = requests.post(
            f"{MANGO_API_BASE}/stats/result",
            data={"vpbx_api_key": api_key, "sign": sign2, "json": json_data2},
            timeout=30,
        )
        if resp2.status_code == 200:
            return resp2.json()

    print("Timeout waiting for stats results")
    return None


def download_recording(api_key: str, api_salt: str, recording_id: str, output_path: str):
    """Download a single call recording."""
    json_data = json.dumps({
        "recording_id": recording_id,
        "action": "download",
    })

    sign = sign_request(api_key, api_salt, json_data)

    resp = requests.post(
        f"{MANGO_API_BASE}/queries/recording/post",
        data={"vpbx_api_key": api_key, "sign": sign, "json": json_data},
        timeout=60,
        stream=True,
    )

    if resp.status_code == 200 and "audio" in resp.headers.get("Content-Type", ""):
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        return True
    else:
        print(f"  Download failed: {resp.status_code} {resp.headers.get('Content-Type', '')}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download Mango Office call recordings")
    parser.add_argument("--api-key", required=True, help="Mango API key (Уникальный код АТС)")
    parser.add_argument("--api-salt", required=True, help="Mango API salt (Ключ)")
    parser.add_argument("--days", type=int, default=21, help="Number of days to look back")
    parser.add_argument("--extension", default="12", help="Internal extension number (Andreev=12)")
    parser.add_argument("--min-duration", type=int, default=30, help="Minimum call duration in seconds")
    parser.add_argument("--max-downloads", type=int, default=10, help="Max recordings to download")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    to_date = datetime.now()
    from_date = to_date - timedelta(days=args.days)

    print(f"Fetching calls from {from_date.date()} to {to_date.date()}...")
    history = get_call_history(args.api_key, args.api_salt, from_date, to_date, args.extension)

    if not history:
        print("No call history returned")
        return

    calls = history if isinstance(history, list) else history.get("list", [])
    print(f"Found {len(calls)} calls")

    # Filter by extension and duration
    filtered = []
    for call in calls:
        dur = int(call.get("duration", 0))
        records = call.get("records", [])
        ext = call.get("ext", "") or call.get("from_extension", "")
        if dur >= args.min_duration and records:
            filtered.append(call)

    print(f"Calls with recordings (>{args.min_duration}s): {len(filtered)}")

    downloaded = 0
    for call in filtered[:args.max_downloads]:
        records = call.get("records", [])
        for rec_id in records:
            date_str = datetime.fromtimestamp(int(call.get("start", 0))).strftime("%Y%m%d_%H%M")
            dur = call.get("duration", "?")
            phone = call.get("to_number", call.get("from_number", "unknown"))
            fname = f"andreev_{date_str}_{phone}_{dur}s.mp3"
            output_path = os.path.join(OUTPUT_DIR, fname)

            if os.path.exists(output_path):
                print(f"  Skip (exists): {fname}")
                continue

            print(f"  Downloading: {fname}...")
            if download_recording(args.api_key, args.api_salt, rec_id, output_path):
                downloaded += 1
                size = os.path.getsize(output_path)
                print(f"  OK: {size / 1024:.0f} KB")

    print(f"\nDownloaded: {downloaded} recordings to {OUTPUT_DIR}/")
    print(f"\nNext steps:")
    print(f"  1. Listen and pick best 5-10s fragment of Andreev's voice")
    print(f"  2. Convert to WAV 16kHz: sox recording.mp3 -r 16000 -c 1 voice_sample.wav")
    print(f"  3. Use for CosyVoice cloning: voice_samples/reference_voice.wav")


if __name__ == "__main__":
    main()
