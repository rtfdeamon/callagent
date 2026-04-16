#!/usr/bin/env python3
"""Pre-warm Ollama qwen2.5:7b with MMVS system prompt KV cache."""
import urllib.request
import json
import sys

SYSTEM_PROMPT = (
    "Ты — Анна, менеджер по продажам компании MMVS «Мультимедиа Видеосистемы» "
    "(Екатеринбург). Говори коротко — это голосовой звонок. Один вопрос за раз. "
    "Только по-русски. Цену не называй — предлагай КП."
)

payload = {
    "model": "qwen2.5:7b",
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Добрый день"},
    ],
    "stream": False,
    "keep_alive": -1,
    "options": {"num_predict": 5, "temperature": 0.3},
}

try:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.load(resp)
    reply = result.get("message", {}).get("content", "").strip()
    eval_dur = result.get("eval_duration", 0) / 1e9
    eval_tok = result.get("eval_count", 0)
    print(f"OK: qwen2.5:7b warm ({eval_tok} tok/{eval_dur:.2f}s): {reply[:40]}")
    sys.exit(0)
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
