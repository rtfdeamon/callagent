#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$SCRIPT_DIR"
export PYTHONUNBUFFERED=1

if [ "${USE_FINETUNED_XTTS:-0}" = "1" ] && [ -z "${XTTS_CHECKPOINT_PATH:-}" ] && [ -f "$ROOT_DIR/data/xtts_finetuned_andreev/training_summary.json" ]; then
  XTTS_CHECKPOINT_PATH="$("$SCRIPT_DIR/venv/bin/python" - <<PY
import json
from pathlib import Path
summary = Path("$ROOT_DIR/data/xtts_finetuned_andreev/training_summary.json")
data = json.loads(summary.read_text(encoding="utf-8"))
print(data.get("checkpoint_path", ""))
PY
)"
  export XTTS_CHECKPOINT_PATH
fi

echo "=== AI Telemarketer — GPU Mode ==="
if [ -n "${XTTS_CHECKPOINT_PATH:-}" ]; then
  echo "Using fine-tuned XTTS checkpoint: $XTTS_CHECKPOINT_PATH"
else
  echo "Using base XTTS model"
fi
exec "$SCRIPT_DIR/venv/bin/uvicorn" main:app --host 0.0.0.0 --port 8000 --app-dir "$SCRIPT_DIR"
