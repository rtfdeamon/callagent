#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$SCRIPT_DIR"
export PYTHONUNBUFFERED=1

# Python: используем venv, если он есть в каталоге скрипта (локальная разработка),
# иначе системный python из PATH (сценарий Docker, где зависимости поставлены
# глобально через pip install -r requirements.txt).
if [ -x "$SCRIPT_DIR/venv/bin/python" ]; then
  PY="$SCRIPT_DIR/venv/bin/python"
else
  PY="$(command -v python3 || command -v python)"
fi
if [ -z "$PY" ]; then
  echo "ОШИБКА: python не найден ни в $SCRIPT_DIR/venv, ни в PATH" >&2
  exit 1
fi

if [ "${USE_FINETUNED_XTTS:-0}" = "1" ] && [ -z "${XTTS_CHECKPOINT_PATH:-}" ] && [ -f "$ROOT_DIR/data/xtts_finetuned_andreev/training_summary.json" ]; then
  XTTS_CHECKPOINT_PATH="$("$PY" - <<PY
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
echo "Python: $PY"
if [ -n "${XTTS_CHECKPOINT_PATH:-}" ]; then
  echo "Using fine-tuned XTTS checkpoint: $XTTS_CHECKPOINT_PATH"
else
  echo "Using base XTTS model"
fi
exec "$PY" -m uvicorn main:app --host 0.0.0.0 --port 8000 --app-dir "$SCRIPT_DIR"
