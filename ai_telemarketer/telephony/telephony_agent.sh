#!/bin/bash
# Обертка для запуска ИИ-агента в среде Asterisk (EAGI)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/../venv/bin/python"

# Запускаем основной скрипт через виртуальное окружение
exec "$VENV_PYTHON" "$SCRIPT_DIR/telephony_agent.py"
