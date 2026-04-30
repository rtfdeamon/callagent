#!/bin/bash
set -e
echo "=== Starting AI Telemarketer ==="

docker rm -f ai_telemarketer 2>/dev/null || true

docker run -d --name ai_telemarketer \
  -u root \
  --network host \
  -v /home/dmitriy/work/callagent/ai_telemarketer:/app \
  -v /home/dmitriy/work/callagent/models:/home/dmitriy/work/callagent/models:ro \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/app \
  asterisk-ai-voice-agent-local-ai-server:latest \
  bash -c "
    /opt/venv/bin/pip install --no-cache-dir fastapi uvicorn python-multipart aiohttp scipy faster-whisper 2>&1 | tail -3 && \
    echo '=== pip done, starting uvicorn ===' && \
    cd /app && \
    /opt/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
  "

echo "Container started. Watching logs (Ctrl+C to stop watching)..."
docker logs -f ai_telemarketer
