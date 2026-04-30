#!/usr/bin/env python3
import os
import sys
import asyncio
import websockets
import json
import numpy as np
from pathlib import Path

# Добавляем корневой путь проекта для импорта
sys.path.append(str(Path(__file__).parent.parent))
from utils.fast_resampler import resample_audio, float_to_pcm16, pcm16_to_float

# Параметры Asterisk (EAGI)
AUDIO_FD = 3
CHUNK_SIZE = 320 # 20ms at 8kHz PCM16 (8000 * 0.02 * 2bytes)

async def agi_command(cmd: str):
    """Отправка AGI команды и чтение ответа."""
    sys.stdout.write(f"{cmd}\n")
    sys.stdout.flush()
    return sys.stdin.readline()

async def read_asterisk_audio(ws):
    """Поток чтения аудио из Asterisk (FD 3) и отправка в ИИ-пайплайн."""
    audio_fd = os.fdopen(AUDIO_FD, 'rb')
    try:
        while True:
            # Читаем 20мс сырого аудио (8 кГц, 16 бит моно)
            raw_audio = audio_fd.read(CHUNK_SIZE)
            if not raw_audio:
                break
                
            # Ресамплинг: 8кГц -> 16кГц (формат, который ждет наш backend)
            audio_np = pcm16_to_float(raw_audio)
            audio_16k_np = resample_audio(audio_np, 8000, 16000)
            audio_16k_pcm = float_to_pcm16(audio_16k_np)
            
            # Отправка в WebSocket
            await ws.send(audio_16k_pcm)
            await asyncio.sleep(0) # Уступаем время другим корутинам
    except Exception as e:
        sys.stderr.write(f"[EAGI] Audio read error: {e}\n")

async def handle_responses(ws):
    """Поток получения ответов от ИИ и воспроизведение их в звонок."""
    try:
        async for message in ws:
            if isinstance(message, str):
                # Текстовые данные (STT, LLM статус) — можно логировать в stderr для отладки Asterisk
                data = json.loads(message)
                sys.stderr.write(f"[EAGI] AI Status: {data.get('type')} - {data.get('text', '')}\n")
            else:
                # Бинарные данные (аудио ответ от XTTS, 24 кГц)
                audio_np = pcm16_to_float(message)
                
                # Ресамплинг: 24кГц -> 8кГц (для телефонии)
                audio_8k_np = resample_audio(audio_np, 24000, 8000)
                audio_8k_pcm = float_to_pcm16(audio_8k_np)
                
                # В EAGI мы можем просто писать PCM прямо в STDOUT, если канал открыт в режиме RAW,
                # но надежнее использовать временный файл и STREAM FILE.
                # Для скорости создаем временный WAV
                tmp_wav = "/tmp/asterisk_ai_response.wav"
                import wave
                with wave.open(tmp_wav, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(8000)
                    wf.writeframes(audio_8k_pcm)
                
                # Команда Asterisk для проигрывания (без расширения .wav)
                await agi_command(f"STREAM FILE /tmp/asterisk_ai_response \"\"")
                
    except Exception as e:
        sys.stderr.write(f"[EAGI] Response handle error: {e}\n")

async def main():
    # 1. Читаем переменные окружения AGI (Asterisk посылает их при запуске)
    params = {}
    while True:
        line = sys.stdin.readline().strip()
        if not line:
            break
        key, val = line.split(":", 1)
        params[key.strip()] = val.strip()

    # 2. Инициализируем звонок
    await agi_command("ANSWER")
    
    # 3. Подключаемся к нашему работающему ИИ-серверу
    uri = "ws://localhost:8000/ws"
    try:
        async with websockets.connect(uri) as ws:
            sys.stderr.write("[EAGI] Connected to AI Backend\n")
            
            # Запускаем два параллельных процесса: чтение и воспроизведение
            await asyncio.gather(
                read_asterisk_audio(ws),
                handle_responses(ws)
            )
    except Exception as e:
        sys.stderr.write(f"[EAGI] Connection failed: {e}\n")
        await agi_command("SAY PHONETIC Error")

if __name__ == "__main__":
    asyncio.run(main())
