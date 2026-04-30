#!/usr/bin/env python3
import io
import os
import sys
import wave
import asyncio
import itertools
import websockets
import json
import numpy as np
from pathlib import Path

# Счётчик реплик для уникальных имён temp-файлов в пределах одного процесса
# (PID + index). Process-local — для разных EAGI-процессов PID уже разный.
_playback_counter = itertools.count()

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
                # Бинарные данные — это WAV-файл от XTTS (с RIFF-заголовком).
                # Декодируем WAV перед ресэмплированием, иначе RIFF-заголовок
                # будет интерпретирован как аудио-семплы → шум в начале и
                # неверная длительность.
                try:
                    with wave.open(io.BytesIO(message), 'rb') as wf_in:
                        src_rate = wf_in.getframerate()
                        n_channels = wf_in.getnchannels()
                        sampwidth = wf_in.getsampwidth()
                        pcm_in = wf_in.readframes(wf_in.getnframes())
                except wave.Error as exc:
                    sys.stderr.write(
                        f"[EAGI] WAV decode error: {exc} "
                        f"(размер сообщения {len(message)} байт)\n"
                    )
                    continue

                if sampwidth != 2 or n_channels != 1:
                    sys.stderr.write(
                        f"[EAGI] Неподдерживаемый WAV: "
                        f"sampwidth={sampwidth}, channels={n_channels}\n"
                    )
                    continue

                audio_np = pcm16_to_float(pcm_in)
                # Ресамплинг: src_rate (обычно 24кГц XTTS) → 8кГц для телефонии
                audio_8k_np = resample_audio(audio_np, src_rate, 8000)
                audio_8k_pcm = float_to_pcm16(audio_8k_np)

                # Уникальное имя файла на звонок: PID процесса EAGI + порядковый
                # номер реплики. Asterisk запускает отдельный процесс на каждый
                # звонок, так что PID гарантированно разный между параллельными
                # звонками; единый /tmp/asterisk_ai_response.wav между ними
                # перетирался бы и приводил к утечке аудио в чужой канал.
                playback_index = next(_playback_counter)
                stem = f"/tmp/asterisk_ai_response_{os.getpid()}_{playback_index}"
                tmp_wav = f"{stem}.wav"
                with wave.open(tmp_wav, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(8000)
                    wf.writeframes(audio_8k_pcm)

                try:
                    # Команда Asterisk для проигрывания (без расширения .wav)
                    await agi_command(f"STREAM FILE {stem} \"\"")
                finally:
                    # Удаляем файл после проигрывания, чтобы не накапливать
                    # мусор в /tmp при долгих звонках.
                    try:
                        os.unlink(tmp_wav)
                    except OSError:
                        pass
                
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

            # Два параллельных процесса: чтение из Asterisk и воспроизведение
            # ответов. При hangup Asterisk закрывает FD 3 → read_asterisk_audio
            # завершается, но handle_responses продолжает ждать на ws.recv()
            # бесконечно. Запускаем через wait(FIRST_COMPLETED) и закрываем
            # WS + cancel'им оставшуюся таску, иначе EAGI-процесс зависает
            # после разъединения.
            reader_task = asyncio.create_task(read_asterisk_audio(ws))
            responder_task = asyncio.create_task(handle_responses(ws))
            done, pending = await asyncio.wait(
                {reader_task, responder_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            sys.stderr.write(
                f"[EAGI] One side completed (done={[t.get_name() for t in done]}), "
                f"закрываем WS и отменяем оставшиеся таски\n"
            )
            try:
                await ws.close()
            except Exception:
                pass
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
    except Exception as e:
        sys.stderr.write(f"[EAGI] Connection failed: {e}\n")
        await agi_command("SAY PHONETIC Error")

if __name__ == "__main__":
    asyncio.run(main())
