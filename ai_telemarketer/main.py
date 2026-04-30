import logging
import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("ai_telemarketer")

# Глобальные сервисы, загружаются при старте
stt_service = None
dialog_manager = None
tts_service = None

# Пути к моделям XTTS — выносим в ENV, чтобы не править код при перезапуске
# из другой машины / с другим обученным голосом.
TELEMARKETER_XTTS_MODEL_DIR = os.getenv(
    "TELEMARKETER_XTTS_MODEL_DIR",
    "/home/dmitriy/work/callagent/models/xtts_v2",
)
TELEMARKETER_XTTS_SPEAKER_WAV = os.getenv(
    "TELEMARKETER_XTTS_SPEAKER_WAV",
    "/home/dmitriy/work/callagent/models/andreev_voice.wav",
)
TELEMARKETER_STT_MODEL_SIZE = os.getenv("TELEMARKETER_STT_MODEL_SIZE", "small")
TELEMARKETER_STT_DEVICE = os.getenv("TELEMARKETER_STT_DEVICE", "cuda")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка моделей при старте, очистка при остановке.

    Порядок: сначала проверяем доступность Ollama (быстро падает, если нет),
    затем грузим STT и TTS (тяжёлые, ~10-30 с). Это экономит минуты, если
    инфраструктура ещё не готова.
    """
    global stt_service, dialog_manager, tts_service

    # === LLM healthcheck — самая быстрая проверка инфраструктуры ===
    logger.info("ИНФО: проверка доступности Ollama...")
    from llm.dialog_manager import DialogManager, OllamaUnavailableError
    dialog_manager = DialogManager()
    try:
        await dialog_manager.healthcheck()
    except OllamaUnavailableError as exc:
        logger.error(
            "КРИТ: Ollama недоступна — старт прерван [error=%s]", exc, exc_info=True
        )
        raise

    # === Проверка путей TTS перед загрузкой (загрузка XTTS долгая) ===
    if not os.path.isdir(TELEMARKETER_XTTS_MODEL_DIR):
        msg = f"Не найден каталог модели XTTS: {TELEMARKETER_XTTS_MODEL_DIR}"
        logger.error("КРИТ: %s", msg)
        raise FileNotFoundError(msg)
    if not os.path.isfile(TELEMARKETER_XTTS_SPEAKER_WAV):
        msg = f"Не найден reference-файл голоса: {TELEMARKETER_XTTS_SPEAKER_WAV}"
        logger.error("КРИТ: %s", msg)
        raise FileNotFoundError(msg)

    # === STT (Faster-Whisper) ===
    logger.info(
        "ИНФО: загрузка STT [model=%s, device=%s]...",
        TELEMARKETER_STT_MODEL_SIZE,
        TELEMARKETER_STT_DEVICE,
    )
    from stt.whisper_service import STTService
    stt_service = await asyncio.to_thread(
        STTService,
        model_size=TELEMARKETER_STT_MODEL_SIZE,
        device=TELEMARKETER_STT_DEVICE,
    )

    # === TTS (XTTS v2) ===
    logger.info("ИНФО: загрузка TTS (XTTS v2 + голос Андреева)...")
    from tts.xtts_service import XTTS_Service
    tts_service = await asyncio.to_thread(
        XTTS_Service,
        model_dir=TELEMARKETER_XTTS_MODEL_DIR,
        speaker_wav=TELEMARKETER_XTTS_SPEAKER_WAV,
    )

    logger.info("ИНФО: все сервисы загружены — AI Telemarketer готов")
    yield
    logger.info("ИНФО: остановка AI Telemarketer")


app = FastAPI(title="AI Telemarketer API", lifespan=lifespan)
# Каталог статики берём от расположения файла, а не хардкодим:
# в Docker (/app) и других checkout-ах путь будет другой, иначе StaticFiles
# падает на старте до того, как FastAPI успеет принять запросы.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=_APP_DIR), name="static")


# ─── Pydantic Models ───

from llm.dialog_manager import DialogState


class GenerateRequest(BaseModel):
    user_text: str
    current_state: DialogState


class TTSRequest(BaseModel):
    text: str


# ─── Endpoints ───


@app.post("/stt")
async def process_stt(file: UploadFile = File(...)):
    """Распознавание речи из аудиофайла."""
    audio_bytes = await file.read()
    text = await asyncio.to_thread(stt_service.transcribe, audio_bytes)
    return JSONResponse({"text": text})


@app.post("/generate_response")
async def generate_response(req: GenerateRequest):
    """Генерация ответа LLM для текущего стейта диалога."""
    result = await dialog_manager.generate_response(req.user_text, req.current_state)
    return result.dict()


@app.post("/tts")
async def process_tts(req: TTSRequest):
    """Синтез речи из текста (голос Андреева)."""
    audio_bytes = await asyncio.to_thread(tts_service.generate_audio, req.text)
    return Response(content=audio_bytes, media_type="audio/wav")


@app.post("/dialog_step")
async def dialog_step(
    audio: UploadFile = File(...),
    current_state: str = Form(...),
):
    """
    Полный цикл диалога: STT → LLM → TTS.
    Принимает аудиофайл + текущий стейт, возвращает WAV ответа.
    Метаданные (текст клиента, текст агента, следующий стейт) — в HTTP-заголовках.
    """
    audio_bytes = await audio.read()

    # 1. STT
    user_text = await asyncio.to_thread(stt_service.transcribe, audio_bytes)
    logger.info(f"[dialog_step] User said: {user_text}")

    # 2. LLM
    state_enum = DialogState(current_state)
    llm_resp = await dialog_manager.generate_response(user_text, state_enum)
    logger.info(f"[dialog_step] Agent response: {llm_resp.response}, next_state={llm_resp.next_state}")

    # 3. TTS
    response_audio = await asyncio.to_thread(tts_service.generate_audio, llm_resp.response)

    # Кодируем текстовые заголовки в latin-1 для HTTP
    headers = {
        "X-User-Text": user_text.encode("utf-8").decode("latin-1"),
        "X-Agent-Text": llm_resp.response.encode("utf-8").decode("latin-1"),
        "X-Next-State": llm_resp.next_state.value,
    }
    return Response(content=response_audio, media_type="audio/wav", headers=headers)


@app.get("/health")
async def health():
    """Проверка готовности всех подсистем.

    Возвращает 503 если хоть один компонент не загружен — это позволяет
    Asterisk-стороне фильтровать запросы и не отправлять звонки в неготовый
    сервис.
    """
    components = {
        "stt": stt_service is not None,
        "llm": dialog_manager is not None,
        "tts": tts_service is not None,
    }
    all_ready = all(components.values())
    body = {"status": "ok" if all_ready else "degraded", **components}
    return JSONResponse(body, status_code=200 if all_ready else 503)


# ─── WebSocket Voice Handler ───

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("[WS] New connection established")
    
    current_state = DialogState.INTRO
    audio_buffer = b""
    silence_threshold = 1000  # Увеличили порог для фильтрации фонового шума
    silence_duration = 0
    is_processing = False
    
    # Счетчик для периодической отправки дебага
    frame_count = 0

    try:
        while True:
            data = await websocket.receive_bytes()
            
            if is_processing:
                continue

            # Конвертируем в numpy для анализа энергии
            audio_chunk = np.frombuffer(data, dtype=np.int16)
            energy = np.max(np.abs(audio_chunk)) if audio_chunk.size > 0 else 0
            
            if energy < silence_threshold:
                silence_duration += len(data) / 32000  # 16k * 2 bytes = 32000 bytes/sec
            else:
                # Обнаружен голос – сбрасываем тишину и копим в буфер
                silence_duration = 0
                audio_buffer += data
                
            # Ограничиваем рост буфера, если это просто шум (не более 15 сек)
            if len(audio_buffer) > 480000:
                audio_buffer = audio_buffer[-320000:]

            # Периодически отправляем дебаг-инфо клиенту (каждый 10-й фрейм)
            frame_count += 1
            if frame_count % 10 == 0:
                await websocket.send_json({
                    "type": "debug",
                    "energy": int(energy),
                    "silence": round(silence_duration, 1),
                    "buffer_size": len(audio_buffer)
                })
            
            # Если тишина > 1.2 сек и в буфере есть голос (> 0.5с) — отвечаем
            if silence_duration > 1.2 and len(audio_buffer) > 16000:
                is_processing = True
                logger.info(f"[WS] Speech detected. Buffer: {len(audio_buffer)} bytes. Processing...")
                
                try:
                    # 1. STT
                    # Конвертируем сырые байты в нормализованный массив float32 (16kHz)
                    audio_np = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                    text = await asyncio.to_thread(stt_service.transcribe, audio_np)
                    audio_buffer = b"" 
                    
                    if text.strip():
                        await websocket.send_json({"type": "stt_result", "text": text})
                        
                        # 2. LLM
                        resp = await dialog_manager.generate_response(text, current_state)
                        current_state = resp.next_state
                        await websocket.send_json({"type": "llm_response", "text": resp.response, "next_state": current_state})
                        
                        # 3. Стриминг TTS по предложениям
                        import re
                        sentences = re.split(r'(?<=[.!?])\s+', resp.response)
                        
                        for sentence in sentences:
                            if not sentence.strip():
                                continue
                            
                            logger.info(f"[WS] Synthesizing sentence: {sentence}")
                            audio_resp = await asyncio.to_thread(tts_service.generate_audio, sentence)
                            await websocket.send_bytes(audio_resp)
                    else:
                        logger.info("[WS] STT returned empty text, skip")
                        
                except Exception as e:
                    logger.error(
                        "ОШИБКА: WS-обработчик сбой при обработке речи [error=%s]",
                        e,
                        exc_info=True,
                    )
                    try:
                        await websocket.send_json(
                            {"type": "error", "message": f"Processing error: {e}"}
                        )
                    except Exception:
                        # Сокет уже разорван — нечего отправлять
                        pass

                finally:
                    is_processing = False
                    silence_duration = 0

    except WebSocketDisconnect:
        logger.info("ИНФО: WS-клиент отключился")
    except Exception as e:
        logger.error(
            "ОШИБКА: непредвиденная ошибка в WS-цикле [error=%s]", e, exc_info=True
        )
