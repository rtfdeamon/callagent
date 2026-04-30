"""
XTTS v2 (Coqui) TTS-адаптер для основного пайплайна.

Назначение:
    Подключение fine-tuned XTTS-чекпойнта (например, голос Андреева,
    обученный из data/xtts_finetuned_andreev_v2/...) к основному
    пайплайну через интерфейс TTSComponent.

Условия применения:
    - Установлен пакет TTS (Coqui), torch, soundfile, numpy.
    - В config указан путь к чекпойнту (model_dir или model_path) и
      reference WAV-файл (speaker_wav) длительностью 6–30 с.
    - Доступен GPU CUDA (CPU-режим работает, но в 10–20 раз медленнее
      и не пригоден для real-time).

Описание логической структуры:
    1. На старте (start) грузим модель в память, прогреваем небольшой
       синтез — снимает первый latency-spike.
    2. На open_call ничего не делаем (модель shared между звонками).
    3. На synthesize:
       a) разбиваем текст на предложения (короткие → меньше задержка
          до первого аудио-фрейма);
       b) каждое предложение синтезируем через TTS.tts() в отдельном
          потоке (sync API Coqui);
       c) ресэмплируем в target_sample_rate (8000 Hz по умолчанию для
          телефонии);
       d) кодируем в нужный формат (mulaw / pcm16);
       e) yield чанками по chunk_size_ms (40 мс по умолчанию — кратно
          фрейму AudioSocket).

Используемые технические средства:
    Python 3.10+, Coqui TTS (≥0.22), PyTorch, NumPy, soundfile.
    Минимум 6 ГБ VRAM (для XTTS v2 базовый + fine-tuned LoRA-стиль
    весов хватает 4 ГБ; полный fine-tune ~6 ГБ).

Вызов и загрузка:
    Регистрация в orchestrator.py: добавить
        elif provider == "xtts":
            tts = XTTSAdapter(component_key="xtts_tts", config=tts_options)
    После проверки на реальной модели и звонке (см. комментарий
    «не регистрируется по умолчанию» ниже).

Входные данные:
    options (TTSOptions):
        model_dir: путь к каталогу с config.json и best_model.pth
        speaker_wav: путь к reference WAV-файлу
        device: "cuda" | "cpu" (по умолчанию "cuda")
        language: код языка по ISO-639-1 ("ru" по умолчанию)
        target_sample_rate: 8000 / 16000 / 24000 (8000 для SIP)
        encoding: "mulaw" | "pcm16" | "wav"
        chunk_size_ms: длина выходного чанка (40 по умолчанию)

Выходные данные:
    AsyncIterator[bytes] — фреймы аудио в указанной кодировке.

Версия: 0.1.0
Автор: callagent team
Дата: 2026-04-29
"""

from __future__ import annotations

import asyncio
import audioop
import io
import logging
import os
import re
from typing import Any, AsyncIterator, Dict, List, Optional

from .base import TTSComponent

logger = logging.getLogger(__name__)


# Импорты тяжёлых зависимостей делаем ленивыми — без них основной пайплайн
# должен импортироваться без ошибок (в проде XTTS может быть не нужен).
def _lazy_imports():
    """Импорт TTS, torch, numpy, soundfile только при первом обращении."""
    import numpy as np
    import soundfile as sf
    import torch
    from TTS.api import TTS
    return np, sf, torch, TTS


class XTTSAdapterError(RuntimeError):
    """Ошибка инициализации или работы XTTS-адаптера."""


class XTTSAdapter(TTSComponent):
    """TTS-компонент на базе Coqui XTTS v2 (с поддержкой fine-tuned весов).

    Не регистрируется в orchestrator по умолчанию: требует ручного
    подтверждения работоспособности на конкретном чекпойнте (модель тяжёлая,
    автоматический healthcheck недоступен).
    """

    component_key = "xtts_tts"

    # Граница для разбиения по предложениям. Включаем характерные русские
    # знаки, чтобы дробить «...», «—» и т.п.
    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|(?<=[.!?])(?=\s*[А-ЯA-Z])")

    def __init__(self, *, options: Optional[Dict[str, Any]] = None) -> None:
        opts = options or {}
        self._model_dir: str = opts.get("model_dir") or os.getenv(
            "XTTS_MODEL_DIR", ""
        )
        self._model_path: str = opts.get("model_path") or os.path.join(
            self._model_dir, "best_model.pth"
        ) if self._model_dir else ""
        self._config_path: str = opts.get("config_path") or os.path.join(
            self._model_dir, "config.json"
        ) if self._model_dir else ""
        self._speaker_wav: str = opts.get("speaker_wav") or os.getenv(
            "XTTS_SPEAKER_WAV", ""
        )
        self._device: str = opts.get("device", "cuda")
        self._language: str = opts.get("language", "ru")
        self._target_sample_rate: int = int(opts.get("target_sample_rate", 8000))
        self._encoding: str = str(opts.get("encoding", "mulaw")).lower()
        self._chunk_size_ms: int = int(opts.get("chunk_size_ms", 40))
        # Coqui XTTS v2 выдаёт 24 кГц
        self._native_sample_rate: int = 24000

        self._tts = None  # type: Any
        self._lock = asyncio.Lock()
        self._warmup_done = False

    async def start(self) -> None:
        """Загрузка модели XTTS и однократный прогрев."""
        if not self._model_dir or not os.path.isdir(self._model_dir):
            raise XTTSAdapterError(
                f"XTTS: каталог модели не задан или не существует: {self._model_dir!r}"
            )
        if not os.path.isfile(self._model_path):
            raise XTTSAdapterError(
                f"XTTS: файл весов не найден: {self._model_path!r}"
            )
        if not os.path.isfile(self._config_path):
            raise XTTSAdapterError(
                f"XTTS: config.json не найден: {self._config_path!r}"
            )
        if not os.path.isfile(self._speaker_wav):
            raise XTTSAdapterError(
                f"XTTS: reference WAV не найден: {self._speaker_wav!r}"
            )

        logger.info(
            "ИНФО: загрузка XTTS [model_dir=%s, device=%s]",
            self._model_dir,
            self._device,
        )

        def _load_sync():
            np, sf, torch, TTS = _lazy_imports()
            tts = TTS(
                model_path=self._model_path,
                config_path=self._config_path,
                progress_bar=False,
                gpu=(self._device == "cuda" and torch.cuda.is_available()),
            )
            return tts

        self._tts = await asyncio.to_thread(_load_sync)

        # Прогрев: первый синтез долгий (компиляция CUDA-ядер). Делаем тихо.
        try:
            await asyncio.to_thread(
                self._tts.tts,
                text="Привет.",
                speaker_wav=self._speaker_wav,
                language=self._language,
            )
            self._warmup_done = True
            logger.info("ИНФО: XTTS прогрет, готов к синтезу")
        except Exception as exc:
            logger.warning(
                "ПРЕДУПРЕЖДЕНИЕ: прогрев XTTS не удался [error=%s]", exc, exc_info=True
            )

    async def stop(self) -> None:
        """Отдаём память GPU. Coqui TTS не имеет явного close, поэтому удаляем."""
        if self._tts is not None:
            try:
                _, _, torch, _ = _lazy_imports()
                self._tts = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                self._tts = None

    async def synthesize(
        self,
        call_id: str,
        text: str,
        options: Dict[str, Any],
    ) -> AsyncIterator[bytes]:
        """Стриминг по предложениям: текст → wav → ресэмпл → encoding → чанки."""
        cleaned = (text or "").strip()
        if not cleaned:
            return
        if self._tts is None:
            logger.error(
                "ОШИБКА: XTTS не инициализирован [call_id=%s]", call_id
            )
            return

        target_sr = int(options.get("target_sample_rate", self._target_sample_rate))
        encoding = str(options.get("encoding", self._encoding)).lower()
        chunk_ms = int(options.get("chunk_size_ms", self._chunk_size_ms))
        speaker_wav = options.get("speaker_wav") or self._speaker_wav
        language = options.get("language") or self._language

        sentences = self._split_sentences(cleaned)

        # Lock защищает разделяемую модель от одновременных вызовов.
        # XTTS не thread-safe при параллельном tts() — потери качества + OOM.
        async with self._lock:
            for sentence in sentences:
                if not sentence:
                    continue
                try:
                    audio_bytes = await asyncio.to_thread(
                        self._synthesize_sentence_sync,
                        sentence,
                        speaker_wav,
                        language,
                        target_sr,
                        encoding,
                    )
                except Exception as exc:
                    logger.error(
                        "ОШИБКА: XTTS синтез сорвался [call_id=%s, sentence=%r, error=%s]",
                        call_id,
                        sentence[:60],
                        exc,
                        exc_info=True,
                    )
                    continue
                if not audio_bytes:
                    continue
                # Режем на чанки по chunk_ms
                for chunk in self._iter_chunks(audio_bytes, target_sr, encoding, chunk_ms):
                    yield chunk

    def _split_sentences(self, text: str) -> List[str]:
        """Разбиение текста на предложения по знакам пунктуации.

        Цель — снизить задержку до первого аудио-фрейма (TTFB): модель
        возвращает первое предложение быстрее, чем весь монолог.
        """
        parts = self._SENTENCE_SPLIT_RE.split(text)
        # Дополнительно объединяем слишком короткие куски (<3 слов) с соседним,
        # иначе XTTS теряет интонацию.
        result: List[str] = []
        buf = ""
        for part in parts:
            piece = part.strip()
            if not piece:
                continue
            if len(piece.split()) < 3 and buf:
                buf = f"{buf} {piece}".strip()
            else:
                if buf:
                    result.append(buf)
                buf = piece
        if buf:
            result.append(buf)
        return result

    def _synthesize_sentence_sync(
        self,
        sentence: str,
        speaker_wav: str,
        language: str,
        target_sr: int,
        encoding: str,
    ) -> bytes:
        """Синхронный синтез одного предложения. Возвращает байты в `encoding`.

        target_sr — частота для декодера AudioSocket (8000 для SIP).
        """
        np, sf, torch, _ = _lazy_imports()
        wav = self._tts.tts(
            text=sentence,
            speaker_wav=speaker_wav,
            language=language,
        )
        # Coqui отдаёт list/np.ndarray float32 в диапазоне ±1.0
        arr = np.asarray(wav, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=-1)  # моно
        # ресэмпл нужно делать качественно: используем soxr через soundfile,
        # либо простой decimation. Здесь — через soundfile in-memory:
        # пишем 24k WAV → читаем с целевой частотой через scipy.signal.resample_poly.
        if self._native_sample_rate != target_sr:
            arr = self._resample_linear(arr, self._native_sample_rate, target_sr)

        # PCM16
        pcm16 = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

        if encoding == "pcm16" or encoding == "slin16" or encoding == "linear16":
            return pcm16
        if encoding == "mulaw" or encoding == "ulaw" or encoding == "g711_ulaw":
            return audioop.lin2ulaw(pcm16, 2)
        if encoding == "wav":
            buf = io.BytesIO()
            sf.write(buf, arr, target_sr, format="WAV", subtype="PCM_16")
            return buf.getvalue()
        # Fallback: PCM16
        logger.warning(
            "ПРЕДУПРЕЖДЕНИЕ: неизвестный encoding %r → выдаю PCM16", encoding
        )
        return pcm16

    @staticmethod
    def _resample_linear(arr, src_sr: int, dst_sr: int):
        """Простой ресэмпл через scipy.signal.resample_poly (anti-alias).

        Не используем torchaudio.functional.resample, чтобы не тащить
        torchaudio как зависимость (его нет в requirements.txt).
        """
        from math import gcd
        from scipy.signal import resample_poly  # type: ignore

        g = gcd(src_sr, dst_sr)
        up = dst_sr // g
        down = src_sr // g
        return resample_poly(arr, up, down).astype(arr.dtype, copy=False)

    @staticmethod
    def _iter_chunks(payload: bytes, sr: int, encoding: str, chunk_ms: int):
        """Нарезка байт-потока на чанки длиной chunk_ms.

        Размер байт на сэмпл: pcm16 → 2, mulaw → 1.
        """
        if encoding in ("mulaw", "ulaw", "g711_ulaw"):
            bytes_per_sample = 1
        elif encoding == "wav":
            # WAV отдаём целиком — нарезка ломает заголовок
            yield payload
            return
        else:
            bytes_per_sample = 2
        chunk_bytes = max(1, int(sr * chunk_ms / 1000) * bytes_per_sample)
        for i in range(0, len(payload), chunk_bytes):
            yield payload[i : i + chunk_bytes]
