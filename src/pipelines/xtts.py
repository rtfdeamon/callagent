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
    """Импорт torch/TTS низкоуровневых модулей только при первом обращении.

    Используем низкоуровневое API Coqui (XttsConfig + Xtts.load_checkpoint)
    вместо высокоуровневого TTS.api.TTS, потому что fine-tuned чекпойнты
    требуют явного `checkpoint_path=` в load_checkpoint, а высокоуровневый
    API ожидает каталог с дефолтной структурой.
    """
    import numpy as np
    import soundfile as sf
    import torch
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    return np, sf, torch, XttsConfig, Xtts


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
        self._model_path: str = opts.get("model_path") or (
            os.path.join(self._model_dir, "best_model.pth") if self._model_dir else ""
        )
        self._config_path: str = opts.get("config_path") or (
            os.path.join(self._model_dir, "config.json") if self._model_dir else ""
        )
        # vocab.json у fine-tuned чекпойнтов часто отсутствует — fallback на
        # vocab из базовой XTTS v2 (XTTS_VOCAB_PATH или модели в models/xtts_v2/).
        self._vocab_path: str = opts.get("vocab_path") or os.getenv(
            "XTTS_VOCAB_PATH",
            "/home/dmitriy/work/callagent/models/xtts_v2/vocab.json",
        )
        self._speaker_wav: str = opts.get("speaker_wav") or os.getenv(
            "XTTS_SPEAKER_WAV", ""
        )
        self._device: str = opts.get("device", "cuda")
        self._language: str = opts.get("language", "ru")
        self._target_sample_rate: int = int(opts.get("target_sample_rate", 8000))
        self._encoding: str = str(opts.get("encoding", "mulaw")).lower()
        self._chunk_size_ms: int = int(opts.get("chunk_size_ms", 40))
        # Параметры инференса (совместимы с ai_telemarketer/tts/xtts_service.py).
        self._temperature: float = float(opts.get("temperature", 0.75))
        self._repetition_penalty: float = float(opts.get("repetition_penalty", 5.0))
        self._top_k: int = int(opts.get("top_k", 50))
        self._top_p: float = float(opts.get("top_p", 0.85))
        self._do_sample: bool = bool(opts.get("do_sample", False))
        self._length_penalty: float = float(opts.get("length_penalty", 1.1))
        self._num_beams: int = int(opts.get("num_beams", 1))
        self._speed: float = float(opts.get("speed", 1.0))
        # Coqui XTTS v2 выдаёт 24 кГц
        self._native_sample_rate: int = 24000

        self._model = None  # Xtts instance
        self._gpt_cond_latent = None  # cached speaker latent (torch.Tensor)
        self._speaker_embedding = None  # cached speaker embedding (torch.Tensor)
        self._lock = asyncio.Lock()
        self._init_lock = asyncio.Lock()
        self._warmup_done = False

    async def open_call(self, call_id: str, options: Dict[str, Any]) -> None:
        """Прогрев модели при открытии звонка.

        Orchestrator не вызывает start() сам — он строит адаптер фабрикой и
        обращается к open_call/synthesize/close_call. Поэтому при первом
        open_call мы лениво подгружаем модель, чтобы первый synthesize не
        тратил 13 секунд на загрузку весов перед ответом клиенту.
        """
        await self._ensure_started()

    async def _ensure_started(self) -> None:
        """Идемпотентный init: грузит модель один раз, защищён от race-условий.

        Вызывается из open_call() и synthesize() — обеспечивает корректную
        работу и при «холодной» интеграции (когда start() не вызван явно).
        """
        if self._model is not None and self._gpt_cond_latent is not None:
            return
        async with self._init_lock:
            if self._model is not None and self._gpt_cond_latent is not None:
                return
            await self.start()

    async def start(self) -> None:
        """Загрузка fine-tuned XTTS и кэширование speaker embedding."""
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
        if not os.path.isfile(self._vocab_path):
            raise XTTSAdapterError(
                f"XTTS: vocab.json не найден: {self._vocab_path!r} "
                f"(укажите vocab_path в config или XTTS_VOCAB_PATH)"
            )
        if not os.path.isfile(self._speaker_wav):
            raise XTTSAdapterError(
                f"XTTS: reference WAV не найден: {self._speaker_wav!r}"
            )

        logger.info(
            "ИНФО: загрузка XTTS [model_dir=%s, checkpoint=%s, device=%s]",
            self._model_dir,
            os.path.basename(self._model_path),
            self._device,
        )

        def _load_sync():
            np, sf, torch, XttsConfig, Xtts = _lazy_imports()
            config = XttsConfig()
            config.load_json(self._config_path)
            model = Xtts.init_from_config(config)
            model.load_checkpoint(
                config,
                checkpoint_path=self._model_path,
                vocab_path=self._vocab_path,
                eval=True,
                use_deepspeed=False,
            )
            use_cuda = self._device == "cuda" and torch.cuda.is_available()
            model = model.to("cuda" if use_cuda else "cpu")
            return model

        self._model = await asyncio.to_thread(_load_sync)

        # Кэшируем speaker latent + embedding (вычисляются один раз).
        def _compute_speaker_sync():
            return self._model.get_conditioning_latents(
                audio_path=[self._speaker_wav]
            )

        gpt_cond_latent, speaker_embedding = await asyncio.to_thread(_compute_speaker_sync)
        self._gpt_cond_latent = gpt_cond_latent
        self._speaker_embedding = speaker_embedding
        logger.info("ИНФО: XTTS speaker embedding закэширован")

        # Прогрев: первый inference долгий (компиляция CUDA-ядер).
        try:
            await asyncio.to_thread(self._inference_sync, "Привет.", self._language)
            self._warmup_done = True
            logger.info("ИНФО: XTTS прогрет, готов к синтезу")
        except Exception as exc:
            logger.warning(
                "ПРЕДУПРЕЖДЕНИЕ: прогрев XTTS не удался [error=%s]", exc, exc_info=True
            )

    async def stop(self) -> None:
        """Отдаём память GPU."""
        if self._model is not None:
            try:
                _, _, torch, _, _ = _lazy_imports()
                self._model = None
                self._gpt_cond_latent = None
                self._speaker_embedding = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                self._model = None
                self._gpt_cond_latent = None
                self._speaker_embedding = None

    def _inference_sync(self, text: str, language: str):
        """Синхронный inference через низкоуровневое API. Возвращает np.ndarray
        float32 с частотой 24 кГц (нативная для XTTS)."""
        np, _, torch, _, _ = _lazy_imports()
        kwargs = {
            "text": f"{text} ",  # padding-пробел: модель не обрывает окончания
            "language": language,
            "gpt_cond_latent": self._gpt_cond_latent,
            "speaker_embedding": self._speaker_embedding,
            "repetition_penalty": self._repetition_penalty,
            "length_penalty": self._length_penalty,
            "do_sample": self._do_sample,
            "num_beams": self._num_beams,
            "speed": self._speed,
            "enable_text_splitting": False,
        }
        if self._do_sample:
            kwargs.update(
                {"temperature": self._temperature, "top_k": self._top_k, "top_p": self._top_p}
            )
        with torch.no_grad():
            out = self._model.inference(**kwargs)
        wav = out["wav"]
        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().float().numpy()
        return np.asarray(wav, dtype=np.float32)

    async def synthesize(
        self,
        call_id: str,
        text: str,
        options: Dict[str, Any],
    ) -> AsyncIterator[bytes]:
        """Стриминг по предложениям: текст → wav → ресэмпл → encoding → чанки.

        Lazy-init: если модель ещё не загружена (orchestrator вызвал synthesize
        без явного start()), грузим здесь — корректно для всех путей вызова.
        """
        cleaned = (text or "").strip()
        if not cleaned:
            return
        try:
            await self._ensure_started()
        except XTTSAdapterError as exc:
            logger.error(
                "ОШИБКА: XTTS не удалось инициализировать [call_id=%s, error=%s]",
                call_id,
                exc,
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

        speaker_wav — необязательный override; если совпадает с self._speaker_wav,
        используется закэшированный speaker_embedding. Иначе пересчитывается
        (медленно, ~1 с) — нужно только при смене голоса в рантайме.
        """
        np, sf, _, _, _ = _lazy_imports()

        if speaker_wav and speaker_wav != self._speaker_wav:
            # Голос сменился — пересчитываем conditioning. Редкий случай.
            self._gpt_cond_latent, self._speaker_embedding = (
                self._model.get_conditioning_latents(audio_path=[speaker_wav])
            )
            self._speaker_wav = speaker_wav

        arr = self._inference_sync(sentence, language)
        if arr.ndim > 1:
            arr = arr.mean(axis=-1)  # моно
        if self._native_sample_rate != target_sr:
            arr = self._resample_linear(arr, self._native_sample_rate, target_sr)

        # PCM16
        pcm16 = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

        if encoding in ("pcm16", "slin16", "linear16"):
            return pcm16
        if encoding in ("mulaw", "ulaw", "g711_ulaw"):
            return audioop.lin2ulaw(pcm16, 2)
        if encoding == "wav":
            buf = io.BytesIO()
            sf.write(buf, arr, target_sr, format="WAV", subtype="PCM_16")
            return buf.getvalue()
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
