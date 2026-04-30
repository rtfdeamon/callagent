import logging
import os
from pathlib import Path
import tempfile
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class STTService:
    def __init__(self, model_size: str = "medium", device: str = "auto"):
        """
        Инициализация Faster-Whisper STT.
        device="auto" → пытается cuda, если не доступен — cpu.
        """
        import torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # На GPU используем float16 для максимальной скорости
        compute_type = "float16" if device == "cuda" else "int8"
        model_ref = self._resolve_model_ref(model_size)
        logger.info(
            f"[STT] Loading Whisper model={model_ref}, device={device}, compute_type={compute_type}"
        )
        self.model = WhisperModel(model_ref, device=device, compute_type=compute_type)
        logger.info("[STT] Whisper model loaded OK")

    @staticmethod
    def _resolve_model_ref(model_size: str) -> str:
        """Prefer a local HF snapshot path to avoid network/cache resolution hangs."""
        if os.path.exists(model_size):
            return model_size

        snapshot_root = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / f"models--Systran--faster-whisper-{model_size}"
            / "snapshots"
        )
        if snapshot_root.is_dir():
            snapshots = sorted(path for path in snapshot_root.iterdir() if path.is_dir())
            if snapshots:
                resolved = str(snapshots[-1])
                logger.info(f"[STT] Using local snapshot for model '{model_size}': {resolved}")
                return resolved

        return model_size

    def transcribe(self, audio_data) -> str:
        """
        Транскрибирует аудио (WAV/MP3 bytes ИЛИ numpy float32 array) в текст.
        """
        import numpy as np

        if isinstance(audio_data, np.ndarray):
            return self.transcribe_np(audio_data)

        # Если пришли байты, сохраняем во временный файл (для файлов с заголовками)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        try:
            segments, info = self.model.transcribe(tmp_path, beam_size=5, language="ru")
            text = " ".join([segment.text for segment in segments])
            logger.info(f"[STT] Transcribed file ({info.language}, {info.duration:.1f}s): {text[:80]}...")
            return text.strip()
        finally:
            os.unlink(tmp_path)

    def transcribe_np(self, audio_np: "np.ndarray") -> str:
        """
        Прямая транскрипция из numpy массива (float32).
        """
        segments, info = self.model.transcribe(audio_np, beam_size=5, language="ru")
        text = " ".join([segment.text for segment in segments])
        duration = audio_np.size / 16000 # Предполагаем 16kHz для Web
        logger.info(f"[STT] Transcribed NP ({info.language}, {duration:.1f}s): {text[:80]}...")
        return text.strip()
