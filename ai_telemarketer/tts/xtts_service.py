import io
import logging
import os
import re
import numpy as np
import torch
import scipy.io.wavfile

logger = logging.getLogger(__name__)


class XTTS_Service:
    """
    Coqui XTTS v2 TTS с speaker cloning (голос Андреева).
    Использует низкоуровневый API для загрузки из локальной директории.
    float16 на GPU для экономии VRAM (RTX 5060 = 8GB).
    """

    def __init__(
        self,
        model_dir: str = "/home/dmitriy/work/callagent/models/xtts_v2",
        speaker_wav: str = "/home/dmitriy/work/callagent/models/andreev_voice.wav",
        checkpoint_path: str | None = None,
        config_path: str | None = None,
        vocab_path: str | None = None,
        temperature: float | None = None,
        repetition_penalty: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        do_sample: bool | None = None,
    ):
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        checkpoint_path = checkpoint_path or os.getenv("XTTS_CHECKPOINT_PATH")
        config_path = config_path or os.getenv("XTTS_CONFIG_PATH") or os.path.join(model_dir, "config.json")
        vocab_path = vocab_path or os.getenv("XTTS_VOCAB_PATH") or os.path.join(model_dir, "vocab.json")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.speaker_wav = speaker_wav
        self.temperature = temperature if temperature is not None else float(os.getenv("XTTS_TEMPERATURE", "0.75"))
        self.repetition_penalty = (
            repetition_penalty
            if repetition_penalty is not None
            else float(os.getenv("XTTS_REPETITION_PENALTY", "5.0"))
        )
        self.top_k = top_k if top_k is not None else int(os.getenv("XTTS_TOP_K", "50"))
        self.top_p = top_p if top_p is not None else float(os.getenv("XTTS_TOP_P", "0.85"))
        self.do_sample = do_sample if do_sample is not None else os.getenv("XTTS_DO_SAMPLE", "0") == "1"
        self.length_penalty = float(os.getenv("XTTS_LENGTH_PENALTY", "1.1"))
        self.num_beams = int(os.getenv("XTTS_NUM_BEAMS", "1"))
        self.speed = float(os.getenv("XTTS_SPEED", "1.15"))
        self.trim_silence_threshold = float(os.getenv("XTTS_TRIM_SILENCE_THRESHOLD", "0.01"))
        self.trim_silence_margin = int(os.getenv("XTTS_TRIM_SILENCE_MARGIN", "3000"))

        logger.info(f"[TTS] Loading XTTS v2 from {model_dir}, device={self.device}")
        if checkpoint_path:
            logger.info(f"[TTS] Using fine-tuned checkpoint: {checkpoint_path}")

        config = XttsConfig()
        config.load_json(config_path)

        self.model = Xtts.init_from_config(config)
        if checkpoint_path:
            self.model.load_checkpoint(
                config,
                checkpoint_path=checkpoint_path,
                vocab_path=vocab_path,
                eval=True,
                use_deepspeed=False,
            )
        else:
            self.model.load_checkpoint(config, checkpoint_dir=model_dir, eval=True)

        # float32 на GPU (~2GB VRAM, влезает в 8GB при выгруженной Ollama)
        self.model = self.model.to(self.device)
        logger.info(f"[TTS] Model loaded to {self.device} in float32")

        logger.info("[TTS] XTTS v2 model loaded OK")

        # Кэшируем speaker embedding один раз при старте
        logger.info(f"[TTS] Computing speaker embedding from {speaker_wav}")
        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(
            audio_path=[self.speaker_wav]
        )
        logger.info("[TTS] Speaker embedding cached OK")

    @staticmethod
    def _split_text(text: str) -> list[tuple[str, int]]:
        """
        Разбивает текст на сегменты с указанием длительности паузы после каждого (в мс).
        .?! -> 700ms, ,;: -> 300ms.
        """
        # Сначала бьем на предложения
        sentence_parts = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks: list[tuple[str, int]] = []
        
        for part in sentence_parts:
            part = part.strip()
            if not part:
                continue
            
            # Определяем тип знака в конце предложения для паузы
            last_char = part[-1] if part else ""
            sentence_pause = 700 if last_char in ".!?" else 300
            
            if len(part) <= 80:
                chunks.append((part, sentence_pause))
                continue
            
            # Если предложение длинное, бьем по запятым/двоеточиям внутри
            clause_parts = re.split(r"(?<=[,;:])\s+", part)
            current = ""
            for i, clause in enumerate(clause_parts):
                clause = clause.strip()
                if not clause:
                    continue
                
                proposal = f"{current} {clause}".strip()
                is_last_clause = (i == len(clause_parts) - 1)
                
                if current and len(proposal) > 80:
                    # Сохраняем текущий накопленный кусок с паузой запятой
                    chunks.append((current, 300))
                    current = clause
                else:
                    current = proposal
                
                if is_last_clause and current:
                    chunks.append((current, sentence_pause))
            
        return chunks or [(text.strip(), 700)]

    def _trim_chunk(self, wav_np: np.ndarray) -> np.ndarray:
        if wav_np.size == 0:
            return wav_np

        mask = np.flatnonzero(np.abs(wav_np) >= self.trim_silence_threshold)
        if mask.size == 0:
            return wav_np

        start = max(0, int(mask[0]) - self.trim_silence_margin)
        end = min(wav_np.shape[0], int(mask[-1]) + self.trim_silence_margin + 1)
        return wav_np[start:end]

    def generate_audio(self, text: str) -> bytes:
        """
        Генерация WAV аудио из текста с клонированным голосом Андреева.
        Возвращает bytes WAV-файла (24kHz, int16).
        """
        logger.info(f"[TTS] Generating: {text[:80]}...")
        pieces = []
        silence = np.zeros(2400, dtype=np.float32)

        with torch.no_grad():
            for chunk, pause_ms in self._split_text(text):
                # Добавляем небольшой паддинг-пробел в конце, чтобы модель не обрывала окончания
                text_chunk = f"{chunk} "
                inference_kwargs = {
                    "text": text_chunk,
                    "language": "ru",
                    "gpt_cond_latent": self.gpt_cond_latent,
                    "speaker_embedding": self.speaker_embedding,
                    "repetition_penalty": self.repetition_penalty,
                    "length_penalty": self.length_penalty,
                    "do_sample": self.do_sample,
                    "num_beams": self.num_beams,
                    "speed": self.speed,
                    "enable_text_splitting": False,
                }
                if self.do_sample:
                    inference_kwargs.update(
                        {
                            "temperature": self.temperature,
                            "top_k": self.top_k,
                            "top_p": self.top_p,
                        }
                    )

                out = self.model.inference(**inference_kwargs)
                wav_np = out["wav"]
                if isinstance(wav_np, torch.Tensor):
                    wav_np = wav_np.cpu().float().numpy()
                
                wav_np = self._trim_chunk(np.asarray(wav_np, dtype=np.float32))
                if wav_np.size == 0:
                    continue
                
                pieces.append(wav_np)
                
                # Добавляем семантическую паузу
                pause_samples = int(24000 * (pause_ms / 1000))
                pieces.append(np.zeros(pause_samples, dtype=np.float32))

        wav_np = np.concatenate(pieces[:-1]) if pieces else silence.copy()

        # Нормализуем в int16
        peak = np.max(np.abs(wav_np))
        if peak > 0:
            wav_int16 = np.int16(wav_np / peak * 32767)
        else:
            wav_int16 = np.int16(wav_np)

        buffer = io.BytesIO()
        scipy.io.wavfile.write(buffer, 24000, wav_int16)
        audio_bytes = buffer.getvalue()
        logger.info(f"[TTS] Generated {len(audio_bytes)} bytes")
        return audio_bytes
