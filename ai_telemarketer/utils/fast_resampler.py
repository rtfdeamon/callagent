import numpy as np
from scipy import signal

def resample_audio(audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Быстрый ресамплинг аудио с использованием полифазной фильтрации.
    Оптимально для речи.
    """
    if orig_sr == target_sr:
        return audio_data
    
    # Вычисляем рациональное соотношение
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    
    return signal.resample_poly(audio_data, up, down)

def float_to_pcm16(audio_float: np.ndarray) -> bytes:
    """Конвертирует float32 [-1.0, 1.0] в PCM 16-bit bytes."""
    audio_pcm = (audio_float * 32767).astype(np.int16)
    return audio_pcm.tobytes()

def pcm16_to_float(audio_bytes: bytes) -> np.ndarray:
    """Конвертирует PCM 16-bit bytes во float32 [-1.0, 1.0]."""
    audio_pcm = np.frombuffer(audio_bytes, dtype=np.int16)
    return audio_pcm.astype(np.float32) / 32768.0
