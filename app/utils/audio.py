from pathlib import Path

import numpy as np
import soundfile as sf

def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return audio.astype(np.float32), sample_rate

def cut_audio(audio: np.ndarray, sample_rate: int, start: float, end: float) -> np.ndarray:
    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)

    return audio[start_sample:end_sample]
