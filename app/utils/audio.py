from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from app.domain.raw_audio import RawAudio

def load_audio(path: str | Path) -> RawAudio:
    audio, sample_rate = sf.read(path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    waveform = torch.from_numpy(audio.astype(np.float32))
    return RawAudio(waveform=waveform, sample_rate=sample_rate)

def cut_audio(audio: RawAudio, start: float, end: float) -> RawAudio:
    start_sample = int(start * audio.sample_rate)
    end_sample = int(end * audio.sample_rate)

    return RawAudio(
        waveform=audio.waveform[start_sample:end_sample],
        sample_rate=audio.sample_rate,
    )
