from dataclasses import dataclass

import torch

@dataclass(frozen=True)
class RawAudio:
    waveform: torch.Tensor
    sample_rate: int

def cut_audio(audio: RawAudio, start: float, end: float) -> RawAudio:
    start_sample = int(start * audio.sample_rate)
    end_sample = int(end * audio.sample_rate)

    return RawAudio(
        waveform=audio.waveform[start_sample:end_sample],
        sample_rate=audio.sample_rate,
    )