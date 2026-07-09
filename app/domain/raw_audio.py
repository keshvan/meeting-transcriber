from dataclasses import dataclass

import torch

@dataclass(frozen=True)
class RawAudio:
    waveform: torch.Tensor
    sample_rate: int

def cut_audio(audio: RawAudio, start_ms: int, end_ms: int) -> RawAudio:
    start_sample = int(start_ms * audio.sample_rate / 1000)
    end_sample = int(end_ms * audio.sample_rate / 1000)

    return RawAudio(
        waveform=audio.waveform[start_sample:end_sample],
        sample_rate=audio.sample_rate,
    )