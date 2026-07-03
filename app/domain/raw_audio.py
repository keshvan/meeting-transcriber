from dataclasses import dataclass

import torch

@dataclass(frozen=True)
class RawAudio:
    waveform: torch.Tensor
    sample_rate: int