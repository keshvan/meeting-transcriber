import base64
import io
import soundfile as sf
import torch
import numpy as np

from app.domain.raw_audio import RawAudio


class AudioLoader:
    @staticmethod
    def from_bytes(data: bytes) -> RawAudio:
        waveform, sample_rate = sf.read(io.BytesIO(data))

        if waveform.ndim > 1:
            waveform.mean(axis=1)

        waveform = torch.from_numpy(waveform.astype(np.float32))

        return RawAudio(
            waveform=waveform,
            sample_rate=sample_rate,
        )