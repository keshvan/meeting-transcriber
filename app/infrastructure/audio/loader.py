import base64
import tempfile
import soundfile as sf
import torch
import numpy as np

from fastapi import UploadFile

from app.domain.raw_audio import RawAudio

class AudioLoader:
    async def from_upload(self, file: UploadFile) -> RawAudio:
        raw_bytes = await file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        waveform, sr = self._build_waveform(tmp_path)

        return RawAudio(
            waveform=waveform,
            sample_rate=sr,
        )

    def from_base64(self, data: str) -> RawAudio:
        raw_bytes = base64.b64decode(data)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        waveform, sr = self._build_waveform(tmp_path)

        return RawAudio(
            waveform=waveform,
            sample_rate=sr,
        )

    def _build_waveform(self, path: str) -> tuple[torch.Tensor, int]:
        audio, sr = sf.read(path)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        waveform = torch.from_numpy(audio.astype(np.float32))

        return (waveform, sr)