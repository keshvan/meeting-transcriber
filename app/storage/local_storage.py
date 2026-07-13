import asyncio
import soundfile as sf
import torch
import numpy as np
from pathlib import Path

from app.domain.raw_audio import RawAudio

class LocalAudioStorage:
    def __init__(self, base_dir: str):
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
    
    def _resolve(self, key: str) -> Path:
        path = (self._base_dir / key).resolve()
        if not str(path).startswith(str(self._base_dir.resolve())):
            raise ValueError("Invalid key: path traversal detected")
        return path

    def _write_sync(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def _read_waveform_sync(self, path: Path) -> RawAudio:
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        waveform = torch.from_numpy(audio)
        return RawAudio(waveform=waveform, sample_rate=sr)
    
    async def save(self, key: str, data: bytes) -> str:
        path = self._resolve(key)
        await asyncio.to_thread(self._write_sync, path, data)
        return f"file://{path}"

    async def load(self, key: str) -> RawAudio:
        path = self._resolve(key)
        if not path.exists():
            raise FileNotFoundError(f"Audio not found: {key}")
        return await asyncio.to_thread(self._read_waveform_sync, path)

    async def get_url(self, key: str) -> str:
        return f"file://{self._resolve(key)}"

    async def delete(self, key: str) -> None:
        path = self._resolve(key)
        await asyncio.to_thread(lambda: path.unlink(missing_ok=True))

    async def exists(self, key: str) -> bool:
        path = self._resolve(key)
        return await asyncio.to_thread(path.exists)