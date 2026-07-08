from typing import Protocol

from app.domain.raw_audio import RawAudio
from app.domain.transcript import Transcription

class STTClient(Protocol):
    def transcribe(
        self,
        audio: RawAudio,
        language: str | None = None
    ) -> Transcription: ...