from typing import Protocol

from app.domain.raw_audio import RawAudio
from app.domain.transcript import Transcription

class STTClient(Protocol):
    def transcribe(
        self,
        audio: RawAudio,
        language: str | None = None
    ) -> Transcription: ...

class STTProcessor:
    def __init__(
        self,
        client: STTClient,
        language: str = "ru"
    ):
        self.client = client
        self.language = language

    def process(self, audio: RawAudio) -> Transcription:
        return self.client.transcribe(audio, language=self.language)