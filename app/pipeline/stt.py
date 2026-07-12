from app.application.ports.stt_client import STTClient
from app.domain.raw_audio import RawAudio
from app.domain.transcript import Transcription

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