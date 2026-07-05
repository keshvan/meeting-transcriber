from app.infrastructure.stt.base import STTClient
from app.domain.raw_audio import RawAudio
from app.domain.transcript import Transcription

class STTProcessor:
    def __init__(
        self,
        client: STTClient
    ):
        self.client = client

    def process(self, audio: RawAudio) -> Transcription:
        return self.client.transcribe(audio)
