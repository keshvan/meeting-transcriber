from app.config.settings import settings
from app.infrastructure.stt.base import STTClient

from app.infrastructure.stt.faster_whisper import FasterWhisperSTT

class STTFactory:
    @staticmethod
    def create() -> STTClient:
        provider = settings.stt_provider

        if provider == "whisper_local":
            return FasterWhisperSTT(model_size=settings.stt_model, device="auto", compute_type="float32")