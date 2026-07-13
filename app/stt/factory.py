from app.config.settings import settings
from app.pipeline.stt import STTClient



class STTFactory:
    @staticmethod
    def create() -> STTClient:
        provider = settings.stt_provider

        if provider == "whisper_local":
            from app.stt.faster_whisper import FasterWhisperSTT
            return FasterWhisperSTT(model_size=settings.stt_model, device="auto", compute_type="float32")
        else:
            raise ValueError(f"Unknown STT provider: {provider}")