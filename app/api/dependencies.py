from functools import lru_cache

from app.application.meeting import MeetingService
from app.infrastructure.service_factory import build_speaker_identification_service
from app.infrastructure.stt.factory import STTFactory
from app.pipeline.alignment import AlignmentProcessor
from app.pipeline.diarization import DiarizationProcessor
from app.pipeline.embedding import EmbeddingProcessor
from app.pipeline.meeting_pipeline import MeetingPipeline
from app.infrastructure.audio.loader import AudioLoader
from app.pipeline.stt import STTProcessor


@lru_cache
def get_service() -> MeetingService:
    return MeetingService(
        MeetingPipeline(
            diarization=DiarizationProcessor(),
            stt=STTProcessor(STTFactory.create()),
            alignment=AlignmentProcessor(),
            embedding=EmbeddingProcessor(),
            speaker_identification=build_speaker_identification_service(),
        ),
        loader=AudioLoader(),
    )


