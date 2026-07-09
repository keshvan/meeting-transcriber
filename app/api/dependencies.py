from functools import lru_cache
from typing import Iterator

from fastapi import Depends
from sqlalchemy.orm import Session

from app.infrastructure.db.session import get_session
from app.infrastructure.db.meeting_repository import PostgresMeetingRepository
from app.infrastructure.db.speaker_repository import PostgresSpeakerRepository
from app.infrastructure.db.segment_repository import PostgresSegmentRepository
from app.infrastructure.db.person_repository import PostgresPersonRepository
from app.infrastructure.service_factory import build_qdrant_client, build_embedding_protector
from app.infrastructure.audio.local_storage import LocalAudioStorage
from app.infrastructure.stt.factory import STTFactory
from app.application.speaker_identification import SpeakerIdentificationService
from app.application.meeting import MeetingService
from app.pipeline.meeting_pipeline import MeetingPipeline
from app.pipeline.diarization import DiarizationProcessor
from app.pipeline.embedding import EmbeddingProcessor
from app.pipeline.alignment import AlignmentProcessor
from app.pipeline.stt import STTProcessor

from app.config.settings import settings

@lru_cache
def get_diarization_processor() -> DiarizationProcessor:
    return DiarizationProcessor()

@lru_cache
def get_stt_processor() -> STTProcessor:
    return STTProcessor(STTFactory.create())

@lru_cache
def get_alignment_processor() -> AlignmentProcessor:
    return AlignmentProcessor()

@lru_cache
def get_embedding_processor() -> EmbeddingProcessor:
    return EmbeddingProcessor()

@lru_cache
def get_audio_storage():
    return LocalAudioStorage(settings.base_dir)


@lru_cache
def _voice_embedding_backend():
    from app.infrastructure.qdrant_voice_embedding_repository import QdrantVoiceEmbeddingRepository

    client = build_qdrant_client()
    repository = QdrantVoiceEmbeddingRepository(client=client, settings=settings.qdrant)
    repository.ensure_storage()
    protector = build_embedding_protector(
        settings=settings.embedding_protection,
        vector_size=settings.qdrant.vector_size,
    )
    return repository, protector

def get_db_session() -> Iterator[Session]:
    with get_session() as session:
        yield session

def get_person_repository(session: Session = Depends(get_db_session)) -> PostgresPersonRepository:
    return PostgresPersonRepository(session)

def get_speaker_repository(session: Session = Depends(get_db_session)) -> PostgresSpeakerRepository:
    return PostgresSpeakerRepository(session)

def get_segment_repository(session: Session = Depends(get_db_session)) -> PostgresSegmentRepository:
    return PostgresSegmentRepository(session)

def get_meeting_repository(session: Session = Depends(get_db_session)) -> PostgresMeetingRepository:
    return PostgresMeetingRepository(session)

def get_speaker_identification_service() -> SpeakerIdentificationService:
    
    voice_repository, protector = _voice_embedding_backend()
    return SpeakerIdentificationService(
        repository=voice_repository,
        protector=protector,
        config=settings.speaker_identification,
    )


def get_service(
    audio_storage=Depends(get_audio_storage),
    meeting_repository: PostgresMeetingRepository = Depends(get_meeting_repository),
    speaker_repository: PostgresSpeakerRepository = Depends(get_speaker_repository),
    segment_repository: PostgresSegmentRepository = Depends(get_segment_repository),
    speaker_identification: SpeakerIdentificationService = Depends(get_speaker_identification_service),
    diarization: DiarizationProcessor = Depends(get_diarization_processor),
    stt: STTProcessor = Depends(get_stt_processor),
    alignment: AlignmentProcessor = Depends(get_alignment_processor),
    embedding: EmbeddingProcessor = Depends(get_embedding_processor),
) -> MeetingService:
    return MeetingService(
        pipeline=MeetingPipeline(
            diarization=diarization,
            stt=stt,
            alignment=alignment,
            embedding=embedding,
            speaker_identification=speaker_identification,
        ),
        audio_storage=audio_storage,
        meeting_repository=meeting_repository,
        speaker_repository=speaker_repository,
        segment_repository=segment_repository,
    )