from functools import lru_cache
from typing import Iterator

from fastapi import Depends
from sqlalchemy.orm import Session

from app.application.formatter import Formatter
from app.application.smtp import SmtpService
from app.infrastructure.db.session import get_session
from app.infrastructure.db.meeting_repository import PostgresMeetingRepository
from app.infrastructure.db.speaker_repository import PostgresSpeakerRepository
from app.infrastructure.db.person_repository import PostgresPersonRepository
from app.infrastructure.service_factory import build_qdrant_client, build_embedding_protector
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

def get_meeting_repository(session: Session = Depends(get_db_session)) -> PostgresMeetingRepository:
    return PostgresMeetingRepository(session)

def get_formatter() -> Formatter:
    return Formatter()

def get_smpt_service() -> SmtpService:
    return SmtpService(
        host=settings.smtp_host,
        port=settings.smtp_port,
        username=settings.smtp_username,
        password=settings.smtp_password,
        sender=settings.smtp_sender
    )

def get_speaker_identification_service() -> SpeakerIdentificationService:
    
    voice_repository, protector = _voice_embedding_backend()
    return SpeakerIdentificationService(
        repository=voice_repository,
        protector=protector,
        config=settings.speaker_identification,
    )

def build_meeting_service(session: Session) -> MeetingService:
    return MeetingService(
        pipeline=MeetingPipeline(
            diarization=get_diarization_processor(),
            stt=get_stt_processor(),
            alignment=get_alignment_processor(),
            embedding=get_embedding_processor(),
            speaker_identification=get_speaker_identification_service(),
        ),
        meeting_repository=PostgresMeetingRepository(session),
        speaker_repository=PostgresSpeakerRepository(session),
        formatter=get_formatter(),
        smtp_service=get_smpt_service()
    )

def get_service(
    session: Session = Depends(get_db_session),
) -> MeetingService:
    return build_meeting_service(session)