from asyncio import Protocol
import base64
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID
import uuid

from fastapi import UploadFile
from sqlalchemy.orm import Session, sessionmaker

from app.db.meeting_repository import PostgresMeetingRepository
from app.db.person_repository import PostgresPersonRepository
from app.db.segment_repository import PostgresSegmentRepository
from app.db.speaker_repository import PostgresSpeakerRepository
from app.domain.entities import Meeting, MeetingStatus, Segment, Speaker
from app.domain.event import MeetingCreatedEvent
from app.domain.pipeline_result import PipelineResult
from app.domain.raw_audio import RawAudio
from app.formatter.formatter import Formatter
from app.pipeline.meeting_pipeline import MeetingPipeline
from app.services.smtp.smtp import SmtpService

class AudioStorage(Protocol):
    async def load(self, key: str) -> RawAudio: ... 

class MeetingRepository(Protocol):
    def update_status(self, meeting_id: UUID, status: str) -> None: ...

class SpeakerRepository(Protocol):
    def bulk_create(self, speakers: list[Speaker]) -> None: ...
    def commit(self) -> None: ...

class SegmentRepository(Protocol):
    def bulk_create(self, segments: list[Segment]) -> None: ...
    def commit(self) -> None: ...


class MeetingWorker:
    def __init__(
        self,
        pipeline: MeetingPipeline,
        audio_storage: AudioStorage,
        formatter: Formatter,
        smtp_service: SmtpService,
        session_factory
    ):
        self.pipeline = pipeline
        self.audio_storage = audio_storage
        self.formatter = formatter
        self.smtp_service = smtp_service
        self.session_factory = session_factory

    async def handle(self, event: MeetingCreatedEvent) -> None:
        session = self.session_factory()

        try:
            meeting_repository = PostgresMeetingRepository(session)
            speaker_repository = PostgresSpeakerRepository(session)
            segment_repository = PostgresSegmentRepository(session)
            person_repository = PostgresPersonRepository(session)

            result = await self.pipeline.process(
                meeting_id=event.meeting_id,
                audio_path=event.audio_key,
                person_repository=person_repository
            )

            speaker_repository.bulk_create(result.speakers)
            segment_repository.bulk_create(result.segments)

            speaker_repository.commit()
            segment_repository.commit()

            meeting_repository.update_status(
                event.meeting_id,
                MeetingStatus.WAITING_ADMIN,
            )
            meeting_repository.commit()

            merged = self.formatter.merge_segments(result.segments)

            txt = self.formatter.to_txt(merged)
            html = self.formatter.to_html_preview(merged)

            self.smtp_service.send_transcript(
                recipient=event.contact,
                subject="Расшифровка встречи",
                html_preview=html,
                txt_content=txt,
            )

        except Exception:
            meeting_repository.update_status(
                event.meeting_id,
                MeetingStatus.FAILED,
            )
            meeting_repository.commit()
            raise
    