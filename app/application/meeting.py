import asyncio
import base64
from datetime import datetime, timezone
from typing import Optional
import uuid

from fastapi import UploadFile

from app.application.formatter import Formatter
from app.application.ports.repositories import MeetingRepository, SpeakerRepository, SegmentRepository
from app.application.smtp import SmtpService
from app.domain.entities import Meeting, MeetingStatus
from app.infrastructure.audio.audio_loader import AudioLoader
from app.infrastructure.db.session import get_session
from app.pipeline.meeting_pipeline import MeetingPipeline

class MeetingService:
    def __init__(
        self,
        pipeline: MeetingPipeline,
        meeting_repository: MeetingRepository,
        speaker_repository: SpeakerRepository,
        segment_repository: SegmentRepository,
        formatter: Formatter,
        smtp_service: SmtpService
    ):
        self.pipeline = pipeline
        self.meeting_repository = meeting_repository
        self.speaker_repository = speaker_repository
        self.segment_repository = segment_repository
        self.formatter = formatter
        self.smtp_service = smtp_service

    def process(self, meeting_id: uuid.UUID, audio: bytes, email: str):
        try:
            result = self.pipeline.process(meeting_id, audio)

            self.speaker_repository.bulk_create(result.speakers)
            self.speaker_repository.commit()

            self.segment_repository.bulk_create(result.segments)
            self.segment_repository.commit()

            self.meeting_repository.update_status(
                meeting_id,
                MeetingStatus.WAITING_ADMIN,
            )
            self.meeting_repository.commit()

            html = self.formatter.to_html_preview(result.segments)
            txt = self.formatter.to_txt(result.segments)

            self.smtp_service.send_transcript(
                recipient=email,
                subject="Стенограмма встречи",
                html_preview=html,
                txt_content=txt
            )

        except Exception:
            self.meeting_repository.update_status(
                meeting_id,
                MeetingStatus.FAILED,
            )
            self.meeting_repository.commit()
            raise
    
    async def create_processing_meeting(
        self,
        *,
        file: UploadFile | None,
        audio_base64: str | None,
    ):
        raw_bytes = await self._read_input_bytes(
            file=file,
            audio_base64=audio_base64,
        )

        audio = AudioLoader.from_bytes(raw_bytes)
        meeting_id = uuid.uuid4()

        meeting = self.meeting_repository.create(
            Meeting(
                id=meeting_id,
                created_at=datetime.now(timezone.utc),
                status=MeetingStatus.PROCESSING,
            )
        )

        self.meeting_repository.commit()

        return meeting.id, audio
    
    @staticmethod
    async def _read_input_bytes(
        *, file: Optional[UploadFile], audio_base64: Optional[str]
    ) -> bytes:
        if file:
            return await file.read()
        if audio_base64:
            return base64.b64decode(audio_base64)
        raise ValueError("No audio provided")

async def process_meeting_background(
        meeting_id: uuid.UUID,
        audio: bytes,
        email: str
    ):
        from app.api.dependencies import build_meeting_service

        with get_session() as session:
            service = build_meeting_service(session)
            await asyncio.to_thread(service.process(meeting_id=meeting_id, audio=audio, email=email))
    