import base64
from datetime import datetime, timezone
from typing import Optional
import uuid

from fastapi import UploadFile

from app.application.ports.audio_storage import AudioStorage
from app.application.ports.repositories import MeetingRepository, SpeakerRepository, SegmentRepository
from app.domain.entities import Meeting, MeetingStatus
from app.domain.pipeline_result import PipelineResult
from app.pipeline.meeting_pipeline import MeetingPipeline

class MeetingService:
    def __init__(
        self,
        pipeline: MeetingPipeline,
        audio_storage: AudioStorage,
        meeting_repository: MeetingRepository,
        speaker_repository: SpeakerRepository,
        segment_repository: SegmentRepository,
    ):
        self.pipeline = pipeline
        self.audio_storage = audio_storage
        self.meeting_repository = meeting_repository
        self.speaker_repository = speaker_repository
        self.segment_repository = segment_repository

    async def process(self, *, file: Optional[UploadFile], audio_base64: Optional[str]) -> PipelineResult:
        raw_bytes = await self._read_input_bytes(file=file, audio_base64=audio_base64)

        meeting_id = uuid.uuid4()
        storage_key = f"{meeting_id}/{uuid.uuid4()}.wav"

        meeting = self.meeting_repository.create(
            Meeting(
                id=meeting_id,
                created_at=datetime.now(timezone.utc),
                audio_key=storage_key,
                status=MeetingStatus.PROCESSING,
            )
        )

        self.meeting_repository.commit()

        await self.audio_storage.save(storage_key, raw_bytes)

        audio = await self.audio_storage.load(storage_key)

        try:
            result = self.pipeline.process(meeting_id, audio)
        except Exception:
            self.meeting_repository.update_status(meeting.id, MeetingStatus.FAILED)
            self.meeting_repository.commit()
            raise

        self.speaker_repository.bulk_create(result.speakers)
        self.speaker_repository.commit()
        self.segment_repository.bulk_create(result.segments)
        self.segment_repository.commit()
        self.meeting_repository.update_status(meeting.id, MeetingStatus.WAITING_ADMIN)

        return result
    
    @staticmethod
    async def _read_input_bytes(
        *, file: Optional[UploadFile], audio_base64: Optional[str]
    ) -> bytes:
        if file:
            return await file.read()
        if audio_base64:
            return base64.b64decode(audio_base64)
        raise ValueError("No audio provided")