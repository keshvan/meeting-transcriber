import base64
from datetime import datetime, timezone
from typing import Optional, Protocol
import uuid

from fastapi import UploadFile

from app.domain.entities import Meeting, MeetingStatus
from app.domain.event import MeetingCreatedEvent

class AudioStorage(Protocol):
    async def save(self, key: str, data: bytes) -> str: ...

class MeetingRepository(Protocol):
    def create(self, meeting: Meeting) -> Meeting: ...
    def update_status(self, meeting_id: uuid.UUID, status: str) -> None: ...
    def commit(self) -> None: ...

class Publisher(Protocol):
    async def publish(self, event: MeetingCreatedEvent) -> None: ...

class MeetingService:
    def __init__(
        self,
        audio_storage: AudioStorage,
        meeting_repository: MeetingRepository,
        publisher: Publisher
    ):
        self.audio_storage = audio_storage
        self.meeting_repository = meeting_repository
        self.publisher = publisher

    async def enqueue(self, *, file: Optional[UploadFile], audio_base64: Optional[str], contact: str) -> Meeting:
        raw_bytes = await self._read_input_bytes(file=file, audio_base64=audio_base64)

        meeting_id = uuid.uuid4()
        audio_key = f"{meeting_id}/audio.wav"

        await self.audio_storage.save(audio_key, raw_bytes)

        meeting = self.meeting_repository.create(
            Meeting(
                id=meeting_id,
                created_at=datetime.now(timezone.utc),
                audio_key=audio_key,
                status=MeetingStatus.PROCESSING,
            )
        )

        await self.publisher.publish(
            MeetingCreatedEvent(
                meeting_id=meeting_id,
                audio_key=audio_key,
                contact=contact
            )
        )

        return meeting

    @staticmethod
    async def _read_input_bytes(
        *, file: Optional[UploadFile], audio_base64: Optional[str]
    ) -> bytes:
        if file:
            return await file.read()
        if audio_base64:
            return base64.b64decode(audio_base64)
        raise ValueError("No audio provided")