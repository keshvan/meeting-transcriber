from dataclasses import dataclass
from uuid import UUID

@dataclass
class MeetingCreatedEvent:
    meeting_id: UUID
    audio_key: str
    contact: str