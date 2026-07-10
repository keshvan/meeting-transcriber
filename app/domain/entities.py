from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

class MeetingStatus(str, Enum):
    PROCESSING = "PROCESSING"
    WAITING_ADMIN = "WAITING_ADMIN"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class SpeakerStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    IDENTIFIED = "IDENTIFIED"

@dataclass
class Person:
    id: UUID
    name: str
    created_at: datetime | None = None

@dataclass
class Speaker:
    id: UUID
    meeting_id: UUID
    diarization_label: str
    person_id: Optional[UUID]
    status: SpeakerStatus
    embedding_id: Optional[UUID]  # qdrant_point_id

@dataclass
class Segment:
    id: UUID
    meeting_id: UUID
    speaker_id: Optional[UUID]
    start_ms: int
    end_ms: int
    text: str

@dataclass
class Meeting:
    id: UUID
    created_at: datetime
    audio_key: str
    status: MeetingStatus
    duration: Optional[int] = None