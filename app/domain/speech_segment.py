from dataclasses import dataclass
from uuid import UUID

@dataclass
class SpeechSegment:
    start_ms: int
    end_ms: int
    speaker_id: str
    meeting_id: UUID | None  = None
    text: str | None = None          # 
    person_name: str | None = None   # После STT 