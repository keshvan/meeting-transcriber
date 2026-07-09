from dataclasses import dataclass
from uuid import UUID

@dataclass
class SpeechSegment:
    start_ms: int
    end_ms: int
    diarization_label: str
    meeting_id: UUID | None  = None
    speaker_id: UUID | None = None
    text: str | None = None          # 
    person_name: str | None = None   # После STT 