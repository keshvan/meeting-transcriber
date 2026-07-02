from dataclasses import dataclass

@dataclass
class SpeechSegment:
    start: float
    end: float
    speaker_id: str # SPEAKER_00...
    text: str | None = None          # 
    person_name: str | None = None   # После STT 