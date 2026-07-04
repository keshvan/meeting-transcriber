from dataclasses import dataclass, field

@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float

@dataclass(frozen=True)
class Transcription:
    text: str
    words: list[Word] = field(default_factory=list)