from dataclasses import dataclass, field

@dataclass(frozen=True)
class Word:
    text: str
    start_ms: int
    end_ms: int

@dataclass(frozen=True)
class Transcription:
    text: str
    words: list[Word] = field(default_factory=list)