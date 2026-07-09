from dataclasses import dataclass

from app.domain.entities import Speaker
from app.domain.speech_segment import SpeechSegment
from app.domain.voice_embedding import EmbeddingResult

@dataclass(slots=True)
class PipelineResult:
    speakers: list[Speaker]
    segments: list[SpeechSegment]
    embeddings: EmbeddingResult