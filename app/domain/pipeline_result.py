from dataclasses import dataclass

from app.domain.speech_segment import SpeechSegment
from app.domain.voice_embedding import EmbeddingResult

@dataclass(slots=True)
class PipelineResult:
    segments: list[SpeechSegment]
    embeddings: EmbeddingResult