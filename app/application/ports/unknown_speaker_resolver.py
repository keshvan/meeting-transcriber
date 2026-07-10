from typing import Protocol

from app.domain.entities import Speaker
from app.domain.voice_embedding import VoiceEmbedding

class UnknownSpeakerResolver(Protocol):
    def resolve(
        self,
        speakers: list[Speaker],
        embeddings: list[VoiceEmbedding]
    ) -> None: ...