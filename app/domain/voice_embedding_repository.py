from typing import Protocol
from datetime import datetime

from app.domain.voice_embedding import (
    PersonCentroid,
    SpeakerCandidate,
    Vector,
    VoiceEmbedding,
)


class VoiceEmbeddingRepository(Protocol):
    def ensure_storage(self) -> None:
        ...

    def upsert_embedding(
        self,
        embedding: VoiceEmbedding,
        embedding_count: int,
        updated_at: datetime,
    ) -> None:
        ...

    def get_centroid(self, person_id: str) -> PersonCentroid | None:
        ...

    def upsert_centroid(
        self,
        person_id: str,
        person_name: str,
        vector: Vector,
        embedding_count: int,
        updated_at: datetime,
    ) -> None:
        ...

    def search_centroids(self, vector: Vector, limit: int) -> list[SpeakerCandidate]:
        ...

    def search_person_embeddings(
        self,
        person_id: str,
        vector: Vector,
        limit: int,
    ) -> list[SpeakerCandidate]:
        ...
