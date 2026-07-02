from dataclasses import dataclass
from datetime import datetime, timezone
from math import sqrt

from app.application.embedding_protection import EmbeddingProtector
from app.config.config import SpeakerIdentificationConfig
from app.domain.voice_embedding import (
    SpeakerIdentificationResult,
    Vector,
    VoiceEmbedding,
)
from app.domain.voice_embedding_repository import VoiceEmbeddingRepository


@dataclass(frozen=True)
class SpeakerIdentificationService:
    repository: VoiceEmbeddingRepository
    protector: EmbeddingProtector
    config: SpeakerIdentificationConfig

    def identify(self, vector: Vector) -> SpeakerIdentificationResult:
        protected_vector = self.protector.protect(vector)
        centroid_candidates = self.repository.search_centroids(
            protected_vector,
            limit=self.config.centroid_top_k,
        )

        best_person_id: str | None = None
        best_person_name = self.config.unknown_speaker_name
        best_score = 0.0

        for candidate in centroid_candidates:
            matches = self.repository.search_person_embeddings(
                person_id=candidate.person_id,
                vector=protected_vector,
                limit=self.config.sample_limit_per_person,
            )
            for match in matches:
                if match.score > best_score:
                    best_person_id = match.person_id
                    best_person_name = match.person_name
                    best_score = match.score

        is_known = best_score >= self.config.similarity_threshold
        if not is_known:
            return SpeakerIdentificationResult(
                person_id=None,
                person_name=self.config.unknown_speaker_name,
                score=best_score,
                is_known=False,
            )

        return SpeakerIdentificationResult(
            person_id=best_person_id,
            person_name=best_person_name,
            score=best_score,
            is_known=True,
        )

    def save_confirmed_embedding(self, embedding: VoiceEmbedding) -> None:
        updated_at = datetime.now(timezone.utc)
        protected_embedding = VoiceEmbedding(
            person_id=embedding.person_id,
            person_name=embedding.person_name,
            vector=self.protector.protect(embedding.vector),
            embedding_id=embedding.embedding_id,
            updated_at=updated_at,
            metadata=embedding.metadata,
        )
        next_vector, next_count = self._next_centroid(protected_embedding)
        self.repository.upsert_embedding(
            protected_embedding,
            embedding_count=next_count,
            updated_at=updated_at,
        )
        self.repository.upsert_centroid(
            person_id=embedding.person_id,
            person_name=embedding.person_name,
            vector=next_vector,
            embedding_count=next_count,
            updated_at=updated_at,
        )

    def _next_centroid(self, embedding: VoiceEmbedding) -> tuple[Vector, int]:
        centroid = self.repository.get_centroid(embedding.person_id)
        if centroid is None:
            return embedding.vector, 1

        next_count = centroid.embedding_count + 1
        next_vector = [
            ((current * centroid.embedding_count) + new_value) / next_count
            for current, new_value in zip(centroid.vector, embedding.vector, strict=True)
        ]
        return next_vector, next_count


def cosine_similarity(left: Vector, right: Vector) -> float:
    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (left_norm * right_norm)
