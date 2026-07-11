from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from qdrant_client import QdrantClient, models

from app.config.settings import QdrantSettings
from app.domain.voice_embedding import (
    PersonCentroid,
    SpeakerCandidate,
    Vector,
    VoiceEmbedding,
)


PERSON_ID_KEY = "person_id"
SPEAKER_ID_KEY = "speaker_id"
PERSON_NAME_KEY = "person_name"
EMBEDDING_COUNT_KEY = "embedding_count"
UPDATED_AT_KEY = "updated_at"


@dataclass
class QdrantVoiceEmbeddingRepository:
    client: QdrantClient
    settings: QdrantSettings

    def ensure_storage(self) -> None:
        if not self.settings.create_collections:
            return

        self._ensure_collection(self.settings.centroids_collection)
        self._ensure_collection(self.settings.embeddings_collection)

        if self.settings.create_payload_indexes:
            self._ensure_keyword_index(self.settings.centroids_collection, PERSON_ID_KEY)
            self._ensure_keyword_index(self.settings.embeddings_collection, PERSON_ID_KEY)

    def upsert_embedding(
        self,
        embedding: VoiceEmbedding,
        embedding_count: int,
        updated_at: datetime,
    ) -> None:
        self._validate_vector(embedding.vector)
        payload = {
            **embedding.metadata,
            PERSON_ID_KEY: embedding.person_id,
            PERSON_NAME_KEY: embedding.person_name,
            SPEAKER_ID_KEY: str(embedding.speaker_id),
            "embedding_id": str(embedding.embedding_id),
            EMBEDDING_COUNT_KEY: embedding_count,
            UPDATED_AT_KEY: updated_at.isoformat(),
        }
        self.client.upsert(
            collection_name=self.settings.embeddings_collection,
            points=[
                models.PointStruct(
                    id=self._point_id(
                        self.settings.embeddings_collection,
                        embedding.embedding_id,
                    ),
                    vector=embedding.vector,
                    payload=payload,
                )
            ],
        )

    def get_centroid(self, person_id: str) -> PersonCentroid | None:
        records = self.client.retrieve(
            collection_name=self.settings.centroids_collection,
            ids=[self._point_id(self.settings.centroids_collection, person_id)],
            with_payload=True,
            with_vectors=True,
        )
        if not records:
            return None

        record = records[0]
        payload = record.payload or {}
        vector = self._vector_from_point(record)
        if vector is None:
            return None

        return PersonCentroid(
            person_id=str(payload.get(PERSON_ID_KEY, person_id)),
            person_name=str(payload.get(PERSON_NAME_KEY, "")),
            vector=vector,
            embedding_count=int(payload.get(EMBEDDING_COUNT_KEY, 0)),
            updated_at=self._parse_updated_at(payload.get(UPDATED_AT_KEY)),
        )

    def upsert_centroid(
        self,
        person_id: str,
        person_name: str,
        vector: Vector,
        embedding_count: int,
        updated_at: datetime,
    ) -> None:
        self._validate_vector(vector)
        self.client.upsert(
            collection_name=self.settings.centroids_collection,
            points=[
                models.PointStruct(
                    id=self._point_id(self.settings.centroids_collection, person_id),
                    vector=vector,
                    payload={
                        PERSON_ID_KEY: person_id,
                        PERSON_NAME_KEY: person_name,
                        EMBEDDING_COUNT_KEY: embedding_count,
                        UPDATED_AT_KEY: updated_at.isoformat(),
                    },
                )
            ],
        )

    def search_centroids(self, vector: Vector, limit: int) -> list[SpeakerCandidate]:
        self._validate_vector(vector)
        response = self.client.query_points(
            collection_name=self.settings.centroids_collection,
            query=vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        return [self._candidate_from_point(point) for point in self._points(response)]

    def search_person_embeddings(
        self,
        person_id: str,
        vector: Vector,
        limit: int,
    ) -> list[SpeakerCandidate]:
        self._validate_vector(vector)
        response = self.client.query_points(
            collection_name=self.settings.embeddings_collection,
            query=vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=PERSON_ID_KEY,
                        match=models.MatchValue(value=person_id),
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        return [self._candidate_from_point(point) for point in self._points(response)]

    def get_embeddings_by_speaker(self, speaker_id: UUID) -> list[VoiceEmbedding]:
        """Возвращает все эмбеддинги для данного speaker_id (обычно один)."""
        response = self.client.scroll(
            collection_name=self.settings.embeddings_collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=SPEAKER_ID_KEY,
                        match=models.MatchValue(value=str(speaker_id)),
                    )
                ]
            ),
            with_payload=True,
            with_vectors=True,
            limit=10,  # обычно хватит одного, но на всякий случай
        )
        points = response[0]  # scroll возвращает (points, next_offset)
        return [self._embedding_from_point(p) for p in points]

    def _embedding_from_point(self, point: Any) -> VoiceEmbedding:
        payload = point.payload or {}
        vector = self._vector_from_point(point)
        if vector is None:
            raise ValueError("Point has no vector")
        return VoiceEmbedding(
            speaker_id=UUID(payload.get(SPEAKER_ID_KEY)),
            person_id=payload.get(PERSON_ID_KEY),
            person_name=payload.get(PERSON_NAME_KEY, ""),
            vector=vector,
            embedding_id=UUID(payload.get("embedding_id")),
            updated_at=self._parse_updated_at(payload.get(UPDATED_AT_KEY)),
            metadata={k: v for k, v in payload.items() if k not in (SPEAKER_ID_KEY, PERSON_ID_KEY, PERSON_NAME_KEY, "embedding_id", EMBEDDING_COUNT_KEY, UPDATED_AT_KEY)},
        )

    def _ensure_collection(self, collection_name: str) -> None:
        if self.client.collection_exists(collection_name):
            return

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=self.settings.vector_size,
                distance=self._distance(),
                on_disk=self.settings.on_disk_vectors,
            ),
        )

    def _ensure_keyword_index(self, collection_name: str, field_name: str) -> None:
        try:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            # Qdrant returns an error when the index already exists; that is fine here.
            return

    def _distance(self) -> models.Distance:
        try:
            return models.Distance[self.settings.distance.upper()]
        except KeyError as exc:
            allowed = ", ".join(distance.name for distance in models.Distance)
            raise ValueError(
                f"Unsupported Qdrant distance {self.settings.distance!r}. "
                f"Allowed values: {allowed}."
            ) from exc

    def _validate_vector(self, vector: Vector) -> None:
        if len(vector) != self.settings.vector_size:
            raise ValueError(
                f"Voice embedding vector must have size {self.settings.vector_size}, "
                f"got {len(vector)}."
            )

    def _candidate_from_point(self, point: Any) -> SpeakerCandidate:
        payload = point.payload or {}
        return SpeakerCandidate(
            person_id=str(payload.get(PERSON_ID_KEY, point.id)),
            person_name=str(payload.get(PERSON_NAME_KEY, "")),
            score=float(point.score),
        )

    def _point_id(self, collection_name: str, external_id: str) -> str:
        return str(uuid5(NAMESPACE_URL, f"{collection_name}:{external_id}"))

    def _points(self, response: Any) -> list[Any]:
        if hasattr(response, "points"):
            return list(response.points)

        return list(response)

    def _vector_from_point(self, point: Any) -> Vector | None:
        vector = getattr(point, "vector", None)
        if vector is None:
            return None

        if isinstance(vector, dict):
            vector = vector.get("")

        return list(vector) if vector is not None else None

    def _parse_updated_at(self, value: Any) -> datetime:
        if isinstance(value, str):
            return datetime.fromisoformat(value)

        return datetime.min
