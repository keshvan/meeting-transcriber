from datetime import datetime, timezone

import pytest

from app.config.config import QdrantConfig
from app.domain.voice_embedding import VoiceEmbedding

pytest.importorskip("qdrant_client")

from app.infrastructure.qdrant_voice_embedding_repository import (
    QdrantVoiceEmbeddingRepository,
)


class FakeQdrantClient:
    def __init__(self):
        self.upserts = []

    def upsert(self, collection_name, points):
        self.upserts.append((collection_name, points))


def _repository(client):
    return QdrantVoiceEmbeddingRepository(
        client=client,
        config=QdrantConfig(
            url="http://localhost:6333",
            api_key=None,
            timeout=5.0,
            prefer_grpc=False,
            centroids_collection="voice_centroids",
            embeddings_collection="voice_embeddings",
            vector_size=2,
            distance="COSINE",
            create_collections=True,
            create_payload_indexes=True,
            on_disk_vectors=False,
        ),
    )


def test_embedding_payload_contains_employee_metadata():
    client = FakeQdrantClient()
    repository = _repository(client)
    updated_at = datetime(2026, 7, 2, tzinfo=timezone.utc)

    repository.upsert_embedding(
        VoiceEmbedding(
            person_id="person-1",
            person_name="Alice",
            vector=[0.1, 0.2],
        ),
        embedding_count=3,
        updated_at=updated_at,
    )

    _, points = client.upserts[0]
    payload = points[0].payload

    assert payload == {
        "person_id": "person-1",
        "person_name": "Alice",
        "embedding_count": 3,
        "updated_at": "2026-07-02T00:00:00+00:00",
    }


def test_centroid_payload_contains_employee_metadata():
    client = FakeQdrantClient()
    repository = _repository(client)
    updated_at = datetime(2026, 7, 2, tzinfo=timezone.utc)

    repository.upsert_centroid(
        person_id="person-1",
        person_name="Alice",
        vector=[0.1, 0.2],
        embedding_count=3,
        updated_at=updated_at,
    )

    _, points = client.upserts[0]
    payload = points[0].payload

    assert payload == {
        "person_id": "person-1",
        "person_name": "Alice",
        "embedding_count": 3,
        "updated_at": "2026-07-02T00:00:00+00:00",
    }
