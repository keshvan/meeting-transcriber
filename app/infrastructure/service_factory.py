from qdrant_client import QdrantClient

from app.application.embedding_protection import build_embedding_protector
from app.application.speaker_identification import SpeakerIdentificationService
from app.config.config import config
from app.infrastructure.qdrant_voice_embedding_repository import (
    QdrantVoiceEmbeddingRepository,
)


def build_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=config.qdrant.url,
        api_key=config.qdrant.api_key,
        timeout=config.qdrant.timeout,
        prefer_grpc=config.qdrant.prefer_grpc,
    )


def build_speaker_identification_service() -> SpeakerIdentificationService:
    repository = QdrantVoiceEmbeddingRepository(
        client=build_qdrant_client(),
        config=config.qdrant,
    )
    repository.ensure_storage()

    protector = build_embedding_protector(
        config=config.embedding_protection,
        vector_size=config.qdrant.vector_size,
    )

    return SpeakerIdentificationService(
        repository=repository,
        protector=protector,
        config=config.speaker_identification,
    )

