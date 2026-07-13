from qdrant_client import QdrantClient

from app.services.embedding_protection.embedding_protection import build_embedding_protector
from app.services.speaker_identification.speaker_identification import SpeakerIdentificationService
from app.config.settings import settings
from app.qdrant.qdrant_voice_embedding_repository import (
    QdrantVoiceEmbeddingRepository,
)


def build_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant.url,
        api_key=settings.qdrant.api_key,
        timeout=settings.qdrant.timeout,
        prefer_grpc=settings.qdrant.prefer_grpc,
    )


def build_speaker_identification_service() -> SpeakerIdentificationService:
    repository = QdrantVoiceEmbeddingRepository(
        client=build_qdrant_client(),
        settings=settings.qdrant,
    )
    repository.ensure_storage()

    protector = build_embedding_protector(
        settings=settings.embedding_protection,
        vector_size=settings.qdrant.vector_size,
    )

    return SpeakerIdentificationService(
        repository=repository,
        protector=protector,
        config=settings.speaker_identification,
    )

