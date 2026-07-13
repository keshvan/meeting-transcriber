from fastapi import Depends

from app.qdrant.qdrant_voice_embedding_repository import (
    QdrantVoiceEmbeddingRepository,
)
from app.config.settings import settings
from app.dependencies.qdrant import get_qdrant_client


def get_embedding_repository(
    client = Depends(get_qdrant_client),
):
    repository = QdrantVoiceEmbeddingRepository(
        client=client,
        settings=settings.qdrant,
    )

    return repository