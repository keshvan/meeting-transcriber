# app/dependencies/identification_service.py

from fastapi import Depends

from app.services.speaker_identification.speaker_identification import (
    SpeakerIdentificationService,
)
from app.services.embedding_protection.embedding_protection import (
    build_embedding_protector,
)
from app.dependencies.embedding_repository import (
    get_embedding_repository,
)
from app.config.settings import settings


def get_identification_service(
    repository = Depends(get_embedding_repository),
):

    protector = build_embedding_protector(
        settings=settings.embedding_protection,
        vector_size=settings.qdrant.vector_size,
    )

    return SpeakerIdentificationService(
        repository=repository,
        protector=protector,
        config=settings.speaker_identification,
    )