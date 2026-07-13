from fastapi import Depends

from app.dependencies.database import get_person_repository, get_speaker_repository
from app.dependencies.embedding_repository import get_embedding_repository
from app.dependencies.identification_service import get_identification_service
from app.services.speaker_service import SpeakerService


def get_speaker_service(
    speaker_repository = Depends(get_speaker_repository),
    person_repository = Depends(get_person_repository),
    embedding_repository = Depends(get_embedding_repository),
    identification_service = Depends(get_identification_service),
):

    return SpeakerService(
        speaker_repository=speaker_repository,
        person_repository=person_repository,
        embedding_repository=embedding_repository,
        identification_service=identification_service,
    )