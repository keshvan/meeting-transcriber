from fastapi import Depends

from app.services.meeting_service import MeetingService
from app.dependencies.storage import get_audio_storage
from app.dependencies.redis_publisher import get_redis_publisher
from app.dependencies.database import get_meeting_repository


def get_meeting_service(
    audio_storage = Depends(get_audio_storage),
    repository = Depends(get_meeting_repository),
    publisher = Depends(get_redis_publisher),
):
    return MeetingService(
        audio_storage=audio_storage,
        meeting_repository=repository,
        publisher=publisher,
    )

