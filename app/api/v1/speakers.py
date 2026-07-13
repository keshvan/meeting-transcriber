from uuid import UUID

from fastapi import APIRouter, Depends

from app.dependencies.speaker_service import get_speaker_service
from app.schemas.speaker import UpdateSpeakerRequest
from app.services.speaker_service import SpeakerService

router = APIRouter()

@router.post(
    "/meetings/{meetingId}/speakers",
)
async def update_speaker(
    meetingId: str,
    request: UpdateSpeakerRequest,
    service: SpeakerService = Depends(get_speaker_service),
):

    service.resolve_unknown_speaker(
        meeting_id=UUID(meetingId),
        speaker_id=request.speaker_id,
        full_name=request.full_name,
    )

    return {
        "status": "OK"
    }