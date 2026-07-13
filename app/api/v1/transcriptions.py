from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException

from app.dependencies.meeting_service import get_meeting_service
from app.schemas.transcription import CreateTranscriptionResponse
from app.services.meeting_service import MeetingService


router = APIRouter()

@router.post(
    "/transcriptions",
    status_code=202,
    response_model=CreateTranscriptionResponse,
)
async def create_transcription(
    file: Optional[UploadFile] = File(None),
    audio_base64: Optional[str] = Form(None),
    contact: str = Form(...),
    service: MeetingService = Depends(get_meeting_service),
):

    if not file and not audio_base64:
        raise HTTPException(
            status_code=400,
            detail="file or audio_base64 required"
        )

    meeting = await service.enqueue(
        file=file,
        audio_base64=audio_base64,
        contact=contact,
    )

    return CreateTranscriptionResponse(
        meeting_id=meeting.id,
        status=meeting.status.value,
    )