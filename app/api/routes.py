import asyncio

from fastapi import APIRouter, BackgroundTasks, UploadFile, File, Form, Depends, HTTPException
from typing import Optional

from app.application.meeting import MeetingService, process_meeting_background
from .dependencies import get_service

router = APIRouter()

@router.post("/process")
async def process_meeting(
    background_tasks: BackgroundTasks,
    email: str = Form(...),
    file: Optional[UploadFile] = File(None),
    audio_base64: Optional[str] = Form(None),
    service: MeetingService = Depends(get_service)
):
    if not file and not audio_base64:
        raise HTTPException(
            status_code=400,
            detail="Provide either file or audio_base64"
        )

    meeting_id, audio = await service.create_processing_meeting(
        file=file,
        audio_base64=audio_base64,
    )

    asyncio.create_task(
        process_meeting_background(
            meeting_id,
            audio,
            email
        )
    )
    
    return {
        "meeting_id": str(meeting_id),
        "email": email,
        "status": "processing",
    }