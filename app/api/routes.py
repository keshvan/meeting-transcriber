from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from typing import Optional
import base64
import tempfile

from app.application.meeting import MeetingService
from .dependencies import get_service

router = APIRouter()

@router.post("/process")
async def process_meeting(
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

    result = await service.process(
        file=file,
        audio_base64=audio_base64,
    )

    return {
        "email": email,
        "result": result
    }
    