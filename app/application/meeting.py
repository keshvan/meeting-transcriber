from typing import Optional

from fastapi import UploadFile

from app.domain.pipeline_result import PipelineResult
from app.infrastructure.audio.loader import AudioLoader
from app.pipeline.meeting_pipeline import MeetingPipeline


class MeetingService:
    def __init__(self, pipeline: MeetingPipeline, loader: AudioLoader):
        self.pipeline = pipeline
        self.loader = loader

    async def process(self, *, file: Optional[UploadFile], audio_base64: Optional[str]) -> PipelineResult:
        if file:
            audio = await self.loader.from_upload(file)
        elif audio_base64:
            audio = self.loader.from_base64(audio_base64)
        else:
            raise ValueError("No audio provided")

        return self.pipeline.process(audio)