from uuid import UUID

from pydantic import BaseModel


class CreateTranscriptionResponse(BaseModel):
    meeting_id: UUID
    status: str