from pydantic import BaseModel


class UpdateSpeakerRequest(BaseModel):
    speaker_id: str
    full_name: str