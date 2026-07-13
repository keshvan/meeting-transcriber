from fastapi import Request

from app.storage.local_storage import LocalAudioStorage

def get_audio_storage(request: Request) -> LocalAudioStorage:
    return request.app.state.audio_storage