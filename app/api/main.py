from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.config.settings import settings
from app.api.v1.transcriptions import router as transcription_router
from app.api.v1.speakers import router as speaker_router
from app.db.init_db import init_db
from app.dependencies.qdrant import build_qdrant_client
from app.qdrant.qdrant_voice_embedding_repository import QdrantVoiceEmbeddingRepository
from app.redis.client import create_redis
from app.redis.publisher import RedisPublisher
from app.storage.local_storage import LocalAudioStorage

@asynccontextmanager
async def lifespan(app: FastAPI):

    redis = create_redis()

    init_db()

    qdrant_client = build_qdrant_client()

    embedding_repository = QdrantVoiceEmbeddingRepository(
        client=qdrant_client,
        settings=settings.qdrant,
    )
    embedding_repository.ensure_storage()

    app.state.redis = redis
    app.state.redis_publisher = RedisPublisher(redis)

    app.state.audio_storage = LocalAudioStorage(
        base_dir=settings.audio_download_dir
    )

    app.state.qdrant = qdrant_client

    yield

    await redis.close()


app = FastAPI(
    lifespan=lifespan
)

app.include_router(
    transcription_router,
    prefix="/api/v1",
    tags=["Transcriptions"],
)

app.include_router(
    speaker_router,
    prefix="/api/v1",
    tags=["Speakers"],
)