from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.config import settings
from app.infrastructure.db.init_db import init_db
from app.api.routes import router
from app.infrastructure.qdrant_voice_embedding_repository import QdrantVoiceEmbeddingRepository
from app.infrastructure.service_factory import build_qdrant_client

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    client = build_qdrant_client()
    repository = QdrantVoiceEmbeddingRepository(
        client=client,
        settings=settings.qdrant,
    )
    repository.ensure_storage()

    yield 

app = FastAPI(lifespan=lifespan)
app.include_router(router)