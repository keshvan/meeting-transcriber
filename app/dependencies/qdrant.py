from fastapi import Request
from qdrant_client import QdrantClient
from app.config.settings import settings

def build_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant.url,
        api_key=settings.qdrant.api_key,
        timeout=settings.qdrant.timeout,
        prefer_grpc=settings.qdrant.prefer_grpc,
    )


def get_qdrant_client(
    request: Request,
) -> QdrantClient:
    return request.app.state.qdrant