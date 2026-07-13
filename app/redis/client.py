from redis.asyncio import Redis

from app.config.settings import settings

def create_redis() -> Redis:
    return Redis(
        host="localhost",
        port=6379,
        decode_responses=True
    )