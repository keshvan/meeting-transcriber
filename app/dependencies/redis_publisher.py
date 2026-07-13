from fastapi import Request

from app.redis.publisher import RedisPublisher

def get_redis_publisher(request: Request) -> RedisPublisher:
    return request.app.state.redis_publisher