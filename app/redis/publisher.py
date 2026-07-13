import json

from redis.asyncio import Redis

from app.domain.event import MeetingCreatedEvent

class RedisPublisher:
    def __init__(self, redis: Redis):
        self._redis = redis

    async def publish(self, event: MeetingCreatedEvent):
        print("send", event)
        await self._redis.xadd(
            "meeting-events",
            {
                "type": "meeting.created",
                "payload": json.dumps({
                    "meeting_id": str(event.meeting_id),
                    "audio_key": event.audio_key,
                    "contact": event.contact
                }),
            },
        )