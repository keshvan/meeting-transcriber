import json
from uuid import UUID

from redis import ResponseError
from redis.asyncio import Redis

from app.domain.event import MeetingCreatedEvent
from app.worker.meeting_worker import MeetingWorker


class RedisMeetingConsumer:
    STREAM = "meeting-events"
    GROUP = "meeting-workers"
    CONSUMER = "worker-1"

    def __init__(
        self,
        redis: Redis,
        worker: MeetingWorker,
    ):
        self.redis = redis
        self.worker = worker

    async def run(self):
        try:
            await self.redis.xgroup_create(
                name=self.STREAM,
                groupname=self.GROUP,
                id="$",
                mkstream=True,
            )
        except ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

        while True:
            messages = await self.redis.xreadgroup(
                groupname=self.GROUP,
                consumername=self.CONSUMER,
                streams={self.STREAM: ">"},
                count=1,
                block=5000,
            )

            if not messages:
                continue

            for _, entries in messages:
                for message_id, fields in entries:
                    try:
                        payload = json.loads(fields["payload"])

                        event = MeetingCreatedEvent(
                            meeting_id=UUID(payload["meeting_id"]),
                            audio_key=payload["audio_key"],
                            contact=payload["contact"],
                        )

                        await self.worker.handle(event)

                        await self.redis.xack(
                            self.STREAM,
                            self.GROUP,
                            message_id,
                        )

                    except Exception as e:
                        print(f"Worker error: {e}")
                        continue