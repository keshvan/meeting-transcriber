import asyncio

from qdrant_client import QdrantClient
from redis.asyncio import Redis

from app.db.session import SessionLocal
from app.pipeline.alignment import AlignmentProcessor
from app.pipeline.diarization import DiarizationProcessor
from app.pipeline.embedding import EmbeddingProcessor
from app.pipeline.stt import STTProcessor
from app.qdrant.qdrant_voice_embedding_repository import QdrantVoiceEmbeddingRepository
from app.services.embedding_protection.embedding_protection import build_embedding_protector
from app.services.speaker_identification.speaker_identification import SpeakerIdentificationService
from app.stt.factory import STTFactory
from app.worker.consumer import RedisMeetingConsumer
from app.worker.meeting_worker import MeetingWorker

from app.pipeline.meeting_pipeline import MeetingPipeline
from app.formatter.formatter import Formatter
from app.services.smtp.smtp import SmtpService
from app.config.settings import settings

from app.storage.local_storage import LocalAudioStorage


client = QdrantClient(
    url=settings.qdrant.url,
    api_key=settings.qdrant.api_key,
    timeout=settings.qdrant.timeout,
    prefer_grpc=settings.qdrant.prefer_grpc,
)

repository = QdrantVoiceEmbeddingRepository(
    client=client,
    settings=settings.qdrant,
)
repository.ensure_storage()

speaker_identification = SpeakerIdentificationService(
    repository=repository,
    protector=build_embedding_protector(settings.embedding_protection, settings.qdrant.vector_size),
    config=settings.speaker_identification,
)


async def create_consumer() -> RedisMeetingConsumer:
    redis = Redis(
        host="localhost",
        port=6379,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=None
    )

    pipeline = MeetingPipeline(
        diarization=DiarizationProcessor(),
        stt=STTProcessor(STTFactory.create()),
        alignment=AlignmentProcessor(),
        embedding=EmbeddingProcessor(),
        speaker_identification=speaker_identification
    )

    worker = MeetingWorker(
        pipeline=pipeline,
        audio_storage=LocalAudioStorage(settings.audio_download_dir),
        formatter=Formatter(),
        smtp_service=SmtpService(
            host=settings.smtp_host,
            port=settings.smtp_port,
            username=settings.smtp_username,
            password=settings.smtp_password,
            sender=settings.smtp_sender,
        ),
        session_factory=SessionLocal
    )

    consumer = RedisMeetingConsumer(
        redis=redis,
        worker=worker,
    )

    return consumer


async def main():
    consumer = await create_consumer()

    await consumer.run()


if __name__ == "__main__":
    asyncio.run(main())