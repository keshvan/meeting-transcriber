# app/cli/resolve.py
import asyncio
import uuid
from typing import Optional

import click
from sqlalchemy.orm import Session

from app.config.settings import settings
from app.infrastructure.db.session import get_session
from app.infrastructure.db.speaker_repository import PostgresSpeakerRepository
from app.infrastructure.db.segment_repository import PostgresSegmentRepository
from app.infrastructure.db.person_repository import PostgresPersonRepository
from app.infrastructure.db.meeting_repository import PostgresMeetingRepository
from app.infrastructure.qdrant_voice_embedding_repository import QdrantVoiceEmbeddingRepository
from app.infrastructure.service_factory import build_qdrant_client, build_embedding_protector
from app.application.speaker_identification import SpeakerIdentificationService
from app.domain.entities import Speaker, SpeakerStatus, Person, MeetingStatus
from app.domain.voice_embedding import VoiceEmbedding


def get_identification_service():
    client = build_qdrant_client()
    repository = QdrantVoiceEmbeddingRepository(client=client, settings=settings.qdrant)
    repository.ensure_storage()
    protector = build_embedding_protector(
        settings=settings.embedding_protection,
        vector_size=settings.qdrant.vector_size,
    )
    return SpeakerIdentificationService(
        repository=repository,
        protector=protector,
        config=settings.speaker_identification,
    )


@click.command()
@click.argument('meeting_id', type=str)
def main(meeting_id: str):
    try:
        meeting_uuid = uuid.UUID(meeting_id)
    except ValueError:
        click.echo("Неверный формат UUID", err=True)
        return

    # Инициализация сервисов и репозиториев
    identification = get_identification_service()
    qdrant_repo = identification.repository

    with get_session() as session:
        speaker_repo = PostgresSpeakerRepository(session)
        segment_repo = PostgresSegmentRepository(session)
        person_repo = PostgresPersonRepository(session)
        meeting_repo = PostgresMeetingRepository(session)

        # Загружаем всех спикеров встречи
        all_speakers = speaker_repo.list_by_meeting(meeting_uuid)
        if not all_speakers:
            click.echo(f"Встреча {meeting_uuid} не найдена или нет спикеров", err=True)
            return

        # Фильтруем неизвестных
        unknown_speakers = [s for s in all_speakers if s.status == SpeakerStatus.UNKNOWN]
        if not unknown_speakers:
            click.echo("Все спикеры уже идентифицированы.")
            meeting_repo.update_status(meeting_uuid, MeetingStatus.COMPLETED)
            session.commit()
            click.echo("Статус встречи обновлён на COMPLETED.")
            return


        # Для каждого неизвестного получаем эмбеддинг из Qdrant
        speaker_embedding_map = {}
        for speaker in unknown_speakers:
            embeddings = qdrant_repo.get_embeddings_by_speaker(speaker.id)
            if embeddings:
                speaker_embedding_map[speaker.id] = embeddings[0]  # берём первый (обычно один)
            else:
                click.echo(f"Предупреждение: для спикера {speaker.diarization_label} не найден эмбеддинг. Пропускаем.")

        if not speaker_embedding_map:
            click.echo("Нет эмбеддингов для неизвестных спикеров. Возможно, пайплайн не сохранил их.")
            return

        # Интерактивный опрос
        for speaker in unknown_speakers:
            embedding = speaker_embedding_map.get(speaker.id)
            if embedding is None:
                continue

            # Получаем примеры реплик этого спикера
            segments = segment_repo.get_by_speaker(speaker.id)
            sample_segments = segments[:10]  # первые три
            sample_text = " | ".join(seg.text for seg in sample_segments)

            click.echo("\n" + "=" * 60)
            click.echo(f"Спикер {speaker.diarization_label} (примеры: '{sample_text}')")

            # Получаем кандидатов через suggest
            candidates = identification.suggest(embedding.vector, limit=5)

            if candidates:
                click.echo("Ближайшие совпадения:")
                for i, cand in enumerate(candidates, start=1):
                    click.echo(f"{i}. {cand.person_name} ({cand.score:.3f})")
                click.echo("0. Создать нового человека")
                click.echo("s. Пропустить")
            else:
                click.echo("Совпадений не найдено.")
                click.echo("0. Создать нового человека")
                click.echo("s. Пропустить")

            choice = input("> ").strip()

            if choice.lower() == 's':
                continue

            if choice == '0':
                # Создать нового человека
                name = input("Введите имя: ").strip()
                if not name:
                    click.echo("Имя не может быть пустым. Пропускаем.")
                    continue
                # Создаём Person
                person = person_repo.create(Person(id=uuid.uuid4(), name=name))
                # Привязываем спикера
                speaker_repo.assign_person(speaker.id, person.id)
                # Сохраняем эмбеддинг как подтверждённый
                confirmed_emb = VoiceEmbedding(
                    speaker_id=speaker.id,
                    person_id=str(person.id),
                    person_name=person.name,
                    vector=embedding.vector,
                    embedding_id=embedding.embedding_id,
                    updated_at=embedding.updated_at,
                    metadata=embedding.metadata,
                )
                identification.save_confirmed_embedding(confirmed_emb)
                click.echo(f"Спикер {speaker.diarization_label} назначен как {person.name}")
                continue

            # Выбор существующего кандидата
            try:
                idx = int(choice) - 1
                if idx < 0 or idx >= len(candidates):
                    raise IndexError
                candidate = candidates[idx]
            except (ValueError, IndexError):
                click.echo("Неверный выбор.")
                continue

            # Привязываем существующего человека
            person_id = uuid.UUID(candidate.person_id)
            # Проверим, есть ли такой Person в БД? Можно создать, если нет, но обычно он есть.
            # Просто назначаем
            speaker_repo.assign_person(speaker.id, person_id)
            confirmed_emb = VoiceEmbedding(
                speaker_id=speaker.id,
                person_id=candidate.person_id,
                person_name=candidate.person_name,
                vector=embedding.vector,
                embedding_id=embedding.embedding_id,
                updated_at=embedding.updated_at,
                metadata=embedding.metadata,
            )
            identification.save_confirmed_embedding(confirmed_emb)
            click.echo(f"Спикер {speaker.diarization_label} назначен как {candidate.person_name}")

        # После обработки всех неизвестных, коммитим изменения в БД (если нужно)
        
        meeting_repo.update_status(meeting_uuid, MeetingStatus.COMPLETED)
        session.commit()
        click.echo("\nВсе изменения сохранены.")

if __name__ == "__main__":
    main()