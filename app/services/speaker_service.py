from uuid import UUID
import uuid

from app.domain.entities import Person
from app.domain.voice_embedding import VoiceEmbedding


class SpeakerService:
    def __init__(
        self,
        speaker_repository,
        person_repository,
        embedding_repository,
        identification_service,
    ):
        self.speaker_repository = speaker_repository
        self.person_repository = person_repository
        self.embedding_repository = embedding_repository
        self.identification_service = identification_service

    def resolve_unknown_speaker(
        self,
        meeting_id: UUID,
        speaker_id: str,
        full_name: str,
    ) -> None:

        person = self.person_repository.get_by_name(full_name)

        if person is None:
            person = self.person_repository.create(
                Person(
                    id=uuid.uuid4(),
                    name=full_name,
                )
            )

        speaker = self.speaker_repository.get_by_meeting_and_id(meeting_id, speaker_id)

        embedding = self.embedding_repository.get_embedding(speaker.embedding_id)
        
        confirmed_embedding = VoiceEmbedding(
            speaker_id=embedding.speaker_id,
            person_id=str(person.id),
            person_name=person.name,
            vector=embedding.vector,
            embedding_id=embedding.embedding_id,
            updated_at=embedding.updated_at,
            metadata=embedding.metadata,
        )

        self.identification_service.save_confirmed_embedding(
            confirmed_embedding,
            False
        )

        self.speaker_repository.assign_person(meeting_id, speaker_id, person.id)