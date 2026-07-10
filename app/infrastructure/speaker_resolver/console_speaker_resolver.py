from uuid import UUID
import uuid

from app.application.ports.repositories import PersonRepository, SpeakerRepository
from app.application.speaker_identification import SpeakerIdentificationService
from app.domain.entities import Person, Speaker, SpeakerStatus
from app.domain.voice_embedding import VoiceEmbedding

class ConsoleSpeakerResolver:
    def __init__(
        self,
        identification: SpeakerIdentificationService,
        person_repository: PersonRepository,
        speaker_repository: SpeakerRepository
    ):
        self.identiication = identification
        self.person_repository = person_repository
        self.speaker_repositoy = speaker_repository
    
    def resolve(
        self,
        *,
        speakers: list[Speaker],
        embeddings: list[VoiceEmbedding]
    ) -> None:
        
        embedding_map = {
            e.speaker_id: e
            for e in embeddings
        }

        for speaker in speakers:
            if speaker.status != SpeakerStatus.UNKNOWN:
                continue

            embedding = embedding_map.get(speaker.id)
            if embedding is None:
                continue
            suggestions = self.identification.suggest(
                embedding.vector,
                limit=5,
            )

            print("\n" + "=" * 60)
            print(f"Unknown speaker: {speaker.diarization_label}")
            print()

            if suggestions:
                print("Closest matches:")
                for i, candidate in enumerate(suggestions, start=1):
                    print(
                        f"{i}. {candidate.person_name} "
                        f"({candidate.score:.3f})"
                    )
            else:
                print("No similar speakers found.")

            print("\n0. Create new person")
            print("s. Skip")

            choice = input("> ").strip()

            if choice.lower() == "s":
                continue

            if choice == "0":
                self._create_person(speaker, embedding)
                continue

            try:
                candidate = suggestions[int(choice) - 1]
            except (ValueError, IndexError):
                print("Invalid choice.")
                continue

            self._assign_person(
                speaker=speaker,
                embedding=embedding,
                person_id=candidate.person_id,
                person_name=candidate.person_name,
            )

        self.speaker_repository.commit()
    
    def _assign_person(
        self,
        *,
        speaker: Speaker,
        embedding: VoiceEmbedding,
        person_id: str,
        person_name: str,
    ) -> None:
        speaker.person_id = UUID(person_id)
        speaker.status = SpeakerStatus.IDENTIFIED
        speaker.embedding_id = embedding.embedding_id

        self.speaker_repository.update(speaker)

        self.speaker_identification.save_confirmed_embedding(
            VoiceEmbedding(
                speaker_id=speaker.id,
                person_id=person_id,
                person_name=person_name,
                vector=embedding.vector,
                embedding_id=embedding.embedding_id,
                updated_at=embedding.updated_at,
                metadata=embedding.metadata,
            )
        )

    def _create_person(
        self,
        *,
        speaker: Speaker,
        embedding: VoiceEmbedding,
    ) -> None:
        name = input("Person name: ").strip()

        person = self.person_repository.create(
            Person(
                id=uuid.uuid4(),
                name=name,
            )
        )

        self._assign_person(
            speaker=speaker,
            embedding=embedding,
            person_id=str(person.id),
            person_name=person.name,
        )