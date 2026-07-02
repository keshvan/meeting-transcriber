from app.application.embedding_protection import NoOpEmbeddingProtector
from app.application.speaker_identification import SpeakerIdentificationService
from app.config.config import SpeakerIdentificationConfig
from app.domain.voice_embedding import SpeakerCandidate


class FakeVoiceEmbeddingRepository:
    def __init__(self):
        self.saved = []

    def ensure_storage(self):
        pass

    def upsert_embedding(self, embedding, embedding_count, updated_at):
        self.saved.append((embedding, embedding_count, updated_at))

    def get_centroid(self, person_id):
        return None

    def upsert_centroid(self, person_id, person_name, vector, embedding_count, updated_at):
        pass

    def search_centroids(self, vector, limit):
        return [
            SpeakerCandidate(person_id="1", person_name="Alice", score=0.81),
            SpeakerCandidate(person_id="2", person_name="Bob", score=0.79),
        ][:limit]

    def search_person_embeddings(self, person_id, vector, limit):
        scores = {
            "1": [SpeakerCandidate("1", "Alice", 0.82)],
            "2": [SpeakerCandidate("2", "Bob", 0.91)],
        }
        return scores[person_id][:limit]


def test_identify_returns_best_person_above_threshold():
    service = SpeakerIdentificationService(
        repository=FakeVoiceEmbeddingRepository(),
        protector=NoOpEmbeddingProtector(),
        config=SpeakerIdentificationConfig(
            centroid_top_k=10,
            sample_limit_per_person=1000,
            similarity_threshold=0.85,
            unknown_speaker_name="Unknown",
        ),
    )

    result = service.identify([0.1, 0.2])

    assert result.is_known is True
    assert result.person_id == "2"
    assert result.person_name == "Bob"
    assert result.score == 0.91


def test_identify_returns_unknown_below_threshold():
    service = SpeakerIdentificationService(
        repository=FakeVoiceEmbeddingRepository(),
        protector=NoOpEmbeddingProtector(),
        config=SpeakerIdentificationConfig(
            centroid_top_k=10,
            sample_limit_per_person=1000,
            similarity_threshold=0.95,
            unknown_speaker_name="Unknown",
        ),
    )

    result = service.identify([0.1, 0.2])

    assert result.is_known is False
    assert result.person_id is None
    assert result.person_name == "Unknown"
    assert result.score == 0.91
