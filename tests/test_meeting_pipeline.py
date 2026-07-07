from app.domain.raw_audio import RawAudio
from app.domain.speech_segment import SpeechSegment
from app.domain.voice_embedding import EmbeddingResult, VoiceEmbedding
from app.pipeline.meeting_pipeline import MeetingPipeline


class FakeDiarizationProcessor:
    def process(self, audio):
        return [
            SpeechSegment(start=0.0, end=1.0, speaker_id="SPEAKER_00"),
            SpeechSegment(start=1.0, end=2.0, speaker_id="SPEAKER_00"),
        ]


class FakeSTTProcessor:
    def process(self, audio):
        return None


class FakeAlignmentProcessor:
    def process(self, segments, transcript):
        return segments


class FakeEmbeddingProcessor:
    def process(self, audio, segments):
        return EmbeddingResult(
            embeddings=[
                VoiceEmbedding(
                    person_id="SPEAKER_00",
                    person_name="Unknown",
                    vector=[0.1, 0.2],
                )
            ],
            failed=[],
        )


class FakeSpeakerIdentificationService:
    def __init__(self):
        self.saved = []

    def identify(self, vector):
        return type(
            "Result",
            (),
            {"person_id": "person-1", "person_name": "Alice", "score": 0.95, "is_known": True},
        )()

    def save_confirmed_embedding(self, embedding):
        self.saved.append(embedding)


def test_meeting_pipeline_assigns_person_names_from_identification_service():
    pipeline = MeetingPipeline(
        diarization=FakeDiarizationProcessor(),
        stt=FakeSTTProcessor(),
        alignment=FakeAlignmentProcessor(),
        embedding=FakeEmbeddingProcessor(),
        speaker_identification=FakeSpeakerIdentificationService(),
    )

    result = pipeline.process(RawAudio(waveform=None, sample_rate=16000))

    assert [segment.person_name for segment in result.segments] == ["Alice", "Alice"]
    assert len(pipeline.speaker_identification.saved) == 1
