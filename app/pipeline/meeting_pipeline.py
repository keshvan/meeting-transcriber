from collections import defaultdict

from typing import TYPE_CHECKING

from app.application.speaker_identification import SpeakerIdentificationService
from app.domain.raw_audio import RawAudio
from app.domain.pipeline_result import PipelineResult
from app.domain.voice_embedding import VoiceEmbedding

if TYPE_CHECKING:
    from app.pipeline.diarization import DiarizationProcessor
    from app.pipeline.stt import STTProcessor
    from app.pipeline.embedding import EmbeddingProcessor
    from app.pipeline.alignment import AlignmentProcessor


class MeetingPipeline:
    def __init__(
        self,
        diarization: "DiarizationProcessor",
        stt: "STTProcessor",
        alignment: "AlignmentProcessor",
        embedding: "EmbeddingProcessor",
        speaker_identification: SpeakerIdentificationService,
    ):
        self.diarization = diarization
        self.stt = stt
        self.alignment = alignment
        self.embedding = embedding
        self.speaker_identification = speaker_identification

    def process(self, audio: RawAudio) -> PipelineResult:
        segments = self.diarization.process(audio)

        transcript = self.stt.process(audio)

        segments = self.alignment.process(
            segments=segments,
            transcript=transcript,
        )

        embedding_result = self.embedding.process(
            audio=audio,
            segments=segments,
        )

        print("==================EMBEDDINGS==================")
        print(f"Embedding result: {embedding_result.embeddings}")
        print("==================EMBEDDINGS==================")

        self._apply_speaker_names(segments=segments, embeddings=embedding_result.embeddings)

        return PipelineResult(
            segments=segments,
            embeddings=embedding_result,
        )

    def _apply_speaker_names(
        self,
        *,
        segments: list,
        embeddings: list[VoiceEmbedding],
    ) -> None:
        if self.speaker_identification is None:
            return

        speaker_segments: defaultdict[str, list] = defaultdict(list)
        for segment in segments:
            speaker_segments[segment.speaker_id].append(segment)

        for embedding in embeddings:
            result = self.speaker_identification.identify(embedding.vector)
            person_name = result.person_name
            for segment in speaker_segments.get(embedding.person_id, []):
                segment.person_name = person_name

            if result.is_known:
                self.speaker_identification.save_confirmed_embedding(
                    VoiceEmbedding(
                        person_id=embedding.person_id,
                        person_name=result.person_name,
                        vector=embedding.vector,
                        embedding_id=embedding.embedding_id,
                        updated_at=embedding.updated_at,
                        metadata=embedding.metadata,
                    )
                )