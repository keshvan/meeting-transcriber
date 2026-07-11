from collections import defaultdict

from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID
import uuid

from app.application.speaker_identification import SpeakerIdentificationService
from app.domain.entities import Speaker, SpeakerStatus
from app.domain.raw_audio import RawAudio
from app.domain.pipeline_result import PipelineResult
from app.domain.speech_segment import SpeechSegment
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

    def process(self, meeting_id: UUID, audio: RawAudio) -> PipelineResult:
        segments = self.diarization.process(audio)
        for segment in segments:
            segment.meeting_id = meeting_id

        speakers = self._build_speakers(meeting_id, segments)

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

        self._apply_speaker_names(speakers=speakers, segments=segments, embeddings=embedding_result.embeddings)

        return PipelineResult(
            speakers=speakers,
            segments=segments,
            embeddings=embedding_result,
        )

    def _build_speakers(
        self,
        meeting_id: UUID,
        segments: list[SpeechSegment]
    ) -> list[Speaker]:
        mapping: dict[str, Speaker] = {}

        for segment in segments:
            speaker = mapping.get(segment.diarization_label)

            if speaker is None:
                speaker = Speaker(
                    id=uuid.uuid4(),
                    meeting_id=meeting_id,
                    diarization_label=segment.diarization_label,
                    person_id=None,
                    status=SpeakerStatus.UNKNOWN,
                    embedding_id=None,
                )
                mapping[segment.diarization_label] = speaker

            segment.speaker_id = speaker.id

        return list(mapping.values())

    def _apply_speaker_names(
        self,
        *,
        speakers: list[Speaker],
        segments: list[SpeechSegment],
        embeddings: list[VoiceEmbedding],
    ) -> None:
        if self.speaker_identification is None:
            return
        
        speaker_map: dict[UUID, Speaker] = {speaker.id: speaker for speaker in speakers}
        speaker_segments: defaultdict[UUID, list[SpeechSegment]] = defaultdict(list)
        for segment in segments:
            speaker_segments[segment.speaker_id].append(segment)

        for embedding in embeddings:
            result = self.speaker_identification.identify(embedding.vector)
            speaker = speaker_map.get(embedding.speaker_id)
            if speaker is None:
                continue

            for segment in speaker_segments.get(embedding.speaker_id, []):
                segment.person_name = result.person_name

            if result.is_known:
                speaker.person_id = result.person_id
                speaker.embedding_id = embedding.embedding_id
                speaker.status = SpeakerStatus.IDENTIFIED

                # Сохраняем как подтверждённый (обновит центроид и перезапишет эмбеддинг)
                self.speaker_identification.save_confirmed_embedding(
                    VoiceEmbedding(
                        speaker_id=embedding.speaker_id,
                        person_id=result.person_id,
                        person_name=result.person_name,
                        vector=embedding.vector,
                        embedding_id=embedding.embedding_id,
                        updated_at=embedding.updated_at,
                        metadata=embedding.metadata,
                    )
                )
            else:
                speaker.status = SpeakerStatus.UNKNOWN
                # Сохраняем эмбеддинг без person_id (как неизвестный)
                self.speaker_identification.repository.upsert_embedding(
                    embedding=embedding,
                    embedding_count=1, # Не важно, центроида не обновляется
                    updated_at=datetime.now(timezone.utc),
                )