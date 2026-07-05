from app.domain.raw_audio import RawAudio
from app.domain.pipeline_result import PipelineResult

from app.pipeline.diarization import DiarizationProcessor
from app.pipeline.stt import STTProcessor
from app.pipeline.embedding import EmbeddingProcessor
from app.pipeline.alignment import AlignmentProcessor

class MeetingPipeline:
    def __init__(
        self,
        diarization: DiarizationProcessor,
        stt: STTProcessor,
        alignment: AlignmentProcessor,
        embedding: EmbeddingProcessor
    ):
        self.diarization = diarization
        self.stt = stt
        self.alignment = alignment
        self.embedding = embedding
    
    def process(self, audio: RawAudio) -> PipelineResult:
        segments = self.diarization.process(audio)

        transcript = self.stt.process(audio)

        segments = self.alignment.process(
            segments=segments,
            transcript=transcript
        )

        embeddings = self.embedding.process(
            audio=audio,
            segments=segments
        )

        return PipelineResult(
            segments=segments,
            embeddings=embeddings
        )