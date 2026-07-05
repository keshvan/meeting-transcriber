from pathlib import Path

from app.utils.audio import load_audio
from app.pipeline.meeting_pipeline import MeetingPipeline
from app.pipeline.diarization import DiarizationProcessor
from app.pipeline.embedding import EmbeddingProcessor
from app.pipeline.stt import STTProcessor
from app.pipeline.alignment import AlignmentProcessor
from app.infrastructure.stt.factory import STTFactory

def main():
    audio = load_audio(Path("data/meetings/test.wav"))

    print(f"Sample rate: {audio.sample_rate}")
    print(f"Shape: {audio.waveform.shape}")
    print(f"Duration: {len(audio.waveform) / audio.sample_rate:.2f} sec")

    pipeline = MeetingPipeline(
        diarization=DiarizationProcessor(),
        stt=STTProcessor(STTFactory.create()),
        alignment=AlignmentProcessor(),
        embedding=EmbeddingProcessor(),
    )

    result = pipeline.process(audio)

    print("\n--- Segments ---")
    for segment in result.segments:
        print(
            f"[{segment.start:.2f}-{segment.end:.2f}] "
            f"{segment.speaker_id}: {segment.text}"
        )

    print("\n--- Embeddings ---")
    print(f"Successful embeddings: {len(result.embeddings.embeddings)}")

    for emb in result.embeddings.embeddings:
        duration = emb.metadata.get("total_duration", 0.0)
        print(
            f"{emb.person_id}: "
            f"vector_size={len(emb.vector)}, "
            f"duration={duration:.2f}s"
        )

    print(f"Failed speakers: {len(result.embeddings.failed)}")

    for failure in result.embeddings.failed:
        print(
            f"{failure.speaker_id}: "
            f"reason={failure.reason}, "
            f"duration={failure.total_duration:.2f}s"
        )


if __name__ == "__main__":
    main()