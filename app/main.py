from pathlib import Path

import torch

from app.utils.audio import load_audio
from app.pipeline.diarization import DiarizationProcessor
from app.pipeline.embedding import EmbeddingProcessor
from app.pipeline.stt import STTProcessor
from app.infrastructure.stt.factory import STTFactory

def main():
    audio_path = Path("data/meetings/test.wav")

    audio = load_audio(audio_path)

    print(f"Sample rate: {audio.sample_rate}")
    print(f"Shape: {audio.waveform.shape}")
    print(f"Duration: {len(audio.waveform) / audio.sample_rate:.2f} sec")

    diarization = DiarizationProcessor()
    segments = diarization.process(audio)

    print(f"found segments: {len(segments)}\n")
    for segment in segments:
        print(segment)

    print("\n--- STT stage ---")
    stt_client = STTFactory.create()
    stt_processor = STTProcessor(client=stt_client)
    segments = stt_processor.process(audio, segments)

    for segment in segments:
        print(f"{segment.speaker_id}: {segment.text}")

    print("\n--- Embedding stage ---")
    embedding_processor = EmbeddingProcessor()
    result = embedding_processor.process(audio, segments)

    print(f"Successful embeddings: {len(result.embeddings)}")
    for emb in result.embeddings:
        duration = emb.metadata.get("total_duration", 0.0)
        print(
            f"  {emb.person_id}: vector_size={len(emb.vector)}, "
            f"duration={duration:.2f}s"
        )

    print(f"Failed speakers: {len(result.failed)}")
    for failure in result.failed:
        print(
            f"  {failure.speaker_id}: reason={failure.reason}, "
            f"duration={failure.total_duration:.2f}s, "
            f"segments={failure.segment_count}"
        )


if __name__ == "__main__":
    main()