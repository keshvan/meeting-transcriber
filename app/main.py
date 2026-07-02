from pathlib import Path

import torch

from app.utils.audio import load_audio
from app.pipeline.diarization import DiarizationProcessor

def main():
    audio_path = Path("data/meetings/test.wav")

    audio, sr = load_audio(audio_path)

    print(f"Sample rate: {sr}")
    print(f"Shape: {audio.shape}")
    print(f"Duration: {len(audio) / sr:.2f} sec")

    diarization = DiarizationProcessor()
    audio = torch.from_numpy(audio)
    segments = diarization.process(audio, sr)

    print(f"found segments: {len(segments)}\n")

    for segment in segments:
        print(segment)

if __name__ == "__main__":
    main()