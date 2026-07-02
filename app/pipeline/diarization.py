from pathlib import Path

import torch
from pyannote.audio import Pipeline

from app.config.config import config
from app.domain.speech_segment import SpeechSegment

class DiarizationProcessor:
    def __init__(self):
        self.pipeline = Pipeline.from_pretrained(
            config.diarization_model,
            token=config.hf_token
        )

        self.pipeline.to(config.device)

    def process(self, waveform: torch.Tensor, sample_rate: int) -> list[SpeechSegment]:
        diarization = self.pipeline(
            {
                "waveform": waveform.unsqueeze(0),
                "sample_rate": sample_rate
            },
        )

        segments = []
        for turn, speaker in diarization.speaker_diarization:
            segments.append(
                SpeechSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker_id=speaker
                )
            )
        
        return segments
            