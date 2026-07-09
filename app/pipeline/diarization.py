import torch
from pyannote.audio import Pipeline

from app.config.settings import settings
from app.domain.speech_segment import SpeechSegment
from app.domain.raw_audio import RawAudio

class DiarizationProcessor:
    def __init__(
        self,
        model_name: str | None = None,
        device: torch.device | None = None,
        hf_token: str | None = None,
    ):
        self.model_name = model_name or settings.diarization_model
        self.device = device or settings.device
        self.hf_token = hf_token or settings.hf_token

        self.pipeline = Pipeline.from_pretrained(
            self.model_name,
            token=self.hf_token,
        )

        self.pipeline.to(self.device)

    def process(self, audio: RawAudio) -> list[SpeechSegment]:
        waveform = audio.waveform.to(self.device)

        diarization = self.pipeline(
            {
                "waveform": waveform.unsqueeze(0),
                "sample_rate": audio.sample_rate,
            },
        )

        segments = []
        for turn, speaker in diarization.speaker_diarization:
            segments.append(
                SpeechSegment(
                    start_ms=round(turn.start * 1000),
                    end_ms=round(turn.end * 1000),
                    diarization_label=speaker,
                )
            )

        return segments
