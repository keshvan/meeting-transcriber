import numpy as np

import torch
from faster_whisper import WhisperModel

from app.domain.raw_audio import RawAudio
from app.domain.transcript import Word, Transcription

class FasterWhisperSTT:
    def __init__(
        self,
        model_size: str,
        device: str,
        compute_type: str,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(
        self,
        audio: RawAudio,
        language: str | None = None
    ) -> Transcription:
        if audio.waveform.ndim != 1:
            raise ValueError(
                f"RawAudio waveform must be 1D, got shape {tuple(audio.waveform.shape)}"
            )
        
        samples = audio.waveform.cpu().numpy().astype(np.float32)

        segments, info = self.model.transcribe(
            samples,
            language=language,
            word_timestamps=True,
        )

        words: list[Word] = []
        for segment in segments:
            for w in segment.words:
                words.append(
                    Word(
                        text=w.word,
                        start=w.start,
                        end=w.end
                    )
                )

        full_text = " ".join(w.text for w in words)
        
        return Transcription(
            words=words,
            text=full_text
        )