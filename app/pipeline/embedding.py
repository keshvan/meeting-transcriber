import logging
from collections import defaultdict

import torch

from speechbrain.inference.speaker import EncoderClassifier

from app.config.settings import settings
from app.domain.speech_segment import SpeechSegment
from app.domain.raw_audio import RawAudio, cut_audio
from app.domain.voice_embedding import (
    EmbeddingResult,
    FailedSpeakerEmbedding,
    Vector,
    VoiceEmbedding,
)

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000

class EmbeddingProcessor:
    def __init__(
        self,
        model_name: str | None = None,
        device: torch.device | None = None,
        min_duration: float | None = None,
    ):
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.device
        self.min_duration = (
            min_duration if min_duration is not None else settings.embedding_min_duration
        )

        self.classifier = EncoderClassifier.from_hparams(
            source=self.model_name,
            run_opts={"device": str(self.device)},
        )
    
    def process(
        self,
        audio: RawAudio,
        segments: list[SpeechSegment]
    ) -> EmbeddingResult:
        waveform = audio.waveform
        if waveform.device != self.device:
            waveform = waveform.to(self.device)

        if waveform.ndim != 1:
            raise ValueError(
                f"RawAudio waveform must be 1D, got shape {tuple(waveform.shape)}"
            )

        speaker_segments: defaultdict[str, list[SpeechSegment]] = defaultdict(list)
        for segment in segments:
            speaker_segments[segment.speaker_id].append(segment)

        embeddings: list[VoiceEmbedding] = []
        failed: list[FailedSpeakerEmbedding] = []
        
        for speaker_id, speaker_segs in speaker_segments.items():
            speaker_segs = sorted(speaker_segs, key=lambda s: s.start_ms)
            filtered = [
                s for s in speaker_segs if (s.end_ms - s.start_ms) >= self.min_duration
            ]

            if not filtered:
                total = sum(s.end_ms - s.start_ms for s in speaker_segs)
                failure = FailedSpeakerEmbedding(
                    speaker_id=speaker_id,
                    reason="no_segments",
                    total_duration=total,
                    segment_count=len(speaker_segs),
                    segments=[(s.start_ms, s.end_ms) for s in speaker_segs],
                )
                failed.append(failure)
                logger.warning(
                    "Speaker %s has no segments longer than %.2fs; skipping",
                    speaker_id,
                    self.min_duration,
                )
                continue

            try:
                speaker_wave = self._build_speaker_waveform(audio, filtered)
                speaker_wave = self._resample_if_needed(speaker_wave, audio.sample_rate)
                vector = self._embed(speaker_wave)

                embeddings.append(
                    VoiceEmbedding(
                        speaker_id=speaker_id,
                        vector=vector,
                        metadata={
                            "source_segments": len(filtered),
                            "total_duration": sum(
                                s.end_ms - s.start_ms for s in filtered
                            ),
                            "embedding_model": self.model_name,
                        },
                    )
                )
            except Exception as exc:
                total = sum(s.end_ms - s.start_ms for s in filtered)
                failure = FailedSpeakerEmbedding(
                    speaker_id=speaker_id,
                    reason="model_error",
                    total_duration=total,
                    segment_count=len(filtered),
                    segments=[(s.start_ms, s.end_ms) for s in filtered],
                )
                failed.append(failure)
                logger.warning(
                    "Failed to compute embedding for %s: %s",
                    speaker_id,
                    exc,
                    exc_info=True,
                )

        return EmbeddingResult(embeddings=embeddings, failed=failed)
    
    def _build_speaker_waveform(
        self,
        audio: RawAudio,
        segments: list[SpeechSegment],
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []

        for segment in segments:
            part_audio = cut_audio(audio, segment.start_ms, segment.end_ms)
            parts.append(part_audio.waveform)

        if parts:
            return torch.cat(parts)

        return torch.zeros(0, device=audio.waveform.device, dtype=audio.waveform.dtype)

    def _resample_if_needed(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
    ) -> torch.Tensor:
        if sample_rate == target_sample_rate:
            return waveform

        try:
            import torchaudio

            return torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=target_sample_rate
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to resample from {sample_rate} to {target_sample_rate}"
            ) from exc
    
    def _embed(self, waveform: torch.Tensor) -> Vector:
        if waveform.numel() == 0:
            raise ValueError("Cannot embed empty waveform.")
        
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        
        embedding = self.classifier.encode_batch(waveform)
        embedding = embedding.squeeze()

        if embedding.ndim == 0:
            raise ValueError("Unexpected embedding shape.")

        if embedding.ndim > 1:
            embedding = embedding[0]

        return embedding.cpu().tolist()