from app.infrastructure.stt.base import STTClient
from app.domain.raw_audio import RawAudio
from app.domain.speech_segment import SpeechSegment
from app.domain.transcript import Word

class STTProcessor:
    def __init__(
        self,
        client: STTClient
    ):
        self.client = client
    
    def process(
        self,
        audio: RawAudio,
        segments: list[SpeechSegment]
    ) -> list[SpeechSegment]:
        transcription = self.client.transcribe(audio)

        result_segments = []
        for diar_seg in segments:
            segment_words = self._collect_words_for_segment(diar_seg, transcription.words)
            text = " ".join(w.text for w in segment_words).strip()
            diar_seg.text = text
            result_segments.append(diar_seg)

        return result_segments
    
    def _collect_words_for_segment(
        self,
        segment: SpeechSegment,
        words: list[Word]
    ) -> list[Word]:
        result = []
        for word in words:
            overlap = (
                min(segment.end, word.end)
                - max(segment.start, word.start)
            )

            if overlap > 0:
                result.append(word)
        
        return result
