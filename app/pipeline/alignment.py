from dataclasses import replace

from app.domain.speech_segment import SpeechSegment
from app.domain.transcript import Transcription, Word


class AlignmentProcessor:
    def process(
        self,
        segments: list[SpeechSegment],
        transcript: Transcription,
    ) -> list[SpeechSegment]:
        words = transcript.words
        word_idx = 0

        result: list[SpeechSegment] = []

        for segment in segments:
            while (
                word_idx < len(words)
                and words[word_idx].end_ms <= segment.start_ms
            ):
                word_idx += 1

            current = word_idx
            segment_words: list[Word] = []

            while (
                current < len(words)
                and words[current].start_ms < segment.end_ms
            ):
                word = words[current]

                if word.end_ms > segment.start_ms:
                    segment_words.append(word)

                current += 1

            segment.text = " ".join(word.text for word in segment_words).strip()

            result.append(segment)

        return result