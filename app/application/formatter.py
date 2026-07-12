from app.domain.speech_segment import SpeechSegment
from html import escape


class Formatter:
    def merge_segments(self, segments: list[SpeechSegment], max_gap_ms: int = 2000) -> list[SpeechSegment]:
        if not segments:
            return []

        merged = [segments[0]]

        for current in segments[1:]:
            last = merged[-1]

            same_speaker = (
                last.speaker_id == current.speaker_id
            )

            small_gap = (
                current.start_ms - last.end_ms <= max_gap_ms
            )

            if same_speaker and small_gap:
                last.end_ms = current.end_ms
                last.text = f"{last.text} {current.text}".strip()
            else:
                merged.append(current)

        return merged

    @staticmethod
    def _format_time(ms: int) -> str:
        total_seconds = ms // 1000
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        if hours:
            return f"{hours:02}:{minutes:02}:{seconds:02}"

        return f"{minutes:02}:{seconds:02}"
    
    def to_txt(self, segments: list[SpeechSegment]) -> str:
        lines = []

        for segment in segments:
            speaker = (
                segment.person_name
                or segment.diarization_label
                or "Unknown"
            )

            timestamp = self._format_time(segment.start_ms)

            lines.append(
                f"[{timestamp}] {speaker}:\n"
                f"{segment.text or ''}\n"
            )

        return "\n".join(lines)

    def to_html_preview(
        self,
        segments: list[SpeechSegment],
        max_segments: int = 30,
    ) -> str:
        items = []

        for segment in segments[:max_segments]:
            speaker = (
                segment.person_name
                or segment.diarization_label
                or "Unknown"
            )

            text = escape(segment.text or "")

            if len(text) > 200:
                text = text[:200] + "..."

            items.append(
                f"""
                <p>
                    <b>{escape(speaker)}</b><br>
                    {text}
                </p>
                """
            )

        more = ""
        if len(segments) > max_segments:
            more = (
                f"<p><i>Показаны первые {max_segments} реплик. "
                "Полная расшифровка находится во вложении.</i></p>"
            )

        return f"""
        <html>
        <body>
            <h2>Расшифровка встречи готова</h2>

            <p>
                Во вложении находится полный текст встречи в формате TXT.
            </p>

            <hr>

            {''.join(items)}

            {more}
        </body>
        </html>
        """