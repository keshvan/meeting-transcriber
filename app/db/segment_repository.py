from typing import Optional, Sequence
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.domain.entities import Segment
from app.domain.speech_segment import SpeechSegment
from app.db.models import SegmentORM

def _segment_to_domain(row: SegmentORM) -> Segment:
    return Segment(
        id=row.id,
        meeting_id=row.meeting_id,
        speaker_id=row.speaker_id,
        start_ms=row.start_ms,
        end_ms=row.end_ms,
        text=row.text
    )

class PostgresSegmentRepository:
    def __init__(self, session: Session):
        self._session = session

    def bulk_create(self, segments: list[SpeechSegment]) -> None:
        rows = [
            SegmentORM(
                meeting_id=s.meeting_id,
                speaker_id=s.speaker_id,
                start_ms=s.start_ms,
                end_ms=s.end_ms,
                text=s.text
            ) 
            for s in segments
        ]
        self._session.add_all(rows)
        self._session.flush()

    def list_by_meeting(self, meeting_id: UUID) -> Sequence[Segment]:
        rows = self._session.scalars(
            select(SegmentORM)
            .where(SegmentORM.meeting_id == meeting_id)
            .order_by(SegmentORM.start_ms)
        ).all()
        return [_segment_to_domain(r) for r in rows]

    def list_by_speaker(self, speaker_id: UUID) -> Sequence[Segment]:
        rows = self._session.scalars(
            select(SegmentORM)
            .where(SegmentORM.speaker_id == speaker_id)
            .order_by(SegmentORM.start_ms)
        ).all()
        return [_segment_to_domain(r) for r in rows]
    
    def get_by_speaker(self, speaker_id: UUID) -> Sequence[Segment]:
        rows = self._session.scalars(
            select(SegmentORM).where(SegmentORM.speaker_id == speaker_id)
        ).all()
        return [_segment_to_domain(r) for r in rows]
    
    def commit(self) -> None:
        self._session.commit()
