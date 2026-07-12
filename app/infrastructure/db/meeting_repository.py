from typing import Optional, Sequence
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.domain.entities import Meeting, MeetingStatus
from app.infrastructure.db.models import MeetingORM

def _meeting_to_domain(row: MeetingORM) -> Meeting:
    return Meeting(
        id=row.id,
        created_at=row.created_at,
        status=MeetingStatus(row.status),
        duration=row.duration,
    )

class PostgresMeetingRepository:
    def __init__(self, session: Session):
        self._session = session
    
    def create(self, meeting: Meeting) -> Meeting:
        row = MeetingORM(
            id=meeting.id,
            status=meeting.status.value,
            duration=meeting.duration,
        )
        self._session.add(row)
        self._session.flush()
        return _meeting_to_domain(row)
    
    def get(self, meeting_id: UUID) -> Optional[Meeting]:
        row = self._session.get(MeetingORM, meeting_id)
        return _meeting_to_domain(row) if row else None
    
    def update_status(self, meeting_id: UUID, status: str) -> None:
        self._session.execute(
            update(MeetingORM).where(MeetingORM.id == meeting_id).values(status=status)
        )
    
    def list_by_status(self, status: str) -> Sequence[Meeting]:
        rows = self._session.scalars(
            select(MeetingORM).where(MeetingORM.status == status)
        ).all()
        return [_meeting_to_domain(r) for r in rows]

    def commit(self):
        self._session.commit()