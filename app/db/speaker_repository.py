from typing import Optional, Sequence
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.domain.entities import Speaker, SpeakerStatus
from app.db.models import SpeakerORM

def _speaker_to_domain(row: SpeakerORM) -> Speaker:
    return Speaker(
        meeting_id=row.meeting_id,
        speaker_id=row.speaker_id,
        person_id=row.person_id,
        status=SpeakerStatus(row.status),
        embedding_id=row.embedding_id,
    )

class PostgresSpeakerRepository:
    def __init__(self, session: Session):
        self._session = session
    
    def bulk_create(self, speakers: list[Speaker]) -> None:
        rows = [
            SpeakerORM(
                meeting_id=s.meeting_id,
                speaker_id=s.speaker_id,
                person_id=s.person_id,
                status=s.status.value,
                embedding_id=s.embedding_id
            )
            for s in speakers
        ]
        self._session.add_all(rows)
        self._session.flush()
    
    def get(self, speaker_id: UUID) -> Optional[Speaker]:
        row = self._session.get(SpeakerORM, speaker_id)
        return _speaker_to_domain(row) if row else None

    def list_by_meeting(self, meeting_id: UUID) -> Sequence[Speaker]:
        rows = self._session.scalars(
            select(SpeakerORM).where(SpeakerORM.meeting_id == meeting_id)
        ).all()
        return [_speaker_to_domain(r) for r in rows]

    def list_unknown_by_meeting(self, meeting_id: UUID) -> Sequence[Speaker]:
        rows = self._session.scalars(
            select(SpeakerORM).where(
                SpeakerORM.meeting_id == meeting_id,
                SpeakerORM.person_id.is_(None),
            )
        ).all()
        return [_speaker_to_domain(r) for r in rows]

    def assign_person(self, speaker_id: UUID, person_id: UUID) -> None:
        self._session.execute(
            update(SpeakerORM)
            .where(SpeakerORM.id == speaker_id)
            .values(person_id=person_id, status=SpeakerStatus.IDENTIFIED.value)
        )
    
    def list_by_ids(self, speaker_ids: list[UUID]) -> Sequence[Speaker]:
        if not speaker_ids:
            return []
        rows = self._session.scalars(
            select(SpeakerORM).where(SpeakerORM.id.in_(speaker_ids))
        ).all()
        return [_speaker_to_domain(r) for r in rows]

    def assign_person_bulk(self, speaker_ids: list[UUID], person_id: UUID) -> None:
        if not speaker_ids:
            return
        self._session.execute(
            update(SpeakerORM)
            .where(SpeakerORM.id.in_(speaker_ids))
            .values(person_id=person_id, status=SpeakerStatus.IDENTIFIED.value)
        )
    
    def get_by_meeting_and_id(
        self, meeting_id: UUID, speaker_id: str
    ) -> Optional[Speaker]:
        row = self._session.scalar(
            select(SpeakerORM).where(
                SpeakerORM.meeting_id == meeting_id,
                SpeakerORM.speaker_id == speaker_id,
            )
        )
        return _speaker_to_domain(row) if row else None

    def commit(self) -> None:
        self._session.commit()
