from datetime import datetime, timezone
from typing import Optional, Sequence
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.domain.entities import Person
from app.infrastructure.db.models import PersonORM

def _person_to_domain(row: PersonORM) -> Person:
    return Person(id=row.id, name=row.name, created_at=row.created_at)

class PostgresPersonRepository:
    def __init__(self, session: Session):
        self._session = session

    def create(self, person: Person) -> Person:
        row = PersonORM(id=person.id, 
                        name=person.name, 
                        created_at=person.created_at or datetime.now(timezone.utc))
        self._session.add(row)
        self._session.flush()
        return _person_to_domain(row)

    def get(self, person_id: UUID) -> Optional[Person]:
        row = self._session.get(PersonORM, person_id)
        return _person_to_domain(row) if row else None

    def get_by_name(self, name: str) -> Optional[Person]:
        row = self._session.scalar(
            select(PersonORM).where(PersonORM.name == name)
        )
        return _person_to_domain(row) if row else None
    
    def list_all(self) -> Sequence[Person]:
        rows = self._session.scalars(select(PersonORM)).all()
        return [_person_to_domain(r) for r in rows]