from typing import Generator

from fastapi import Depends
from sqlalchemy.orm import Session

from app.db.person_repository import PostgresPersonRepository
from app.db.session import SessionLocal, get_session
from app.db.meeting_repository import PostgresMeetingRepository
from app.db.speaker_repository import PostgresSpeakerRepository

def get_db_session() -> Generator[Session, None, None]:
    session = SessionLocal()

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
        
def get_meeting_repository(
    session: Session = Depends(get_db_session)
):
    return PostgresMeetingRepository(session)

def get_speaker_repository(
    session: Session = Depends(get_db_session),
):
    return PostgresSpeakerRepository(session)

def get_person_repository(
    session: Session = Depends(get_db_session),
):
    return PostgresPersonRepository(session)