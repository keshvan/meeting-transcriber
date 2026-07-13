from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.config.settings import settings

engine = create_engine(
    settings.postgres_dsn,  
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=True
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

@contextmanager
def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()