from app.infrastructure.db.models import Base
from app.infrastructure.db.session import engine

def init_db() -> None:
    Base.metadata.create_all(bind=engine)