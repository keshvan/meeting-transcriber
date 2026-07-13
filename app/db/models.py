import uuid
from datetime import datetime, timezone

from sqlalchemy import ForeignKeyConstraint, String, Integer, Float, ForeignKey, DateTime, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

class MeetingORM(Base):
    __tablename__ = "meetings"

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    audio_key: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="NEW")
    duration: Mapped[int | None] = mapped_column(Integer, nullable=True)

    speakers: Mapped[list["SpeakerORM"]] = relationship(back_populates="meeting", cascade="all, delete-orphan")
    segments: Mapped[list["SegmentORM"]] = relationship(back_populates="meeting", cascade="all, delete-orphan")

class PersonORM(Base):
    __tablename__ = "persons"

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

class SpeakerORM(Base):
    __tablename__ = "speakers"

    meeting_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("meetings.id", ondelete="CASCADE"), primary_key=True, nullable=False)
    speaker_id: Mapped[str] = mapped_column(String, nullable=False, primary_key=True)
    person_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("persons.id", ondelete="SET NULL"), nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="UNKNOWN")
    embedding_id: Mapped[str | None] = mapped_column(String, nullable=True)  # qdrant point id

    meeting: Mapped["MeetingORM"] = relationship(back_populates="speakers")
    segments: Mapped[list["SegmentORM"]] = relationship(back_populates="speaker")

class SegmentORM(Base):
    __tablename__ = "segments"

    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    meeting_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False)
    speaker_id: Mapped[str | None] = mapped_column(String, nullable=True)
    start_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    end_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        ForeignKeyConstraint(
            ["meeting_id", "speaker_id"],
            ["speakers.meeting_id", "speakers.speaker_id"],
            ondelete="CASCADE",
        ),
    )

    meeting: Mapped["MeetingORM"] = relationship(back_populates="segments")
    speaker: Mapped["SpeakerORM"] = relationship(back_populates="segments")