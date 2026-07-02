from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


Vector = list[float]


@dataclass(frozen=True)
class VoiceEmbedding:
    person_id: str
    person_name: str
    vector: Vector
    embedding_id: str = field(default_factory=lambda: str(uuid4()))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpeakerCandidate:
    person_id: str
    person_name: str
    score: float


@dataclass(frozen=True)
class PersonCentroid:
    person_id: str
    person_name: str
    vector: Vector
    embedding_count: int
    updated_at: datetime


@dataclass(frozen=True)
class SpeakerIdentificationResult:
    person_id: str | None
    person_name: str
    score: float
    is_known: bool
