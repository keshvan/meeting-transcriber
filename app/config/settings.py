from dataclasses import dataclass
import os

import torch

from dotenv import load_dotenv
load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default

    return int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default

    return float(value)


@dataclass(frozen=True)
class QdrantSettings:
    url: str
    api_key: str | None
    timeout: float
    prefer_grpc: bool
    centroids_collection: str
    embeddings_collection: str
    vector_size: int
    distance: str
    create_collections: bool
    create_payload_indexes: bool
    on_disk_vectors: bool


@dataclass(frozen=True)
class SpeakerIdentificationSettings:
    centroid_top_k: int
    sample_limit_per_person: int
    similarity_threshold: float
    unknown_speaker_name: str


@dataclass(frozen=True)
class EmbeddingProtectionSettings:
    transform_enabled: bool
    transform_matrix_path: str | None

@dataclass(frozen=True)
class Settings:
    hf_token: str
    device: torch.device
    diarization_model: str
    embedding_model: str
    embedding_min_duration: float
    stt_provider: str
    stt_model: str
    qdrant: QdrantSettings
    speaker_identification: SpeakerIdentificationSettings
    embedding_protection: EmbeddingProtectionSettings
    postgres_dsn: str
    base_dir: str

def resolve_device(device_name: str) -> torch.device:
    device_name = device_name.lower()

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DEVICE=cuda, but CUDA is not available."
            )
        return torch.device("cuda")
    
    if device_name == "cpu":
        return torch.device("cpu")
    
    raise RuntimeError(
        f"{device_name} - unknown device"
    )

settings = Settings(
    hf_token=os.getenv("HF_TOKEN", ""),
    device=resolve_device(os.getenv("DEVICE", "auto")),
    diarization_model=os.getenv(
        "DIARIZATION_MODEL",
        "pyannote/speaker-diarization-community-1",
    ),
    embedding_model=os.getenv("EMBEDDING_MODEL", "speechbrain/ecapa-tdnn"),
    embedding_min_duration=_get_float("EMBEDDING_MIN_DURATION", 0.5),
    stt_provider=os.getenv("STT_PROVIDER", "whisper_local"),
    stt_model=os.getenv("STT_MODEL", "base"),
    qdrant=QdrantSettings(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
        timeout=_get_float("QDRANT_TIMEOUT", 5.0),
        prefer_grpc=_get_bool("QDRANT_PREFER_GRPC", False),
        centroids_collection=os.getenv(
            "QDRANT_CENTROIDS_COLLECTION",
            "voice_centroids",
        ),
        embeddings_collection=os.getenv(
            "QDRANT_EMBEDDINGS_COLLECTION",
            "voice_embeddings",
        ),
        vector_size=_get_int("QDRANT_VECTOR_SIZE", 192),
        distance=os.getenv("QDRANT_DISTANCE", "COSINE"),
        create_collections=_get_bool("QDRANT_CREATE_COLLECTIONS", True),
        create_payload_indexes=_get_bool("QDRANT_CREATE_PAYLOAD_INDEXES", True),
        on_disk_vectors=_get_bool("QDRANT_ON_DISK_VECTORS", False),
    ),
    speaker_identification=SpeakerIdentificationSettings(
        centroid_top_k=_get_int("SPEAKER_CENTROID_TOP_K", 10),
        sample_limit_per_person=_get_int("SPEAKER_SAMPLE_LIMIT_PER_PERSON", 50),
        similarity_threshold=_get_float("SPEAKER_SIMILARITY_THRESHOLD", 0.75),
        unknown_speaker_name=os.getenv("SPEAKER_UNKNOWN_NAME", "Unknown"),
    ),
    embedding_protection=EmbeddingProtectionSettings(
        transform_enabled=_get_bool("EMBEDDING_TRANSFORM_ENABLED", False),
        transform_matrix_path=os.getenv("EMBEDDING_TRANSFORM_MATRIX_PATH") or None,
    ),
    postgres_dsn=os.getenv("POSTGRRES_DSN", "postgresql+psycopg2://postgres:postgres@localhost:5432/meetings"),
    base_dir=os.getenv("BASE_DIR", "./data/meetings")
)
