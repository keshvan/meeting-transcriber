from typing import Protocol

import numpy as np

from app.config.settings import EmbeddingProtectionSettings
from app.domain.voice_embedding import Vector


class EmbeddingProtector(Protocol):
    def protect(self, vector: Vector) -> Vector:
        ...


class NoOpEmbeddingProtector:
    def protect(self, vector: Vector) -> Vector:
        return list(vector)


class MatrixEmbeddingProtector:
    def __init__(self, matrix_path: str, vector_size: int):
        self._matrix = np.load(matrix_path)
        expected_shape = (vector_size, vector_size)
        if self._matrix.shape != expected_shape:
            raise ValueError(
                f"Embedding transform matrix must have shape {expected_shape}, "
                f"got {self._matrix.shape}."
            )

    def protect(self, vector: Vector) -> Vector:
        return (self._matrix @ np.asarray(vector, dtype=np.float32)).tolist()


def build_embedding_protector(
    config: EmbeddingProtectionSettings,
    vector_size: int,
) -> EmbeddingProtector:
    if not config.transform_enabled:
        return NoOpEmbeddingProtector()

    if not config.transform_matrix_path:
        raise RuntimeError(
            "EMBEDDING_TRANSFORM_ENABLED=true requires "
            "EMBEDDING_TRANSFORM_MATRIX_PATH."
        )

    return MatrixEmbeddingProtector(config.transform_matrix_path, vector_size)

