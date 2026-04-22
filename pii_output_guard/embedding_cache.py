from pathlib import Path
from typing import Tuple

import numpy as np

from pii_output_guard.config import EMBEDDING_MODEL_NAME
from pii_output_guard.run_paths import CACHE_DIR


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(":", "_").replace("-", "_")


def get_cache_path() -> Path:
    safe_model = _safe_name(EMBEDDING_MODEL_NAME)
    return CACHE_DIR / f"embeddings_{safe_model}.npz"


def save_embeddings_cache(
    cache_path: Path,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
    )


def load_embeddings_cache(
    cache_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(cache_path)
    return (
        data["X_train"],
        data["y_train"],
        data["X_valid"],
        data["y_valid"],
    )
