from pathlib import Path
from typing import Tuple

import numpy as np

from harmful_content_input_guard.run_paths import CACHE_DIR


def _safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("\\", "_").replace(":", "_")


def get_cache_path(model_name: str) -> Path:
    safe_name = _safe_model_name(model_name)
    return CACHE_DIR / f"embeddings_{safe_name}.npz"


def save_embeddings_cache(cache_path: Path, X_train, y_train, X_valid, y_valid, X_test, y_test) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
    )


def load_embeddings_cache(cache_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(cache_path)
    return data["X_train"], data["y_train"], data["X_valid"], data["y_valid"], data["X_test"], data["y_test"]
