import json
import os
import sys
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from prompt_injection.config import DATASET_NAME, DATASET_CONFIG, EMBEDDING_MODEL_NAME
from prompt_injection.data import build_label_map, load_splits
from prompt_injection.embedding_cache import get_cache_path, load_embeddings_cache, save_embeddings_cache
from prompt_injection.features import EmbeddingEncoder
from prompt_injection.model import build_model, evaluate_model, save_model
from prompt_injection.run_paths import create_run_dir


def main():
    print("Loading prompt injection dataset...")
    train_df, valid_df, test_df = load_splits()

    cache_path = get_cache_path(EMBEDDING_MODEL_NAME)

    if cache_path.exists():
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_embeddings_cache(cache_path)
        print(f"Loaded embeddings cache from: {cache_path}")
    else:
        encoder = EmbeddingEncoder(EMBEDDING_MODEL_NAME)
        X_train = encoder.encode(train_df["text"].tolist())
        y_train = train_df["label"].to_numpy()
        X_valid = encoder.encode(valid_df["text"].tolist())
        y_valid = valid_df["label"].to_numpy()
        X_test = encoder.encode(test_df["text"].tolist())
        y_test = test_df["label"].to_numpy()
        save_embeddings_cache(cache_path, X_train, y_train, X_valid, y_valid, X_test, y_test)
        print(f"Saved embeddings cache to: {cache_path}")

    model = build_model()
    model.fit(X_train, y_train)

    run_dir = create_run_dir()
    model_path = run_dir / "xgb_prompt_injection.joblib"
    metrics_path = run_dir / "metrics.json"
    label_map_path = run_dir / "label_map.json"

    save_model(model, model_path)

    metrics = {
        "dataset_name": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "cache_path": str(cache_path),
        "validation": evaluate_model(model, X_valid, y_valid),
        "test": evaluate_model(model, X_test, y_test),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(build_label_map(train_df), f, indent=2)

    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved label map to: {label_map_path}")


if __name__ == "__main__":
    main()
