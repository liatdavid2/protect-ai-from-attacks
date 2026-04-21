import os
import sys
import json

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from config import EMBEDDING_MODEL_NAME
from data import load_splits, build_label_map
from embedding_cache import get_cache_path, load_embeddings_cache, save_embeddings_cache
from features import EmbeddingEncoder
from model import build_model, save_model
from run_paths import create_run_dir


def main():
    train_df, valid_df, test_df = load_splits()

    cache_path = get_cache_path()

    if cache_path.exists():
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_embeddings_cache(cache_path)
        print(f"Loaded embeddings cache from: {cache_path}")
    else:
        encoder = EmbeddingEncoder(EMBEDDING_MODEL_NAME)

        X_train = encoder.encode(train_df["text"].tolist())
        y_train = train_df["target"].to_numpy()

        X_valid = encoder.encode(valid_df["text"].tolist())
        y_valid = valid_df["target"].to_numpy()

        X_test = encoder.encode(test_df["text"].tolist())
        y_test = test_df["target"].to_numpy()

        save_embeddings_cache(
            cache_path=cache_path,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            X_test=X_test,
            y_test=y_test,
        )
        print(f"Saved embeddings cache to: {cache_path}")

    model = build_model()
    model.fit(X_train, y_train)

    run_dir = create_run_dir()

    model_path = run_dir / "xgb_harmful_content.joblib"
    metrics_path = run_dir / "metrics.json"
    label_map_path = run_dir / "label_map.json"

    save_model(model, model_path)

    metrics = {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "cache_path": str(cache_path),
    }

    for split_name, X, y in [
        ("validation", X_valid, y_valid),
        ("test", X_test, y_test),
    ]:
        preds = model.predict(X)

        metrics[split_name] = {
            "accuracy": accuracy_score(y, preds),
            "f1_macro": f1_score(y, preds, average="macro"),
            "f1_binary": f1_score(y, preds, average="binary"),
            "classification_report": classification_report(y, preds, output_dict=True),
            "confusion_matrix": confusion_matrix(y, preds).tolist(),
        }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(build_label_map(), f, indent=2)

    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved label map to: {label_map_path}")


if __name__ == "__main__":
    main()