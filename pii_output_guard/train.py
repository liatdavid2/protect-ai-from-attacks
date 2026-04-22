import os
import sys
import json

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from pii_output_guard.config import DATASET_NAME, EMBEDDING_MODEL_NAME
from pii_output_guard.data import load_splits, build_label_map
from pii_output_guard.embedding_cache import get_cache_path, load_embeddings_cache, save_embeddings_cache
from pii_output_guard.features import EmbeddingEncoder
from pii_output_guard.model import build_model, save_model
from pii_output_guard.run_paths import create_run_dir


def main():
    print("Loading PII output guard dataset...")
    train_df, valid_df = load_splits()

    cache_path = get_cache_path()

    if cache_path.exists():
        X_train, y_train, X_valid, y_valid = load_embeddings_cache(cache_path)
        print(f"Loaded embeddings cache from: {cache_path}")
    else:
        encoder = EmbeddingEncoder(EMBEDDING_MODEL_NAME)

        X_train = encoder.encode(train_df["text"].tolist())
        y_train = train_df["target"].to_numpy()

        X_valid = encoder.encode(valid_df["text"].tolist())
        y_valid = valid_df["target"].to_numpy()

        save_embeddings_cache(
            cache_path=cache_path,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
        )
        print(f"Saved embeddings cache to: {cache_path}")

    model = build_model()
    model.fit(X_train, y_train)

    run_dir = create_run_dir()

    model_path = run_dir / "xgb_pii_output_guard.joblib"
    metrics_path = run_dir / "metrics.json"
    label_map_path = run_dir / "label_map.json"

    save_model(model, model_path)

    preds = model.predict(X_valid)

    metrics = {
        "dataset_name": DATASET_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "cache_path": str(cache_path),
        "validation": {
            "accuracy": accuracy_score(y_valid, preds),
            "f1_macro": f1_score(y_valid, preds, average="macro"),
            "f1_binary": f1_score(y_valid, preds, average="binary"),
            "classification_report": classification_report(y_valid, preds, output_dict=True),
            "confusion_matrix": confusion_matrix(y_valid, preds).tolist(),
        },
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
