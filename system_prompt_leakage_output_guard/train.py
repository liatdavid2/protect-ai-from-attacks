import json
import os
import sys

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from config import DATASET_NAME, EMBEDDING_MODEL_NAME, TASK_NAME
from data import EVAL_SAMPLE_SIZE, TRAIN_SAMPLE_SIZE, build_label_map, load_splits
from embedding_cache import get_cache_path, load_embeddings_cache, save_embeddings_cache
from features import EmbeddingEncoder
from model import build_model, save_model
from run_paths import create_run_dir


def main():
    print("Loading system prompt leakage dataset...")
    train_df, eval_df, eval_split_name = load_splits()

    cache_path = get_cache_path(TRAIN_SAMPLE_SIZE, EVAL_SAMPLE_SIZE)

    if cache_path.exists():
        X_train, y_train, X_eval, y_eval = load_embeddings_cache(cache_path)
        print(f"Loaded embeddings cache from: {cache_path}")
    else:
        encoder = EmbeddingEncoder(EMBEDDING_MODEL_NAME)

        X_train = encoder.encode(train_df["text"].tolist())
        y_train = train_df["target"].to_numpy()

        X_eval = encoder.encode(eval_df["text"].tolist())
        y_eval = eval_df["target"].to_numpy()

        save_embeddings_cache(
            cache_path=cache_path,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_eval,
            y_eval=y_eval,
        )
        print(f"Saved embeddings cache to: {cache_path}")

    model = build_model()
    model.fit(X_train, y_train)

    run_dir = create_run_dir()

    model_path = run_dir / f"xgb_{TASK_NAME}.joblib"
    metrics_path = run_dir / "metrics.json"
    label_map_path = run_dir / "label_map.json"

    save_model(model, model_path)

    preds = model.predict(X_eval)
    metrics = {
        "dataset_name": DATASET_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "cache_path": str(cache_path),
        eval_split_name: {
            "accuracy": accuracy_score(y_eval, preds),
            "f1_macro": f1_score(y_eval, preds, average="macro"),
            "f1_binary": f1_score(y_eval, preds, average="binary"),
            "classification_report": classification_report(y_eval, preds, output_dict=True),
            "confusion_matrix": confusion_matrix(y_eval, preds).tolist(),
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
