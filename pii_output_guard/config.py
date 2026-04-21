from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "pii_output_guard"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_NAME = "ai4privacy/pii-masking-300k"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
