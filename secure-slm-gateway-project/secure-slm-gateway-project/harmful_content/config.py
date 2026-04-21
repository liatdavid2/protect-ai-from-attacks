from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "harmful_content"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_NAME = "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
