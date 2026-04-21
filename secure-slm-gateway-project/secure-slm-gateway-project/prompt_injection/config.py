from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "prompt_injection"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_NAME = "neuralchemy/Prompt-injection-dataset"
DATASET_CONFIG = "core"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
