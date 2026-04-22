from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "system_prompt_leakage_output_guard"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_NAME = "gabrielchua/system-prompt-leakage"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TASK_NAME = "system_prompt_leakage_output_guard"
