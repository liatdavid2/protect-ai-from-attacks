from pathlib import Path

from harmful_content_input_guard.run_paths import RUNS_DIR


def get_latest_run_dir() -> Path:
    run_dirs = [p for p in RUNS_DIR.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in: {RUNS_DIR}")
    return sorted(run_dirs)[-1]
