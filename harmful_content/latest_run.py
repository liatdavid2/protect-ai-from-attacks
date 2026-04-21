from pathlib import Path

from run_paths import RUNS_DIR


def get_latest_run_dir() -> Path:
    run_dirs = [p for p in RUNS_DIR.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in: {RUNS_DIR}")

    latest_run = sorted(run_dirs)[-1]
    return latest_run