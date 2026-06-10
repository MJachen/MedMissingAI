from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def save_run_metadata(save_dir: str | Path, config_paths: list[Path]) -> None:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    (save_path / "git_commit.txt").write_text(_git_commit(), encoding="utf-8")
    (save_path / "command.txt").write_text(_command(), encoding="utf-8")

    snapshot_dir = save_path / "config_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for config_path in config_paths:
        path = Path(config_path)
        if path.exists():
            shutil.copy2(path, snapshot_dir / path.name)


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return f"unavailable: {exc}\n"

    if result.returncode == 0:
        return result.stdout.strip() + "\n"
    return f"unavailable: {result.stderr.strip()}\n"


def _command() -> str:
    return " ".join([sys.executable, *sys.argv]) + "\n"
