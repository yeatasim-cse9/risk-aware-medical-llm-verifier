"""Verify synthetic Kaggle output before accepting it as a resume source."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from research_automation.pilot import CHECKPOINT_NAME, validate  # noqa: E402


def verify(directory: Path, expected_commit: str | None = None) -> dict:
    matches = list(directory.rglob(CHECKPOINT_NAME))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {CHECKPOINT_NAME}; found {len(matches)}")
    state = json.loads(matches[0].read_text(encoding="utf-8-sig"))
    # The Kaggle-runner commit is embedded and may differ from the local
    # checkout. This verifier validates record integrity, and verifies the
    # claimed git SHA format separately.
    from research_automation import pilot
    original = pilot.CODE_COMMIT
    try:
        pilot.CODE_COMMIT = state.get("git_commit")
        validate(state)
    finally:
        pilot.CODE_COMMIT = original
    commit = state["git_commit"]
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("Invalid source commit")
    if expected_commit is not None and commit != expected_commit:
        raise ValueError("Input checkpoint belongs to a different Git commit")
    print(f"VALID CHECKPOINT: {len(state['completed_task_ids'])}/{state['total_tasks']} "
          f"commit={commit} status={state['status']}")
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-commit")
    args = parser.parse_args()
    verify(args.directory, args.expected_commit)
