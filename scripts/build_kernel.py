"""Build a CPU-only Kaggle script from a specific GitHub commit.

Alternating A/B notebook slots allow a future run to attach the previously
completed slot as a notebook output source. Automated chaining is NOT enabled.
"""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OWNER = "yeatasimbillah"
SLUGS = {"a": "medical-llm-cpu-auto-a", "b": "medical-llm-cpu-auto-b"}


def build(commit: str, slot: str, resume: bool, output: Path) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("An exact lowercase 40-character Git commit SHA is required")
    if slot not in SLUGS:
        raise ValueError("Unknown Kaggle slot")
    previous = "b" if slot == "a" else "a"
    source = ROOT / "src" / "research_automation" / "pilot.py"
    text = source.read_text(encoding="utf-8")
    marker = '"__GIT_COMMIT__"'
    if text.count(marker) != 1:
        raise ValueError("Code commit injection marker missing or duplicated")
    text = text.replace(marker, json.dumps(commit))
    output.mkdir(parents=True, exist_ok=True)
    (output / "runner.py").write_text(text, encoding="utf-8")
    metadata = {
        "id": f"{OWNER}/{SLUGS[slot]}",
        "title": SLUGS[slot],
        "code_file": "runner.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [f"{OWNER}/{SLUGS[previous]}"] if resume else [],
        "model_sources": [],
    }
    (output / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--slot", choices=("a", "b"), default="a")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("build/kaggle"))
    args = parser.parse_args()
    print(json.dumps(build(args.commit, args.slot, args.resume, args.out), indent=2))
