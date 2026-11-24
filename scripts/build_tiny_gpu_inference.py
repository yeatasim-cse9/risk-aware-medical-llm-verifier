"""Build an isolated private, offline and frozen Kaggle inference GPU kernel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
KERNEL = "yeatasimbillah/medical-llm-gpu-inference-pilot"
MARKER = '"__SOURCE_COMMIT__"'


def build(source_commit: str, directory: Path) -> dict:
    if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("Exact lowercase 40-character Git SHA required")
    raw = (ROOT / "scripts" / "tiny_gpu_inference_runner.py").read_text(encoding="utf-8")
    if raw.count(MARKER) != 1:
        raise ValueError("Expected exactly one pinned source SHA marker")
    raw = raw.replace(MARKER, json.dumps(source_commit))
    metadata = {
        "id": KERNEL,
        "title": "medical-llm-gpu-inference-pilot",
        "code_file": "runner.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "runner.py").write_text(raw, encoding="utf-8")
    (directory / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--commit", required=True)
    p.add_argument("--out", type=Path, default=Path("build/gpu-inference"))
    args = p.parse_args()
    info = build(args.commit, args.out)
    print("PRIVATE_TINY_GPU_INFERENCE_PACKAGE_OK: " + info["id"])
