"""Build a single-file isolated Kaggle GPU smoke script at an exact Git SHA."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
KERNEL = "yeatasimbillah/medical-llm-gpu-smoke"
MARKER = '"__SOURCE_COMMIT__"'

def build(source_commit: str, out: Path) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("Exact lowercase 40-character Git commit required")
    raw = (ROOT / "scripts" / "gpu_smoke_runner.py").read_text(encoding="utf-8")
    if raw.count(MARKER) != 1:
        raise ValueError("Source SHA placeholder absent or ambiguous")
    raw = raw.replace(MARKER, json.dumps(source_commit))
    metadata = {
        "id": KERNEL,
        "title": "medical-llm-gpu-smoke",
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
    out.mkdir(parents=True, exist_ok=True)
    (out / "runner.py").write_text(raw, encoding="utf-8")
    (out / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--commit", required=True)
    p.add_argument("--out", type=Path, default=Path("build/gpu-smoke"))
    args = p.parse_args()
    data = build(args.commit, args.out)
    print("PRIVATE_GPU_SMOKE_PACKAGE_OK: " + data["id"])
