"""Build private Kaggle CPU package from pinned Git source; no data bundled."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
KERNEL = "yeatasimbillah/medical-llm-medhallu-metadata-audit"

def build(source_commit: str, output: Path) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("Exact Git source commit required")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing stale Kaggle package directory")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "src" / "research_data" / "manager.py", output / "manager.py")
    runner = (
        '"""MedHallu public metadata only; NO dataset rows are requested."""\n'
        "from pathlib import Path\n"
        "from manager import build_report, verify_report, write_report\n"
        f"SOURCE_COMMIT = {source_commit!r}\n"
        "if __name__ == '__main__':\n"
        "    report = build_report(SOURCE_COMMIT)\n"
        "    verify_report(report, SOURCE_COMMIT)\n"
        "    write_report(report, Path('/kaggle/working'))\n"
        "    print('PRIVATE_MEDHALLU_METADATA_ONLY_COMPLETED')\n"
    )
    (output / "runner.py").write_text(runner, encoding="utf-8")
    metadata = {
        "id": KERNEL,
        "title": "medical-llm-medhallu-metadata-audit",
        "code_file": "runner.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (output / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--out", type=Path, default=Path("build/medhallu-metadata"))
    args = parser.parse_args()
    result = build(args.commit, args.out)
    print("MEDHALLU_PRIVATE_METADATA_PACKAGE_READY: " + result["id"])
