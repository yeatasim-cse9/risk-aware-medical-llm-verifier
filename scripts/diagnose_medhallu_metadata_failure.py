"""Read-only, privacy-limited failure diagnostic for the public metadata-only Kaggle notebook.

This code only requests outputs of the specifically named private MedHallu METADATA
pilot. It never uploads/starts a kernel, reads MedVAL or prints raw remote logs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess

KERNEL = "yeatasimbillah/medical-llm-medhallu-metadata-audit"
SENTINELS = (
    ("ModuleNotFoundError", "PYTHON_IMPORT_ERROR"),
    ("No module named", "PYTHON_IMPORT_ERROR"),
    ("ImportError", "PYTHON_IMPORT_ERROR"),
    ("HTTP Error 403", "REMOTE_HTTP_403"),
    ("HTTPError: 403", "REMOTE_HTTP_403"),
    ("HTTP Error 404", "REMOTE_HTTP_404"),
    ("HTTPError: 404", "REMOTE_HTTP_404"),
    ("HTTP Error 429", "REMOTE_HTTP_429"),
    ("HTTPError: 429", "REMOTE_HTTP_429"),
    ("HTTP Error 502", "REMOTE_HTTP_502"),
    ("HTTPError: 502", "REMOTE_HTTP_502"),
    ("HTTP Error 503", "REMOTE_HTTP_503"),
    ("HTTPError: 503", "REMOTE_HTTP_503"),
    ("URLError", "REMOTE_CONNECTION_ERROR"),
    ("ConnectionError", "REMOTE_CONNECTION_ERROR"),
    ("RemoteDisconnected", "REMOTE_CONNECTION_ERROR"),
    ("PermissionError", "PERMISSION_ERROR"),
    ("KeyError", "SCHEMA_KEY_ERROR"),
    ("ValueError", "SCHEMA_VALUE_ERROR"),
    ("TypeError", "SCHEMA_TYPE_ERROR"),
    ("NameError", "PYTHON_NAME_ERROR"),
    ("FileNotFoundError", "FILE_MISSING"),
)
MAX_FILES = 30
MAX_FILE_BYTES = 1_000_000
SUFFIXES = {".json", ".ipynb", ".txt", ".log", ".html"}


def categories(text: str) -> list[str]:
    return sorted({category for marker, category in SENTINELS if marker in text})


def diagnose(dest: Path) -> dict:
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError("Diagnostic directory must be empty")
    dest.mkdir(parents=True, exist_ok=True)
    command = ["kaggle", "kernels", "output", KERNEL, "-p", str(dest)]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=120,
        check=False,
    )
    if len(result.stdout) > MAX_FILE_BYTES or len(result.stderr) > MAX_FILE_BYTES:
        raise ValueError("Unexpectedly large CLI diagnostic; suppressing")
    paths = sorted(p for p in dest.rglob("*") if p.is_file())
    if len(paths) > MAX_FILES:
        raise ValueError("Unexpected file count; diagnostic suppressed")
    classes = set(categories(result.stdout + "\n" + result.stderr))
    reviewed = 0
    for p in paths:
        if p.is_symlink():
            raise ValueError("Symlink encountered in diagnostic output")
        if p.suffix.lower() not in SUFFIXES or p.stat().st_size > MAX_FILE_BYTES:
            continue
        reviewed += 1
        classes.update(categories(p.read_text(encoding="utf-8", errors="replace")))
    report = {
        "kind": "metadata_notebook_failure_read_only_diagnostic",
        "kaggle_output_command_accepted": result.returncode == 0,
        "downloaded_output_file_count": len(paths),
        "text_diagnostic_files_inspected": reviewed,
        "safe_error_categories": sorted(classes),
        "notebook_rerun": False,
        "dataset_rows_acquired": False,
    }
    print("MEDHALLU_DIAGNOSTIC_OUTPUT_COMMAND_ACCEPTED=" + str(report["kaggle_output_command_accepted"]))
    print("MEDHALLU_DIAGNOSTIC_FILE_COUNT=" + str(report["downloaded_output_file_count"]))
    print("MEDHALLU_DIAGNOSTIC_TEXT_FILE_COUNT=" + str(report["text_diagnostic_files_inspected"]))
    print("MEDHALLU_DIAGNOSTIC_ERROR_CATEGORIES=" + ",".join(report["safe_error_categories"] or ["NOT_IDENTIFIED"]))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("build/medhallu-metadata-diagnostic"))
    args = parser.parse_args()
    diagnose(args.out)
