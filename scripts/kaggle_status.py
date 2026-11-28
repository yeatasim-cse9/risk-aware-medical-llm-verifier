"""Fail-closed Kaggle status preflight for the synthetic CPU pilot.

Use the Kaggle CLI's supported JSON listing rather than brittle CSV/text
header parsing. Unknown responses and API failures never authorize submission.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess

ACTIVE = {"running", "queued", "pending", "starting", "initializing"}
TERMINAL = {"complete", "error", "cancelacknowledged", "cancelled", "failed"}


def parse_status(text: str) -> str:
    match = re.search(
        r'has status\s+["\']?(?:KernelWorkerStatus\.)?([A-Za-z_]+)',
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError("Unrecognized Kaggle status format")
    state = match.group(1).lower()
    if state not in ACTIVE | TERMINAL:
        raise ValueError("Unexpected Kaggle notebook status")
    return state


def parse_kernel_list(text: str, kernel: str) -> bool:
    """Detect an exact Kaggle notebook ref from JSON or CSV listings.

    Some Kaggle CLI versions produce an empty CSV header even when
    format=json was requested. A failed API call never implies absence.
    """
    cleaned = text.strip()
    if not cleaned or re.fullmatch(
        r"(?:No kernels(?: found)?|Not found)\.?", cleaned, flags=re.IGNORECASE
    ):
        return False

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        reader = csv.DictReader(io.StringIO(cleaned))
        fields = reader.fieldnames or []
        names = {x.strip().lower(): x for x in fields if isinstance(x, str)}
        if "ref" not in names or "title" not in names or len(fields) < 2:
            raise ValueError("Unknown Kaggle list format (not JSON or CSV)")
        ref_col = names["ref"]
        refs = set()
        for row in reader:
            if not row.get(ref_col) or any(key is None for key in row):
                raise ValueError("Malformed Kaggle CSV notebook listing")
            refs.add(row[ref_col].strip())
        return kernel in refs

    if not isinstance(data, list):
        raise ValueError("Unexpected Kaggle JSON notebook list structure")
    refs = set()
    for row in data:
        if not isinstance(row, dict):
            raise ValueError("Unexpected row in Kaggle JSON notebook list")
        normalized = {str(k).strip().lower(): v for k, v in row.items()}
        ref = normalized.get("ref")
        if not isinstance(ref, str):
            raise ValueError("Kaggle list entry missing ref")
        refs.add(ref.strip())
    return kernel in refs

def kernel_exists(kernel: str) -> bool:
    owner, sep, slug = kernel.partition("/")
    if not sep or not owner or not slug or "/" in slug:
        raise ValueError("Expected exact owner/notebook-slug")
    cmd = [
        "kaggle", "kernels", "list", "--mine", "--search", slug,
        "--format", "json", "--page-size", "100",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    if proc.returncode != 0:
        raise RuntimeError("Kaggle notebook listing failed; refusing to proceed")
    if proc.stderr.strip():
        # Do not leak arbitrary remote response or credentials in logs.
        raise RuntimeError("Kaggle CLI produced unexpected listing diagnostics")
    try:
        return parse_kernel_list(proc.stdout, kernel)
    except ValueError as exc:
        # Deliberately non-sensitive diagnostic: helps future debugging,
        # without revealing notebook names, credentials or raw CLI output.
        fingerprint = hashlib.sha256(proc.stdout.encode("utf-8")).hexdigest()
        print(
            f"Unexpected notebook list format (characters={len(proc.stdout)}, "
            f"sha256={fingerprint}); submission blocked."
        )
        raise


def check(kernel: str, require: str) -> str:
    if not kernel_exists(kernel):
        state = "missing"
    else:
        proc = subprocess.run(
            ["kaggle", "kernels", "status", kernel],
            capture_output=True, text=True, timeout=90,
        )
        if proc.returncode != 0:
            raise RuntimeError("Existing Kaggle notebook status failed; refusing to proceed")
        state = parse_status(proc.stdout + "\n" + proc.stderr)
    print(f"Kernel status: {kernel} = {state}")
    if require == "complete" and state != "complete":
        raise RuntimeError("Previous Kaggle notebook must be complete")
    if require == "not-active" and state not in {"missing", "complete"}:
        raise RuntimeError("Kaggle notebook active or failed; manual review required")
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel")
    parser.add_argument("--require", choices=("complete", "not-active", "any"),
                        default="any")
    args = parser.parse_args()
    check(args.kernel, args.require)
