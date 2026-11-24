"""Fail-closed Kaggle status preflight for a synthetic CPU pilot.

A notebook that has never been created has no status. Check the
authenticated owner's notebook list first; never confuse an API failure
with an absent notebook.
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import subprocess

ACTIVE = {"running", "queued", "pending", "starting", "initializing"}
TERMINAL = {"complete", "error", "cancelacknowledged", "cancelled", "failed"}


def parse_status(text: str) -> str:
    match = re.search(
        r'has status\s+["\']?(?:KernelWorkerStatus\.)?([A-Za-z_]+)',
        text, flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError("Unrecognized Kaggle status format")
    value = match.group(1).lower()
    if value not in ACTIVE | TERMINAL:
        raise ValueError("Unknown Kaggle status; fail closed")
    return value


def parse_list(text: str, kernel: str) -> bool:
    """Check exact owner/slug against successful owner-scoped CSV listing."""
    text = text.strip()
    if not text or text.lower() == "no kernels found":
        return False
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("Missing Kaggle CSV header")
    fields = {f.strip().lower(): f for f in reader.fieldnames}
    if "ref" not in fields:
        raise ValueError("Unknown Kaggle CSV format; fail closed")
    ref_col = fields["ref"]
    refs = {row[ref_col].strip() for row in reader if row.get(ref_col)}
    return kernel in refs


def kernel_exists(kernel: str) -> bool:
    owner, sep, slug = kernel.partition("/")
    if not sep or not owner or not slug or "/" in slug:
        raise ValueError("Expected owner/notebook-slug")
    cmd = ["kaggle", "kernels", "list", "--mine",
           "--search", slug, "--csv", "--page-size", "100"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    if proc.returncode != 0:
        raise RuntimeError("Kaggle notebook listing failed; refusing to submit")
    return parse_list(proc.stdout, kernel)


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
