"""Fail-closed parser of Kaggle CLI status for safe CPU pilot submission.

The Kaggle CLI may change its display format; unknown statuses are blocked.
"""
import argparse
import re
import subprocess

ACTIVE = {"running", "queued", "pending", "starting", "initializing"}
TERMINAL = {"complete", "error", "cancelacknowledged", "cancelled", "failed"}


def parse_status(text: str) -> str:
    match = re.search(
        r'has status\s+["\']?(?:KernelWorkerStatus\.)?([A-Za-z_]+)',
        text, flags=re.IGNORECASE
    )
    if not match:
        raise ValueError("Unrecognized Kaggle status format")
    value = match.group(1).lower()
    if value not in ACTIVE | TERMINAL:
        raise ValueError("Unknown Kaggle status; fail closed")
    return value


def check(kernel: str, require: str) -> str:
    proc = subprocess.run(["kaggle", "kernels", "status", kernel],
                          capture_output=True, text=True, timeout=90)
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        if re.search(r'\b404\b|not found|does not exist', output, re.IGNORECASE):
            state = "missing"
        else:
            raise RuntimeError("Kaggle status request failed; refusing to proceed")
    else:
        state = parse_status(output)
    print(f"Kernel status: {kernel} = {state}")
    if require == "complete" and state != "complete":
        raise RuntimeError("The previous Kaggle kernel must be complete")
    if require == "not-active" and state not in {"missing", "complete"}:
        raise RuntimeError("Kernel active or failed; manual review required")
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel")
    parser.add_argument("--require", choices=("complete", "not-active", "any"),
                        default="any")
    args = parser.parse_args()
    check(args.kernel, args.require)
