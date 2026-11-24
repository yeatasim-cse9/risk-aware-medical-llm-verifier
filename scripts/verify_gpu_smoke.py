"""Verify the saved private CUDA smoke checkpoint independently on GitHub."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

FILENAME = "gpu-smoke-result.json"
FIELDS = {
    "schema_version", "test_id", "source_commit", "status", "device",
    "gpu_available", "gpu_count", "gpu_name", "torch_version",
    "torch_cuda_version", "matrix_size", "expected_sum", "observed_sum",
    "max_abs_error", "completed_at_utc", "integrity_sha256",
}

def verify(root: Path, expected_commit: str | None = None) -> dict:
    paths = list(root.rglob(FILENAME))
    if len(paths) != 1:
        raise ValueError("Exactly one GPU smoke checkpoint required")
    state = json.loads(paths[0].read_text(encoding="utf-8-sig"))
    if not isinstance(state, dict) or set(state) != FIELDS:
        raise ValueError("Unrecognized GPU smoke checkpoint schema")
    data = {k: v for k, v in state.items() if k != "integrity_sha256"}
    sha = hashlib.sha256(json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")).hexdigest()
    if state["integrity_sha256"] != sha:
        raise ValueError("GPU smoke SHA256 integrity mismatch")
    commit = state["source_commit"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Invalid Git source SHA")
    if expected_commit is not None and commit != expected_commit:
        raise ValueError("Wrong frozen source commit")
    if (
        type(state["schema_version"]) is not int or state["schema_version"] != 1
        or state["test_id"] != "medical-llm-private-gpu-smoke-v1"
        or state["status"] != "completed"
        or state["device"] != "cuda:0"
        or state["gpu_available"] is not True
        or type(state["gpu_count"]) is not int or state["gpu_count"] < 1
        or not isinstance(state["gpu_name"], str) or not state["gpu_name"].strip()
        or not isinstance(state["torch_version"], str) or not state["torch_version"]
        or not isinstance(state["torch_cuda_version"], str) or not state["torch_cuda_version"]
        or type(state["matrix_size"]) is not int or state["matrix_size"] != 256
        or type(state["expected_sum"]) is not int or state["expected_sum"] != 256**3
        or type(state["observed_sum"]) is not int or state["observed_sum"] != 256**3
        or type(state["max_abs_error"]) is not float or state["max_abs_error"] != 0.0
        or not isinstance(state["completed_at_utc"], str)
        or not state["completed_at_utc"].endswith("+00:00")
    ):
        raise ValueError("CUDA checkpoint values not valid")
    print("VERIFIED_PRIVATE_GPU_SMOKE: CUDA_MATRIX_OK")
    print("VERIFIED_PRIVATE_GPU_SMOKE: SOURCE_SHA_MATCHED" if expected_commit else
          "VERIFIED_PRIVATE_GPU_SMOKE: SHA_FORMAT_VALID")
    return state

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("--expected-commit")
    a = p.parse_args()
    verify(a.directory, a.expected_commit)
