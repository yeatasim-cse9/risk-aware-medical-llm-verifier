"""Verify synthetic tiny-transformer GPU inference and saved model bytes.

No torch import or model deserialization on GitHub, avoiding pickle execution.
Proof is the pinned Kaggle execution plus this independently fetched result.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re

from tiny_gpu_inference_runner import (
    PILOT_ID, MANIFEST_FILE, WEIGHTS_FILE, LABELS, SAMPLES,
    SEQUENCE_LENGTH, SEED, canonical, input_tokens,
)

FIELDS = {
    "schema_version", "pilot_id", "source_commit", "status", "model",
    "model_pretrained", "medical_experiment", "device", "gpu_count",
    "gpu_name", "torch_version", "torch_cuda_version", "parameter_count",
    "seed", "num_samples", "sequence_length", "labels", "input_sha256",
    "logits", "logits_sha256", "probabilities", "predictions",
    "weights_file", "weights_size_bytes", "weights_sha256",
    "restore_max_abs_error", "completed_at_utc", "integrity_sha256",
}


def hex_digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def verify(root: Path, expected_commit: str | None = None) -> dict:
    files = list(root.rglob(MANIFEST_FILE))
    if len(files) != 1:
        raise ValueError("Exactly one synthetic GPU model manifest required")
    raw = files[0].read_bytes()
    if len(raw) > 200_000:
        raise ValueError("GPU model manifest too large")
    state = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(state, dict) or set(state) != FIELDS:
        raise ValueError("GPU inference manifest schema unexpected")
    checksum = state["integrity_sha256"]
    body = {k: v for k, v in state.items() if k != "integrity_sha256"}
    if not hex_digest(checksum) or checksum != hashlib.sha256(canonical(body)).hexdigest():
        raise ValueError("GPU inference manifest checksum mismatch")
    sha = state["source_commit"]
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("GPU inference invalid source commit")
    if expected_commit is not None and sha != expected_commit:
        raise ValueError("GPU inference source provenance mismatch")
    try:
        stamp = datetime.fromisoformat(state["completed_at_utc"])
    except (ValueError, TypeError):
        raise ValueError("GPU inference completed timestamp invalid") from None
    if stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
        raise ValueError("GPU inference timestamp not UTC")

    if (
        type(state["schema_version"]) is not int or state["schema_version"] != 1
        or state["pilot_id"] != PILOT_ID
        or state["status"] != "completed"
        or state["model"] != "untrained_tiny_transformer_one_encoder_layer"
        or state["model_pretrained"] is not False
        or state["medical_experiment"] is not False
        or state["device"] != "cuda:0"
        or type(state["gpu_count"]) is not int or state["gpu_count"] < 1
        or not isinstance(state["gpu_name"], str) or not state["gpu_name"].strip()
        or not isinstance(state["torch_version"], str) or not state["torch_version"]
        or not isinstance(state["torch_cuda_version"], str) or not state["torch_cuda_version"]
        or type(state["parameter_count"]) is not int or not 1_000 <= state["parameter_count"] <= 100_000
        or type(state["seed"]) is not int or state["seed"] != SEED
        or type(state["num_samples"]) is not int or state["num_samples"] != SAMPLES
        or type(state["sequence_length"]) is not int or state["sequence_length"] != SEQUENCE_LENGTH
        or state["labels"] != list(LABELS)
        or not hex_digest(state["input_sha256"])
        or state["input_sha256"] != hashlib.sha256(canonical(input_tokens())).hexdigest()
        or state["weights_file"] != WEIGHTS_FILE
        or type(state["weights_size_bytes"]) is not int
        or not 100 <= state["weights_size_bytes"] <= 10_000_000
        or not hex_digest(state["weights_sha256"])
        or not hex_digest(state["logits_sha256"])
        or type(state["restore_max_abs_error"]) is not float
        or not math.isfinite(state["restore_max_abs_error"])
        or not 0.0 <= state["restore_max_abs_error"] <= 1e-5
    ):
        raise ValueError("GPU inference manifest attributes invalid")

    logits = state["logits"]
    probabilities = state["probabilities"]
    predictions = state["predictions"]
    if (
        not isinstance(logits, list) or len(logits) != SAMPLES
        or not isinstance(probabilities, list) or len(probabilities) != SAMPLES
        or not isinstance(predictions, list) or len(predictions) != SAMPLES
        or hashlib.sha256(canonical(logits)).hexdigest() != state["logits_sha256"]
    ):
        raise ValueError("GPU inference output structure or logits digest invalid")
    for row, probs, prediction in zip(logits, probabilities, predictions):
        if not isinstance(row, list) or len(row) != len(LABELS):
            raise ValueError("GPU inference logits invalid")
        if not isinstance(probs, list) or len(probs) != len(LABELS):
            raise ValueError("GPU inference probability dimensions invalid")
        if any(type(v) is not float or not math.isfinite(v) for v in row + probs):
            raise ValueError("GPU inference nonfinite or nonfloat output")
        if any(v < 0.0 or v > 1.0 for v in probs):
            raise ValueError("GPU inference probabilities out of bounds")
        if abs(sum(probs) - 1.0) > 1e-5:
            raise ValueError("GPU inference probability normalization failed")
        peak = max(range(len(row)), key=lambda i: row[i])
        if type(prediction) is not int or prediction != peak:
            raise ValueError("GPU inference predicted class mismatch")
        shifted = [math.exp(v - max(row)) for v in row]
        denom = sum(shifted)
        expected = [v / denom for v in shifted]
        if any(abs(a - b) > 1e-5 for a, b in zip(probs, expected)):
            raise ValueError("GPU inference softmax mismatch")

    weights = list(root.rglob(WEIGHTS_FILE))
    if len(weights) != 1 or weights[0].parent != files[0].parent:
        raise ValueError("GPU model weights missing or misplaced")
    weight_bytes = weights[0].read_bytes()
    if len(weight_bytes) != state["weights_size_bytes"]:
        raise ValueError("GPU model weights size mismatch")
    if hashlib.sha256(weight_bytes).hexdigest() != state["weights_sha256"]:
        raise ValueError("GPU model weights SHA256 mismatch")
    print("TINY_GPU_MODEL_REMOTE_INFERENCE_VERIFIED")
    print("TINY_GPU_MODEL_CHECKPOINT_AND_RESTORE_VERIFIED")
    print("TINY_GPU_MODEL_SOURCE_COMMIT_MATCHED" if expected_commit else
          "TINY_GPU_MODEL_SOURCE_SHA_FORMAT_VALID")
    return state


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("--expected-commit")
    a = p.parse_args()
    verify(a.directory, a.expected_commit)
