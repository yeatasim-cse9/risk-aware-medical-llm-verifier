"""Private GPU inference pilot on a tiny, UNTRAINED transformer.

No medical evidence, pretrained weights, patient data, downloads or internet.
This tests real CUDA neural-network inference + saved/reloaded weights only.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

SOURCE_COMMIT = "__SOURCE_COMMIT__"
PILOT_ID = "private-gpu-tiny-transformer-inference-v1"
MANIFEST_FILE = "tiny-transformer-inference.json"
WEIGHTS_FILE = "tiny-transformer-weights.pt"
SAMPLES = 4
SEQUENCE_LENGTH = 8
LABELS = ("synthetic_class_0", "synthetic_class_1", "synthetic_class_2")
EMBEDDING = 32
SEED = 1729


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def seal(data: dict) -> dict:
    return {**data, "integrity_sha256": sha256(canonical(data))}


def atomic_json_write(data: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, delete=False,
            prefix=".model-manifest-", suffix=".tmp"
        ) as handle:
            temp = Path(handle.name)
            handle.write(json.dumps(data, sort_keys=True, indent=2,
                                    allow_nan=False).encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, destination)
    finally:
        if temp is not None and temp.exists():
            temp.unlink()


def input_tokens() -> list[list[int]]:
    # Synthetic token IDs only. They are not tokenized medical text.
    return [list(range(i * SEQUENCE_LENGTH, (i + 1) * SEQUENCE_LENGTH))
            for i in range(SAMPLES)]


def build_model(torch):
    nn = torch.nn

    class TinyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(128, EMBEDDING)
            layer = nn.TransformerEncoderLayer(
                d_model=EMBEDDING, nhead=4, dim_feedforward=64,
                dropout=0.0, batch_first=True, activation="gelu"
            )
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=1, enable_nested_tensor=False
            )
            self.head = nn.Linear(EMBEDDING, len(LABELS))

        def forward(self, ids):
            embeddings = self.embedding(ids)
            encoded = self.encoder(embeddings)
            return self.head(encoded.mean(dim=1))

    return TinyTransformer()


def run(output_root: Path) -> dict:
    import torch  # Kaggle GPU runtime only; GitHub tests have no torch dependency.

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("GPU_MODEL_INFERENCE_FAILED: CUDA_UNAVAILABLE")
    torch.manual_seed(SEED)
    device = torch.device("cuda:0")
    model = build_model(torch).to(device).eval()
    if not all(p.device.type == "cuda" for p in model.parameters()):
        raise RuntimeError("GPU_MODEL_INFERENCE_FAILED: WEIGHTS_NOT_ON_GPU")

    inputs = torch.tensor(input_tokens(), dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model(inputs)
        probs_tensor = torch.softmax(output, dim=-1)
    torch.cuda.synchronize()
    if (
        output.device.type != "cuda" or probs_tensor.device.type != "cuda"
        or tuple(output.shape) != (SAMPLES, len(LABELS))
        or not bool(torch.isfinite(output).all().item())
        or not bool(torch.isfinite(probs_tensor).all().item())
    ):
        raise RuntimeError("GPU_MODEL_INFERENCE_FAILED: OUTPUT_INVALID")

    checkpoints = output_root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    weights = checkpoints / WEIGHTS_FILE
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=checkpoints, delete=False,
            prefix=".tiny-weights-", suffix=".tmp"
        ) as handle:
            temp = Path(handle.name)
            torch.save(model.state_dict(), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, weights)
    finally:
        if temp is not None and temp.exists():
            temp.unlink()

    # Verify saved weights can restore an *identical* CUDA model in same run.
    restored = build_model(torch).to(device).eval()
    restored.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    with torch.inference_mode():
        restored_logits = restored(inputs)
    torch.cuda.synchronize()
    restore_error = float((output - restored_logits).abs().max().item())
    if not math.isfinite(restore_error) or restore_error > 1e-5:
        raise RuntimeError("GPU_MODEL_INFERENCE_FAILED: CHECKPOINT_RESTORE_MISMATCH")

    logits = [[round(float(v), 8) for v in row]
              for row in output.detach().cpu().tolist()]
    probabilities = [[round(float(v), 8) for v in row]
                     for row in probs_tensor.detach().cpu().tolist()]
    predictions = [max(range(len(row)), key=lambda i: row[i])
                   for row in logits]
    weights_bytes = weights.read_bytes()
    if not 100 <= len(weights_bytes) <= 10_000_000:
        raise RuntimeError("GPU_MODEL_INFERENCE_FAILED: WEIGHTS_SIZE_UNEXPECTED")

    body = {
        "schema_version": 1,
        "pilot_id": PILOT_ID,
        "source_commit": SOURCE_COMMIT,
        "status": "completed",
        "model": "untrained_tiny_transformer_one_encoder_layer",
        "model_pretrained": False,
        "medical_experiment": False,
        "device": "cuda:0",
        "gpu_count": int(torch.cuda.device_count()),
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "parameter_count": sum(int(p.numel()) for p in model.parameters()),
        "seed": SEED,
        "num_samples": SAMPLES,
        "sequence_length": SEQUENCE_LENGTH,
        "labels": list(LABELS),
        "input_sha256": sha256(canonical(input_tokens())),
        "logits": logits,
        "logits_sha256": sha256(canonical(logits)),
        "probabilities": probabilities,
        "predictions": predictions,
        "weights_file": WEIGHTS_FILE,
        "weights_size_bytes": len(weights_bytes),
        "weights_sha256": sha256(weights_bytes),
        "restore_max_abs_error": restore_error,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest = seal(body)
    atomic_json_write(manifest, checkpoints / MANIFEST_FILE)
    print("TINY_GPU_MODEL_INFERENCE_OK")
    print("TINY_GPU_MODEL_CUDA: YES")
    print("TINY_GPU_MODEL_WEIGHTS_RESTORE_OK")
    print("TINY_GPU_MODEL_CHECKPOINT_SAVED")
    print("TINY_GPU_MODEL_SOURCE_SHA: " + SOURCE_COMMIT)
    return manifest


if __name__ == "__main__":
    run(Path("/kaggle/working"))
