"""Private offline Kaggle CUDA smoke. Purely synthetic; no medical data."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

SOURCE_COMMIT = "__SOURCE_COMMIT__"
TEST_ID = "medical-llm-private-gpu-smoke-v1"
MATRIX_SIZE = 256

def canonical(data: dict) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

def seal(data: dict) -> dict:
    return {**data, "integrity_sha256": hashlib.sha256(canonical(data)).hexdigest()}

def atomic_save(data: dict, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=dest.parent, prefix=".gpu-smoke-", suffix=".tmp",
            encoding="utf-8", delete=False
        ) as handle:
            temp = Path(handle.name)
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, dest)
    finally:
        if temp is not None and temp.exists():
            temp.unlink()

def run(output: Path) -> dict:
    import torch  # Only imported within real Kaggle GPU execution.
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("GPU_SMOKE_FAILED: CUDA_DEVICE_UNAVAILABLE")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    if not gpu_name:
        raise RuntimeError("GPU_SMOKE_FAILED: EMPTY_GPU_NAME")
    a = torch.ones((MATRIX_SIZE, MATRIX_SIZE), dtype=torch.float32, device=device)
    b = torch.ones((MATRIX_SIZE, MATRIX_SIZE), dtype=torch.float32, device=device)
    result = torch.mm(a, b)
    torch.cuda.synchronize()
    error = float((result - MATRIX_SIZE).abs().max().item())
    observed_sum = int(result.sum().item())
    if result.device.type != "cuda" or error != 0.0 or observed_sum != MATRIX_SIZE**3:
        raise RuntimeError("GPU_SMOKE_FAILED: CUDA_COMPUTATION_INCORRECT")
    data = seal({
        "schema_version": 1,
        "test_id": TEST_ID,
        "source_commit": SOURCE_COMMIT,
        "status": "completed",
        "device": "cuda:0",
        "gpu_available": True,
        "gpu_count": int(torch.cuda.device_count()),
        "gpu_name": str(gpu_name),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "matrix_size": MATRIX_SIZE,
        "expected_sum": MATRIX_SIZE**3,
        "observed_sum": observed_sum,
        "max_abs_error": error,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    atomic_save(data, output / "checkpoints" / "gpu-smoke-result.json")
    print("GPU_SMOKE_OK")
    print("GPU_SMOKE_DEVICE: CUDA")
    print("GPU_SMOKE_SOURCE_COMMIT: " + SOURCE_COMMIT)
    print("GPU_SMOKE_CHECKPOINT_SAVED")
    return data

if __name__ == "__main__":
    run(Path("/kaggle/working"))
