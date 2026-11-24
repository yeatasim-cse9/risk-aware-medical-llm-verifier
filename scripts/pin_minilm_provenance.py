"""Step 10.5D: metadata-only provenance and feasibility for one MiniLM model.

No model/tokenizer bytes, weights, datasets, patient records, or credentials are
downloaded. This reads ONLY public Hugging Face model metadata and tree entries.
It does not execute, install, submit, or benchmark a pretrained model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import urllib.error
import urllib.request

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ROOT = "https://huggingface.co/api/models/" + MODEL
REVISION_PATTERN = re.compile(r"[a-f0-9]{40}\Z")
SHA256_PATTERN = re.compile(r"[a-f0-9]{64}\Z")
FILES = (
    "model.safetensors", "config.json", "tokenizer.json",
    "tokenizer_config.json", "vocab.txt", "modules.json",
    "1_Pooling/config.json",
)
MAX_METADATA_BYTES = 4 * 1024 * 1024


class ProvenanceError(ValueError):
    pass


def request_json(url: str):
    if url != ROOT + "/revision/main" and not re.fullmatch(
        re.escape(ROOT) + r"/tree/[0-9a-f]{40}\?recursive=true", url
    ):
        raise ProvenanceError("UNTRUSTED_METADATA_ENDPOINT")
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "thesis-safe-model-provenance/1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            content = response.read(MAX_METADATA_BYTES + 1)
            if len(content) > MAX_METADATA_BYTES:
                raise ProvenanceError("MODEL_METADATA_TOO_LARGE")
            return json.loads(content.decode("utf-8"))
    except ProvenanceError:
        raise
    except (urllib.error.URLError, TimeoutError, OSError, UnicodeError, ValueError):
        raise ProvenanceError("MODEL_METADATA_QUERY_FAILED") from None


def validate_metadata(info: object, tree: object) -> dict:
    if not isinstance(info, dict):
        raise ProvenanceError("MODEL_INFO_NOT_OBJECT")
    sha = info.get("sha")
    if not isinstance(sha, str) or REVISION_PATTERN.fullmatch(sha) is None:
        raise ProvenanceError("MODEL_REVISION_UNTRUSTED")
    if (info.get("id") != MODEL or info.get("private") is not False
            or info.get("gated") not in {False, "false"}):
        raise ProvenanceError("MODEL_PUBLIC_IDENTITY_NOT_VERIFIED")
    card = info.get("cardData")
    if not isinstance(card, dict) or card.get("license") != "apache-2.0":
        raise ProvenanceError("MODEL_CARD_LICENSE_MISMATCH")
    if info.get("pipeline_tag") != "sentence-similarity":
        raise ProvenanceError("MODEL_TASK_MISMATCH")
    sf = info.get("safetensors")
    if not isinstance(sf, dict):
        raise ProvenanceError("MODEL_SAFETENSORS_PARAMETER_METADATA_MISSING")
    parameter_count = sf.get("total")
    if type(parameter_count) is not int or not 1_000_000 <= parameter_count <= 200_000_000:
        raise ProvenanceError("MODEL_PARAMETER_COUNT_INVALID")
    if not isinstance(tree, list) or len(tree) > 1000:
        raise ProvenanceError("PINNED_MODEL_TREE_UNAVAILABLE")
    found: dict[str, dict] = {}
    for entry in tree:
        if not isinstance(entry, dict):
            raise ProvenanceError("PINNED_MODEL_TREE_MALFORMED")
        name = entry.get("path")
        if name not in FILES:
            continue
        if name in found or entry.get("type") != "file":
            raise ProvenanceError("MODEL_DUPLICATE_OR_INVALID_FILE")
        size = entry.get("size")
        if type(size) is not int or not 1 <= size <= 1_000_000_000:
            raise ProvenanceError("MODEL_FILE_SIZE_UNTRUSTED")
        lfs = entry.get("lfs")
        digest = None
        if lfs is not None:
            if not isinstance(lfs, dict):
                raise ProvenanceError("MODEL_LFS_METADATA_INVALID")
            oid = lfs.get("oid")
            if oid is not None and isinstance(oid, str) and SHA256_PATTERN.fullmatch(oid):
                digest = oid
        found[name] = {
            "size_bytes_metadata": size,
            "lfs_sha256_metadata": digest,
        }
    if set(found) != set(FILES):
        raise ProvenanceError("REQUIRED_MODEL_FILES_NOT_FOUND_IN_PINNED_REVISION")
    if found["model.safetensors"]["size_bytes_metadata"] < 10_000_000:
        raise ProvenanceError("MODEL_WEIGHT_FILE_UNEXPECTEDLY_SMALL")
    body = {
        "schema_version": 1,
        "model_id": MODEL,
        "revision_sha": sha,
        "revision_url": f"https://huggingface.co/{MODEL}/tree/{sha}",
        "license_declared_by_model_card": "apache-2.0",
        "license_full_text_audit_complete": False,
        "pipeline_tag": "sentence-similarity",
        "model_parameter_count_from_hub": parameter_count,
        "required_files": {k: found[k] for k in FILES},
        "theoretical_parameter_storage_bytes_fp32": parameter_count * 4,
        "theoretical_parameter_storage_bytes_fp16": parameter_count * 2,
        "theoretical_storage_is_not_peak_vram": True,
        "gpu_peak_vram_bytes_measured": None,
        "kaggle_gpu_fit_verified": False,
        "kaggle_gpu_time_consumed": False,
        "model_or_tokenizer_content_downloaded": False,
        "datasets_processed": False,
        "medical_evaluation_performed": False,
        "model_loading_approved": False,
        "remaining_required_controls": [
            "Inspect full model license and attribution notice obligations",
            "Verify model and tokenizer byte-level SHA-256 independently",
            "Review upstream training data and downstream model redistribution rights",
            "Stage pinned safe model files privately with an approved route",
            "Check actual Kaggle GPU available VRAM and remaining quota",
            "Get separate approval for isolated GPU feasibility inference",
        ],
    }
    return body


def inspect() -> dict:
    info = request_json(ROOT + "/revision/main")
    if not isinstance(info, dict):
        raise ProvenanceError("MODEL_INFO_NOT_OBJECT")
    sha = info.get("sha")
    if not isinstance(sha, str) or REVISION_PATTERN.fullmatch(sha) is None:
        raise ProvenanceError("MODEL_REVISION_UNTRUSTED")
    tree = request_json(ROOT + "/tree/" + sha + "?recursive=true")
    return validate_metadata(info, tree)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path,
        default=Path("build/model-provenance/all-minilm-l6-v2.json"),
    )
    args = parser.parse_args()
    try:
        data = inspect()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception:
        print("MODEL_PROVENANCE: FAIL_CLOSED_NO_MODEL_LOADING")
        return 1
    print("MODEL_PROVENANCE: PINNED_REVISION " + data["revision_sha"])
    print("MODEL_PROVENANCE: LICENSE_DECLARED " + data["license_declared_by_model_card"])
    print("MODEL_PROVENANCE: HUB_PARAMETER_COUNT " + str(data["model_parameter_count_from_hub"]))
    print("MODEL_PROVENANCE: SAFETENSORS_FILE_BYTES " +
          str(data["required_files"]["model.safetensors"]["size_bytes_metadata"]))
    print("MODEL_PROVENANCE: WEIGHTS_LFS_SHA256_METADATA " +
          str(data["required_files"]["model.safetensors"]["lfs_sha256_metadata"] or "NOT_PROVIDED"))
    print("MODEL_PROVENANCE: PARAM_STORAGE_FP32_BYTES " +
          str(data["theoretical_parameter_storage_bytes_fp32"]))
    print("MODEL_PROVENANCE: PARAM_STORAGE_FP16_BYTES " +
          str(data["theoretical_parameter_storage_bytes_fp16"]))
    print("MODEL_PROVENANCE: ACTUAL_PEAK_GPU_VRAM_NOT_MEASURED")
    print("MODEL_PROVENANCE: NO_WEIGHTS_DOWNLOADED_NO_GPU_CONSUMED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
