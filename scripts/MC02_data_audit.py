"""MC02: audit already-normalized, authorized JSONL inside its permitted environment."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from research_data.audit import audit_jsonl

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dataset", choices=("medhallu", "bioasq14b"), required=True)
    p.add_argument("--near-threshold", type=float, default=0.82)
    p.add_argument("--max-candidate-pairs", type=int, default=2_000_000)
    a = p.parse_args()
    report = audit_jsonl(a.input, a.out, a.dataset, a.near_threshold, a.max_candidate_pairs)
    print("MC02_LEAKAGE_AUDIT_" + ("PASSED" if report["ready_for_frozen_split"] else "BLOCKED"))
    if not report["ready_for_frozen_split"]:
        raise SystemExit(2)
