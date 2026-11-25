"""E1 candidate *design*, not approved scientific labels or leakage-free gold splits.

Only call inside an approved PRIVATE data environment. No network/downloads.
Returns a sanitized aggregate and in-memory row-index assignment; only the
aggregate is permitted for publication after independent review.

This planner deliberately quarantines anomalies and conservatively groups by
normalized question and exact Knowledge context. These proxies CANNOT prove
original PubMedQA source/lineage isolation or absence of semantic near-duplicates.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from .medhallu_native_audit import (
    CONFIGS, DATASET, NativeAuditError, _norm, _sha, audit_native_rows,
)

SPLITS = ("train", "dev", "test")
SEED = 20261125
MAX_SUMMARY_BYTES = 10_240


def _hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def candidate_plan(rows_by_config: dict[str, list[dict]], seed: int = SEED) -> tuple[dict, dict]:
    """Deterministically partition *eligible* parent question rows by groups.

    The returned index map is PRIVATE, remains in memory, and contains no text.
    A reference and a hallucinated answer always share their parent split.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise NativeAuditError("SEED_NOT_VALID")
    audit = audit_native_rows(rows_by_config)  # schema, length and text checks
    rows = []
    q_counts = Counter()
    for config in CONFIGS:
        for index, row in enumerate(rows_by_config[config]):
            q = _sha(row["Question"])
            k = _norm(" ".join(row["Knowledge"]))
            g = _sha(row["Ground Truth"])
            h = _sha(row["Hallucinated Answer"])
            rows.append((config, index, q, _hash(k) if k else None, g == h))
            q_counts[q] += 1

    # Quarantine the entire repeated-question group, rather than selecting an
    # arbitrary duplicate that might leak into an evaluation partition.
    eligible = []
    excluded = Counter()
    for item in rows:
        if item[4]:
            excluded["same_answer_role_collision"] += 1
        elif q_counts[item[2]] > 1:
            excluded["repeated_question_group"] += 1
        else:
            eligible.append(item)

    if len(eligible) < 3:
        raise NativeAuditError("TOO_FEW_ELIGIBLE_ROWS_FOR_SPLIT_DESIGN")

    parent = list(range(len(eligible)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def merge(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # Exact matching Knowledge may connect otherwise different questions from
    # the same paper, but this is NOT original PMID/PubMedQA lineage validation.
    seen_question = {}
    seen_context = {}
    for position, item in enumerate(eligible):
        for key, cache in ((item[2], seen_question), (item[3], seen_context)):
            if key is None:
                continue
            if key in cache:
                merge(position, cache[key])
            else:
                cache[key] = position

    clusters = defaultdict(list)
    for position, item in enumerate(eligible):
        clusters[find(position)].append(item)
    if len(clusters) < 3:
        raise NativeAuditError("INSUFFICIENT_INDEPENDENT_PROXY_GROUPS")

    # Order based on stable private question fingerprints, then allocate
    # 80/10/10 by group count; not exact row counts if group sizes differ.
    groups = sorted(
        clusters.values(),
        key=lambda values: (
            _hash(str(seed) + "|" + min(x[2] for x in values)),
            min(x[2] for x in values),
        ),
    )
    n = len(groups)
    n_train = min(n - 2, max(1, round(n * 0.8)))
    n_dev = min(n - n_train - 1, max(1, round(n * 0.1)))
    n_test = n - n_train - n_dev
    if min(n_train, n_dev, n_test) < 1:
        raise NativeAuditError("EMPTY_SPLIT_GROUPS")
    quotas = (n_train, n_dev, n_test)
    assignment = {}
    split_rows = Counter()
    split_groups = Counter()
    offset = 0
    for split, quota in zip(SPLITS, quotas):
        for group in groups[offset:offset + quota]:
            split_groups[split] += 1
            for config, index, _q, _k, _collision in group:
                assignment[(config, index)] = split
                split_rows[split] += 1
        offset += quota
    if len(assignment) != len(eligible):
        raise NativeAuditError("PRIVATE_ASSIGNMENT_INCOMPLETE")

    # Digest reproducibly identifies an index-only mapping, not a publication
    # of row IDs, a proof of source lineage or an authenticated file signature.
    digest_body = [[c, i, assignment[(c, i)]] for c, i in sorted(assignment)]
    assignment_digest = _hash(
        json.dumps({"seed": seed, "assignment": digest_body}, separators=(",", ":"))
    )
    summary = {
        "schema_version": 1,
        "audit_kind": "medhallu_candidate_binary_and_proxy_split_design",
        "dataset_id": DATASET,
        "seed": seed,
        "source_config_rows": {c: audit["config_rows"][c] for c in CONFIGS},
        "quarantined_by_reason": dict(sorted(excluded.items())),
        "quarantined_total_rows": len(rows) - len(eligible),
        "eligible_parent_rows": len(eligible),
        "candidate_answer_pair_examples": 2 * len(eligible),
        "candidate_role_0_ground_truth_examples": len(eligible),
        "candidate_role_1_hallucinated_examples": len(eligible),
        "candidate_split_parent_rows": {s: split_rows[s] for s in SPLITS},
        "candidate_split_answer_examples": {s: 2 * split_rows[s] for s in SPLITS},
        "candidate_split_proxy_groups": {s: split_groups[s] for s in SPLITS},
        "candidate_assignment_sha256": assignment_digest,
        "native_category_to_claim_label_mapping_approved": False,
        "binary_role_labels_scientifically_adjudicated": False,
        "pubmedqa_lineage_verified": False,
        "semantic_near_duplicate_leakage_excluded": False,
        "final_train_dev_test_approved": False,
        "scientific_experiments_executed": False,
        "raw_text_or_ids_exported": False,
        "status": "CANDIDATE_ONLY_BLOCKED_PENDING_LABEL_AND_LINEAGE_REVIEW",
    }
    verify_summary(summary)
    return summary, assignment


def verify_summary(r: dict) -> None:
    if not isinstance(r, dict) or r.get("audit_kind") != "medhallu_candidate_binary_and_proxy_split_design":
        raise NativeAuditError("INVALID_CANDIDATE_SUMMARY_TYPE")
    permitted = {
        "schema_version", "audit_kind", "dataset_id", "seed", "source_config_rows",
        "quarantined_by_reason", "quarantined_total_rows", "eligible_parent_rows",
        "candidate_answer_pair_examples", "candidate_role_0_ground_truth_examples",
        "candidate_role_1_hallucinated_examples", "candidate_split_parent_rows",
        "candidate_split_answer_examples", "candidate_split_proxy_groups",
        "candidate_assignment_sha256", "native_category_to_claim_label_mapping_approved",
        "binary_role_labels_scientifically_adjudicated", "pubmedqa_lineage_verified",
        "semantic_near_duplicate_leakage_excluded", "final_train_dev_test_approved",
        "scientific_experiments_executed", "raw_text_or_ids_exported", "status",
    }
    if set(r) != permitted or r.get("schema_version") != 1 or r.get("dataset_id") != DATASET:
        raise NativeAuditError("SUMMARY_SCHEMA_NOT_ALLOWED")
    if r.get("status") != "CANDIDATE_ONLY_BLOCKED_PENDING_LABEL_AND_LINEAGE_REVIEW":
        raise NativeAuditError("FALSE_READY_CLAIM")
    blocked_flags = (
        "native_category_to_claim_label_mapping_approved",
        "binary_role_labels_scientifically_adjudicated",
        "pubmedqa_lineage_verified", "semantic_near_duplicate_leakage_excluded",
        "final_train_dev_test_approved", "scientific_experiments_executed",
        "raw_text_or_ids_exported",
    )
    if any(r.get(k) is not False for k in blocked_flags):
        raise NativeAuditError("FALSE_SCIENTIFIC_APPROVAL")
    if type(r.get("seed")) is not int or not 0 <= r["seed"] < 2**32:
        raise NativeAuditError("SEED_NOT_VALID")
    if not isinstance(r.get("candidate_assignment_sha256"), str) or len(r["candidate_assignment_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in r["candidate_assignment_sha256"]):
        raise NativeAuditError("ASSIGNMENT_HASH_INVALID")
    maps = (
        ("source_config_rows", CONFIGS),
        ("candidate_split_parent_rows", SPLITS),
        ("candidate_split_answer_examples", SPLITS),
        ("candidate_split_proxy_groups", SPLITS),
    )
    for name, keys in maps:
        m = r.get(name)
        if not isinstance(m, dict) or set(m) != set(keys) or any(type(v) is not int or v < 0 or v > 20_000 for v in m.values()):
            raise NativeAuditError("SPLIT_COUNTER_MAP_INVALID")
    excluded = r.get("quarantined_by_reason")
    if not isinstance(excluded, dict) or not set(excluded).issubset({"same_answer_role_collision", "repeated_question_group"}):
        raise NativeAuditError("QUARANTINE_SCHEMA_INVALID")
    if any(type(v) is not int or v < 0 for v in excluded.values()):
        raise NativeAuditError("QUARANTINE_COUNTER_INVALID")
    for name in ("quarantined_total_rows", "eligible_parent_rows", "candidate_answer_pair_examples", "candidate_role_0_ground_truth_examples", "candidate_role_1_hallucinated_examples"):
        if type(r.get(name)) is not int or r[name] < 0 or r[name] > 40_000:
            raise NativeAuditError("ROW_COUNTER_INVALID")
    total = sum(r["source_config_rows"].values())
    n = r["eligible_parent_rows"]
    if (total != n + r["quarantined_total_rows"] or
        r["quarantined_total_rows"] != sum(excluded.values()) or
        n != sum(r["candidate_split_parent_rows"].values()) or
        n != r["candidate_role_0_ground_truth_examples"] or
        n != r["candidate_role_1_hallucinated_examples"] or
        2*n != r["candidate_answer_pair_examples"] or
        any(r["candidate_split_answer_examples"][s] != 2*r["candidate_split_parent_rows"][s] for s in SPLITS) or
        min(r["candidate_split_proxy_groups"].values()) < 1):
        raise NativeAuditError("COUNTERS_NOT_CONSISTENT")
    if len(json.dumps(r, sort_keys=True).encode("utf-8")) > MAX_SUMMARY_BYTES:
        raise NativeAuditError("SUMMARY_TOO_LARGE")


def write_safe_summary(summary: dict, output_path: Path) -> None:
    verify_summary(summary)
    if output_path.exists() or output_path.is_symlink() or (output_path.parent.exists() and output_path.parent.is_symlink()):
        raise NativeAuditError("SUMMARY_OVERWRITE_OR_SYMLINK")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
