"""MC02: privacy-preserving, deterministic exact/group/near split leakage audit.

Input is already NORMALIZED JSONL, not benchmark-native records. Never
serialize medical text, IDs, or pairwise matches into the report.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def normalize(text: str) -> str:
    return re.sub(r"[^\w]+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def fingerprint(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def shingles(text: str) -> frozenset[str]:
    words = normalize(text).split()
    if not words:
        return frozenset()
    word_grams = {"w:" + " ".join(words[i:i+3]) for i in range(max(1, len(words)-2))}
    chars = " ".join(words)
    char_grams = {"c:" + chars[i:i+5] for i in range(max(1, len(chars)-4))}
    return frozenset(word_grams | char_grams)


def read_normalized_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Non-object JSONL record at line " + str(line_number))
            rows.append(row)
    return rows


def audit_rows(rows: list[dict], near_threshold: float = 0.82,
               max_candidate_pairs: int = 2_000_000,
               group_fields: tuple[str, ...] = ("question_id", "source_id", "group_id")) -> dict:
    if not 0 < near_threshold <= 1 or type(max_candidate_pairs) is not int or max_candidate_pairs < 1:
        raise ValueError("Invalid audit threshold or candidate limit")
    if not rows:
        raise ValueError("Cannot audit empty records")
    records = []
    unique = set()
    split_sizes = Counter()
    fields_present = Counter()
    for row in rows:
        if not isinstance(row, dict) or any(k not in row for k in ("id", "split", "text")):
            raise ValueError("Normalized rows require id, split and text")
        if any(not isinstance(row[k], str) or not row[k].strip() for k in ("id", "split", "text")):
            raise ValueError("Nonempty string id/split/text required")
        key = (row["split"], row["id"])
        if key in unique:
            raise ValueError("Duplicate sample ID within split")
        unique.add(key)
        if not normalize(row["text"]):
            raise ValueError("Empty normalized text")
        for field in group_fields:
            if field in row and row[field] is not None:
                if not isinstance(row[field], str):
                    raise ValueError("Group IDs must be strings (never auto-coerce)")
                if row[field].strip():
                    fields_present[field] += 1
        records.append({
            "split": row["split"], "hash": fingerprint(row["text"]),
            "shingles": shingles(row["text"]),
            "groups": {field: row[field] for field in group_fields if isinstance(row.get(field), str) and row[field].strip()},
        })
        split_sizes[row["split"]] += 1
    if len(split_sizes) < 2:
        raise ValueError("At least two distinct splits required for cross-split leakage audit")

    exact_index = defaultdict(set)
    group_index = defaultdict(set)
    grams_index = defaultdict(set)
    finding_counts = Counter()
    compared = 0
    cross_pairs = 0
    for i, row in enumerate(records):
        candidates = set(exact_index[row["hash"]])
        for field, value in row["groups"].items():
            candidates.update(group_index[(field, value)])
        for token in row["shingles"]:
            candidates.update(grams_index[token])
        for j in sorted(candidates):
            other = records[j]
            if other["split"] == row["split"]:
                continue
            compared += 1
            if compared > max_candidate_pairs:
                raise ValueError("Candidate limit exceeded: audit INCOMPLETE; raise limit or partition by source")
            if other["hash"] == row["hash"]:
                kind = "exact_text"
            elif any(value == other["groups"].get(field) for field, value in row["groups"].items()):
                kind = "shared_source_group"
            else:
                inter = len(row["shingles"] & other["shingles"])
                union = len(row["shingles"]) + len(other["shingles"]) - inter
                similarity = inter / union if union else 0.0
                kind = "near_text" if similarity >= near_threshold else None
            if kind:
                finding_counts[kind] += 1
                cross_pairs += 1
        exact_index[row["hash"]].add(i)
        for field, value in row["groups"].items():
            group_index[(field, value)].add(i)
        for token in row["shingles"]:
            grams_index[token].add(i)
    report = {
        "schema_version": 1,
        "audit_kind": "cross_split_privacy_preserving",
        "passed": cross_pairs == 0,
        "checked_rows": len(records),
        "split_sizes": dict(sorted(split_sizes.items())),
        "near_threshold": near_threshold,
        "near_candidate_pairs_compared": compared,
        "findings_by_type": {k: finding_counts.get(k, 0) for k in ("exact_text", "shared_source_group", "near_text")},
        "group_fields_observed": dict(sorted(fields_present.items())),
        "raw_text_or_identifiers_exported": False,
        "limitations": [
            "Near-duplicate search compares pairs sharing at least one word/character shingle",
            "Semantic paraphrases and external pretraining contamination are not ruled out",
            "Group overlap detection is only as complete as supplied lineage identifiers",
        ],
    }
    return report


def audit_jsonl(input_path: Path, output_path: Path, dataset: str,
                near_threshold: float = 0.82, max_candidate_pairs: int = 2_000_000) -> dict:
    if dataset.casefold().replace("-", "").startswith("medval"):
        raise PermissionError("MedVAL forbidden in cloud data audit. Use only approved restricted environment")
    if dataset not in ("medhallu", "bioasq14b"):
        raise PermissionError("Dataset not approved for this cloud-safe normalized audit")
    if output_path.resolve() == input_path.resolve():
        raise ValueError("Input cannot be overwritten")
    report = audit_rows(read_normalized_jsonl(input_path), near_threshold, max_candidate_pairs)
    report["dataset_role"] = "development" if dataset == "medhallu" else "external_test"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError("Cannot overwrite audit result")
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
