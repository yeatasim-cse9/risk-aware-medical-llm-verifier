"""One-cell Kaggle hybrid: offline-reviewed E1 label/lineage *screen*, not E1 training.

Run only on a PRIVATE Kaggle CPU notebook with Internet ON and cleared
MedHallu/PubMedQA cloud-processing rights. The bootstrap cell pins this script
to a reviewed Git commit and byte SHA256. No medical text, original dataset
file, question fingerprint, or private row-level assignment is persisted in
/kaggle/working or sent to GitHub/Telegram.

Real PubMedQA PMID/article identity is NOT present in native MedHallu columns:
this screen can never independently authorize final train/dev/test splits.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import urllib.request

REPO = "yeatasim-cse9/risk-aware-medical-llm-verifier"
DATASET = "UTAustin-AIHealth/MedHallu"
REVISION = "515060458a945c633debc6fd5baac7764416b724"
SEED = 20261125
EXPECTED_ASSIGNMENT_SHA = "ddd32f57ebc49ec6591cffd2d953a869c30518525bf66d6e2fb931a68bb0945f"
EXPECTED_FILES = {
    "pqa_labeled": ("pqa_labeled/train-00000-of-00001.parquet",
                    1116475, "ac963f9aaeb163f494edd4dbfec04086c3e35c7cc55e3f9c8b1d7b31188fa34c", 1000),
    "pqa_artificial": ("pqa_artificial/train-00000-of-00001.parquet",
                       10283429, "70de8ce7f89a38314fa231a12e45168f3ee1aa591e8a38f6556b89aa8b1ca513", 9000),
}
REPORT_NAME = "e1_hybrid_label_lineage_screen_v1.json"
MAX_REPORT_BYTES = 12000
STAGE = "BOOT"


class SafeAuditError(ValueError):
    """Only fixed in-repository uppercase codes should be passed here."""


def stage(name: str) -> None:
    global STAGE
    STAGE = name
    print("E1_HYBRID_STAGE=" + name, flush=True)


def hash_file(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _dependencies():
    try:
        from huggingface_hub import snapshot_download
        import pyarrow.parquet as pq
        return snapshot_download, pq
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-input",
             "--disable-pip-version-check", "--quiet",
             "huggingface_hub==0.36.0", "pyarrow==21.0.0"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=240,
        )
        from huggingface_hub import snapshot_download
        import pyarrow.parquet as pq
        return snapshot_download, pq


def _reviewed_modules(git_commit: str, root: Path):
    """Import two exact repository modules from the SAME Git commit as this code."""
    if re.fullmatch(r"[0-9a-f]{40}", git_commit) is None:
        raise SafeAuditError("SOURCE_GIT_COMMIT_NOT_PINNED")
    package = root / "research_data"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    module_hashes = {}
    for module in ("medhallu_native_audit", "e1_candidate_splits"):
        name = module + ".py"
        url = f"https://raw.githubusercontent.com/{REPO}/{git_commit}/src/research_data/{name}"
        request = urllib.request.Request(url, headers={"User-Agent": "e1-hybrid-private-audit"})
        with urllib.request.urlopen(request, timeout=30) as response:
            source = response.read(70_000)
        if not (1000 < len(source) < 60_000):
            raise SafeAuditError("PINNED_REVIEWED_MODULE_SIZE_INVALID")
        compile(source, name, "exec")
        module_hashes[module] = hashlib.sha256(source).hexdigest()
        (package / name).write_bytes(source)
    sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    native = importlib.import_module("research_data.medhallu_native_audit")
    candidate = importlib.import_module("research_data.e1_candidate_splits")
    if candidate.SEED != SEED or native.DATASET != DATASET:
        raise SafeAuditError("REVIEWED_MODULE_PROTOCOL_CHANGED")
    return native, candidate, module_hashes


def screen_near_duplicate_questions(rows: dict, assignment: dict,
                                    normalizer, candidate_limit: int = 150_000) -> dict:
    """Bounded rare 5-word-gram candidate screen; NOT exhaustive/semantic search."""
    if type(candidate_limit) is not int or candidate_limit < 1:
        raise SafeAuditError("INVALID_CANDIDATE_LIMIT")
    entries = []
    for config in sorted(rows):
        for row_index, row in enumerate(rows[config]):
            key = (config, row_index)
            if key not in assignment:
                continue
            words = normalizer(row["Question"]).split()
            fivegrams = set(tuple(words[i:i + 5]) for i in range(max(0, len(words) - 4)))
            trigrams = set(tuple(words[i:i + 3]) for i in range(max(0, len(words) - 2)))
            entries.append((assignment[key], fivegrams, trigrams))
    postings = defaultdict(list)
    for index, (_split, grams, _trigrams) in enumerate(entries):
        for gram in grams:
            postings[gram].append(index)
    compared_pairs = set()
    high_similarity = 0
    skipped_common_grams = 0
    cap_reached = False
    for gram in sorted(postings):
        bucket = postings[gram]
        if len(bucket) > 40:
            skipped_common_grams += 1
            continue
        for left_pos in range(len(bucket)):
            a = bucket[left_pos]
            for right_pos in range(left_pos + 1, len(bucket)):
                b = bucket[right_pos]
                if entries[a][0] == entries[b][0]:
                    continue
                pair = (min(a, b), max(a, b))
                if pair in compared_pairs:
                    continue
                if len(compared_pairs) >= candidate_limit:
                    cap_reached = True
                    break
                compared_pairs.add(pair)
                a3, b3 = entries[a][2], entries[b][2]
                union = len(a3 | b3)
                if union and len(a3 & b3) / union >= 0.80:
                    high_similarity += 1
            if cap_reached:
                break
        if cap_reached:
            break
    return {
        "eligible_parent_rows_screened": len(entries),
        "eligible_questions_too_short_for_5_word_screen": sum(not e[1] for e in entries),
        "rare_5_word_grams_common_buckets_skipped": skipped_common_grams,
        "cross_partition_candidate_pairs_compared": len(compared_pairs),
        "cross_partition_high_token_trigram_overlap_pairs": high_similarity,
        "candidate_limit_reached": cap_reached,
        "complete_semantic_duplicate_search": False,
        "pubmedqa_article_lineage_verified": False,
    }


def audit_rows(rows: dict, native, candidate) -> tuple[dict, dict]:
    previous = native.audit_native_rows(rows)
    for code, expected in {
        "unique_questions_total": 9999,
        "questions_repeated_within_or_across_configs": 1,
        "questions_shared_across_configs": 0,
        "ground_truth_equals_hallucinated_answer_pairs": 1,
    }.items():
        if previous.get(code) != expected:
            raise SafeAuditError("PREVIOUS_VERIFIED_AGGREGATE_NOT_REPRODUCED")
    summary, assignment = candidate.candidate_plan(rows, seed=SEED)
    if (summary["candidate_assignment_sha256"] != EXPECTED_ASSIGNMENT_SHA or
            summary["eligible_parent_rows"] != 9997 or
            summary["candidate_split_parent_rows"] != {"train": 7998, "dev": 1000, "test": 999}):
        raise SafeAuditError("CANDIDATE_SPLIT_ASSIGNMENT_CHANGED")
    category_counts = {}
    difficulty_counts = {}
    for config in sorted(rows):
        category_counts[config] = Counter(row["Category of Hallucination"].strip()
                                          for row in rows[config])
        difficulty_counts[config] = Counter(row["Difficulty Level"].strip()
                                            for row in rows[config])
    screen = screen_near_duplicate_questions(rows, assignment, native._norm)
    public_stats = {
        "native_category_unique_value_count": {c: len(category_counts[c]) for c in sorted(rows)},
        "native_difficulty_unique_value_count": {c: len(difficulty_counts[c]) for c in sorted(rows)},
        "category_frequencies_without_names": {
            c: sorted(category_counts[c].values(), reverse=True) for c in sorted(rows)
        },
        "difficulty_frequencies_without_names": {
            c: sorted(difficulty_counts[c].values(), reverse=True) for c in sorted(rows)
        },
        "private_category_names_exported": False,
        "source_pmid_field_present_in_native_columns": False,
        "native_label_to_claim_gold_mapping_approved": False,
    }
    # Private notebook output ONLY: display strictly allowlisted short category
    # names, not medical records or free-text answers. Never export these names
    # to GitHub, Telegram, or the aggregate JSON.
    print("PRIVATE_KAGGLE_CATEGORY_LABEL_REVIEW_BEGIN")
    for config in sorted(rows):
        for label, count in sorted(category_counts[config].items()):
            printable = label if re.fullmatch(r"[A-Za-z][A-Za-z0-9 _/-]{0,55}", label) else "[REDACTED]"
            print("CONFIG", config, "CATEGORY", printable, "COUNT", count)
    print("PRIVATE_KAGGLE_CATEGORY_LABEL_REVIEW_END")
    return {"candidate": summary, "labels": public_stats, "near_duplicate_screen": screen}, assignment


def main() -> None:
    commit = os.environ.get("E1_PINNED_GITHUB_COMMIT", "")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise SafeAuditError("MISSING_PINNED_GITHUB_COMMIT")
    stage("PRIVATE_KAGGLE_ENVIRONMENT")
    working = Path("/kaggle/working")
    temp_root = Path("/kaggle/temp")
    if not working.is_dir() or working.is_symlink() or temp_root.is_symlink():
        raise SafeAuditError("PRIVATE_KAGGLE_WORKING_UNAVAILABLE")
    temp_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not temp_root.is_dir() or not temp_root.resolve().is_relative_to(Path("/kaggle").resolve()):
        raise SafeAuditError("PRIVATE_KAGGLE_SCRATCH_UNAVAILABLE")
    output = working / REPORT_NAME
    if output.exists() or output.is_symlink():
        raise SafeAuditError("EXISTING_AUDIT_REPORT_DO_NOT_OVERWRITE")
    with tempfile.TemporaryDirectory(dir=str(temp_root), prefix="e1-hybrid-") as temp:
        base = Path(temp)
        os.environ["HF_HOME"] = str(base / "hf-cache")
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        stage("REVIEWED_GITHUB_MODULES")
        native, candidate, module_hashes = _reviewed_modules(commit, base / "src")
        stage("DEPENDENCIES")
        snapshot_download, parquet = _dependencies()
        stage("PINNED_DATASET")
        folder = Path(snapshot_download(
            repo_id=DATASET, repo_type="dataset", revision=REVISION,
            allow_patterns=["pqa_labeled/*.parquet", "pqa_artificial/*.parquet"],
            local_dir=str(base / "data"), max_workers=2, token=False,
        ))
        rows = {}
        file_evidence = {}
        total_bytes = 0
        stage("SOURCE_FILE_INTEGRITY")
        for config, (relative, expected_size, expected_hash, expected_count) in EXPECTED_FILES.items():
            target = folder / relative
            if (not target.is_file() or target.is_symlink() or
                    not target.resolve().is_relative_to(base.resolve())):
                raise SafeAuditError("EXPECTED_PARQUET_FILE_MISSING")
            size = target.stat().st_size
            total_bytes += size
            if size != expected_size or total_bytes > 60_000_000 or hash_file(target) != expected_hash:
                raise SafeAuditError("PINNED_DATASET_SIZE_OR_SHA256_MISMATCH")
            table = parquet.read_table(target)
            if table.num_rows != expected_count or set(table.column_names) != native.NATIVE_FIELDS:
                raise SafeAuditError("NATIVE_DATASET_SCHEMA_OR_ROW_COUNT_MISMATCH")
            rows[config] = table.to_pylist()
            del table
            file_evidence[config] = {"file_sha256": expected_hash, "row_count": expected_count}
        stage("PRIVATE_LABEL_AND_SPLIT_CHECK")
        findings, private_assignments = audit_rows(rows, native, candidate)
        if len(private_assignments) != 9997:
            raise SafeAuditError("PRIVATE_ASSIGNMENTS_MISSING")
        del rows, private_assignments
        stage("AGGREGATE_ONLY_REPORT")
        report = {
            "schema_version": 1,
            "audit_kind": "e1_hybrid_private_label_and_lineage_screen_v1",
            "github_source_commit": commit,
            "dataset_id": DATASET,
            "dataset_revision": REVISION,
            "candidate_seed": SEED,
            "source_files": file_evidence,
            "reviewed_module_sha256": module_hashes,
            "findings": findings,
            "final_lineage_verified": False,
            "final_train_dev_test_approved": False,
            "scientific_model_experiment_executed": False,
            "raw_medical_text_or_source_ids_exported": False,
        }
        report["integrity_sha256"] = hashlib.sha256(canonical(report)).hexdigest()
        raw = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if len(raw.encode("utf-8")) > MAX_REPORT_BYTES:
            raise SafeAuditError("AGGREGATE_REPORT_SIZE_EXCEEDED")
        output.write_text(raw, encoding="utf-8")
        print("E1_HYBRID_AUDIT_COMPLETE_PRIVATE_CPU_ONLY")
        print("E1_HYBRID_REPORT_PATH=/kaggle/working/" + REPORT_NAME)
        print("E1_HYBRID_NEAR_DUPLICATE_FLAGS=" +
              str(findings["near_duplicate_screen"]["cross_partition_high_token_trigram_overlap_pairs"]))
        print("E1_HYBRID_LINEAGE_AND_FINAL_E1_STATUS=NOT_APPROVED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        code = str(exc) if isinstance(exc, SafeAuditError) else ""
        if not (code and len(code) <= 64 and code == code.upper() and
                code.replace("_", "").isalnum()):
            code = "SAFE_FAILURE_REVIEW_REQUIRED"
        print("E1_HYBRID_FAILED_STAGE=" + STAGE, file=sys.stderr)
        print("E1_HYBRID_FAILED_CODE=" + code, file=sys.stderr)
        raise SystemExit(1)
