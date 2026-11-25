"""Manual-only private Kaggle E1 source-group remediation, not model training.

Retains NO private row assignments, article IDs or original questions in output.
Requires one SHA256-pinned loader cell pointing to the reviewed merged Git SHA.
Every source and original test set is re-downloaded privately at immutable hashes.
"""
from __future__ import annotations

from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import types
import urllib.request

REPO = "yeatasim-cse9/risk-aware-medical-llm-verifier"
MODULE = "src/research_data/e1_pubmedqa_test_exclusion.py"
HELPER = "scripts/e1_pubmedqa_lineage_kaggle.py"
OUTPUT_NAME = "e1_official_test_exclusion_candidate_splits_v1.json"
MAX_JSON_SIZE = 18000
STEP = "BOOT"


class SafeRemediationError(ValueError):
    """Fixed safe codes only."""


def stage(s):
    global STEP
    STEP = s
    print("E1_REMEDIATION_STAGE=" + s, flush=True)


def _read_reviewed(commit: str, path: str) -> bytes:
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise SafeRemediationError("EXACT_GIT_SHA_REQUIRED")
    url = f"https://raw.githubusercontent.com/{REPO}/{commit}/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "private-kaggle-e1-remediation"})
    with urllib.request.urlopen(req, timeout=45) as response:
        data = response.read(95_000)
    if not 1000 < len(data) < 90_000:
        raise SafeRemediationError("REVIEWED_SOURCE_BYTE_LENGTH_OUT_OF_POLICY")
    compile(data, path, "exec")
    return data


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def main():
    commit = os.environ.get("E1_REMEDIATION_PINNED_COMMIT", "")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise SafeRemediationError("PINNED_REVIEWED_GIT_SHA_MISSING")
    stage("PRIVATE_CPU_ENVIRONMENT")
    scratch = Path("/kaggle/temp")
    working = Path("/kaggle/working")
    if not working.is_dir() or working.is_symlink() or scratch.is_symlink():
        raise SafeRemediationError("PRIVATE_KAGGLE_DIRECTORIES_REQUIRED")
    scratch.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not scratch.is_dir() or not scratch.resolve().is_relative_to(Path("/kaggle").resolve()):
        raise SafeRemediationError("KAGGLE_TEMP_NOT_ALLOWED")
    report_path = working / OUTPUT_NAME
    if report_path.exists() or report_path.is_symlink():
        raise SafeRemediationError("DO_NOT_OVERWRITE_PREVIOUS_REPORT")

    with tempfile.TemporaryDirectory(dir=str(scratch), prefix="e1-source-remediation-") as folder:
        root = Path(folder)
        os.environ["HF_HOME"] = str(root / "hf-cache")
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        stage("REVIEWED_EXACT_GIT_SOURCE")
        helper_source = _read_reviewed(commit, HELPER)
        helper = types.ModuleType("pinned_e1_pubmedqa_lineage")
        helper.__file__ = str(root / "pinned_lineage_helper.py")
        helper.__package__ = ""
        exec(compile(helper_source, HELPER, "exec"), helper.__dict__)
        native, candidate, advanced, old_hashes = helper.checked_modules(commit, root / "src")
        hard_source = _read_reviewed(commit, MODULE)
        package = root / "src/research_data"
        (package / "e1_pubmedqa_test_exclusion.py").write_bytes(hard_source)
        importlib.invalidate_caches()
        hard = importlib.import_module("research_data.e1_pubmedqa_test_exclusion")
        if hard.SEED != 20261125:
            raise SafeRemediationError("SOURCE_REMEDIATION_SEED_CHANGED")
        module_hashes = dict(old_hashes)
        module_hashes["e1_pubmedqa_test_exclusion"] = sha256(hard_source).hexdigest()
        module_hashes["e1_pubmedqa_lineage_kaggle_helper"] = sha256(helper_source).hexdigest()

        stage("DOWNLOAD_PINNED_SOURCE_FILES")
        snapshot_download, parquet = helper.libraries()
        med = helper.checked_parquet(
            snapshot_download, parquet, helper.MEDHALLU,
            helper.MEDHALLU_REVISION, helper.SOURCE_SHA256, root / "medhallu",
            is_source=False,
        )
        prior, assignment = candidate.candidate_plan(med)
        if (prior["candidate_assignment_sha256"] != helper.CANDIDATE_ASSIGNMENT_SHA256 or
                prior["eligible_parent_rows"] != 9997):
            raise SafeRemediationError("ORIGINAL_CANDIDATE_DIGEST_CHANGED")
        pub = helper.checked_parquet(
            snapshot_download, parquet, helper.PUBMEDQA,
            helper.PUBMEDQA_REVISION, helper.PUBMEDQA_SHA256, root / "pubmedqa",
            is_source=True,
        )
        official_ids, official_hash = helper.official_test_ids()
        if official_hash != "939fe566f09017d13b1ca64d2ddfee0bc2374b366048152997669cccedc44d51":
            raise SafeRemediationError("ORIGINAL_500_TEST_SHA_CHANGED")
        source_index = advanced.pubmed_index(pub)

        stage("REPRODUCE_PREVIOUS_SOURCE_RISK_COUNTS")
        old_findings = advanced.audit_lineage(med, assignment, source_index, official_ids)
        advanced.validate_report(old_findings)
        if (old_findings["ambiguous_multiple_pubids_parent_rows"] != 23 or
                old_findings["matched_unique_pubid_parent_rows"] != 9974 or
                old_findings["source_pubids_spanning_candidate_partitions"] != 0 or
                old_findings["official_pubmedqa_500_test_overlap_parent_rows_by_candidate_split"]
                != {"train": 401, "dev": 61, "test": 38}):
            raise SafeRemediationError("PREVIOUS_SOURCE_LINEAGE_AGGREGATES_CHANGED")

        stage("EXCLUDE_OFFICIAL_TEST_AND_UNRESOLVED_PMIDS")
        summary, new_assignment = hard.candidate_remediation(
            med, assignment, source_index, official_ids, seed=20261125,
        )
        hard.verify_summary(summary)
        if (summary["source_reviewed_candidate_parent_records"] != 9474 or
                summary["excluded_parent_records_by_reason"]["original_official_500_test_heldout"] != 500 or
                summary["excluded_parent_records_by_reason"]["multiple_pubmedqa_pubids"] != 23 or
                not summary["original_official_500_test_removed_from_all_new_candidate_splits"]):
            raise SafeRemediationError("CRITICAL_TEST_OR_AMBIGUITY_GATE_NOT_PASSED")

        del med, pub, source_index, official_ids, assignment, new_assignment
        stage("SANITIZED_AGGREGATE_ONLY")
        report = {
            "schema_version": 1,
            "audit_kind": "e1_pubmedqa_original_500_test_excluded_private_candidate_v1",
            "source_git_commit": commit,
            "medhallu_revision": helper.MEDHALLU_REVISION,
            "pubmedqa_revision": helper.PUBMEDQA_REVISION,
            "original_500_test_sha256": official_hash,
            "reviewed_modules_sha256": module_hashes,
            "medhallu_source_sha256": {k:v[2] for k,v in helper.SOURCE_SHA256.items()},
            "pubmedqa_source_sha256": {k:v[2] for k,v in helper.PUBMEDQA_SHA256.items()},
            "original_candidate_assignment_sha256": helper.CANDIDATE_ASSIGNMENT_SHA256,
            "summary": summary,
            "source_lineage_fully_certified": False,
            "final_split_approved": False,
            "e1_scientific_experiment_executed": False,
            "medical_text_pubids_private_assignments_exported": False,
        }
        report["integrity_sha256"] = sha256(_canonical(report)).hexdigest()
        payload = json.dumps(report, sort_keys=True, indent=2) + "\n"
        if len(payload.encode("utf-8")) > MAX_JSON_SIZE:
            raise SafeRemediationError("SANITIZED_AGGREGATE_SIZE_EXCEEDED")
        report_path.write_text(payload, encoding="utf-8")
        print("E1_REMEDIATION_SAFE_AGGREGATE_SAVED")
        print("E1_REMEDIATION_PREVIOUS_500_TEST_EXCLUDED=" + str(
            summary["excluded_parent_records_by_reason"]["original_official_500_test_heldout"]))
        print("E1_REMEDIATION_AMBIGUOUS_PMIDS_QUARANTINED=" + str(
            summary["excluded_parent_records_by_reason"]["multiple_pubmedqa_pubids"]))
        print("E1_REMEDIATION_REMAINING_PARENTS=" + str(
            summary["source_reviewed_candidate_parent_records"]))
        print("E1_REMEDIATION_FINAL_E1_STATUS=BLOCKED_PENDING_SCIENTIFIC_APPROVAL")
        print("E1_REMEDIATION_OUTPUT_PATH=/kaggle/working/" + OUTPUT_NAME)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        code = str(exc) if isinstance(exc, SafeRemediationError) else ""
        if not (code and len(code) <= 64 and code == code.upper() and code.replace("_", "").isalnum()):
            code = "SAFE_FAILURE_REQUIRES_REVIEW"
        print("E1_REMEDIATION_FAILED_STAGE=" + STEP, file=sys.stderr)
        print("E1_REMEDIATION_FAILED_CODE=" + code, file=sys.stderr)
        raise SystemExit(1)
