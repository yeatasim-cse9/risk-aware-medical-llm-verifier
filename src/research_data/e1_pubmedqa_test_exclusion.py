"""E1 provisional remediation: exclude official PubMedQA-500 and unsafe provenance.

Pure in-memory routine for an APPROVED private Kaggle environment. Does NOT
download datasets, infer medical correctness, expose PMIDs, produce scientific
gold labels, or certify a final train/dev/test partition.

Inputs are reviewed MedHallu rows, existing deterministic candidate assignments,
PubMedQA question->(pubid,context fingerprint) index and official original
PubMedQA 500 heldout PMID set. All input data remains private.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json

from .e1_advanced_lineage import CONFIGS, SPLITS, fingerprint
from .medhallu_native_audit import NativeAuditError, _norm

SEED = 20261125
MAX_ROWS = 20_000
MAX_SUMMARY_BYTES = 12_000


class RemediationError(ValueError):
    """Short constant safe codes only."""


def digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _context(row: dict) -> str | None:
    text = _norm(" ".join(row["Knowledge"]))
    return digest(text) if text else None


def candidate_remediation(
    rows_by_config: dict[str, list[dict]],
    old_assignment: dict[tuple[str, int], str],
    pubmed_index: dict,
    official_original_500: set[str],
    *,
    seed: int = SEED,
) -> tuple[dict, dict]:
    """Return count-only summary and PRIVATE ephemeral (config,index)->split.

    First quarantine every original PubMedQA official-500 row, ambiguous or
    missing pubid, and nonmatching abstract context. Then group the remainder
    by PubMedQA PMID, exact normalized question and exact normalized Knowledge,
    so source and exact-text proxies cannot straddle new splits.
    """
    if (not isinstance(rows_by_config, dict) or set(rows_by_config) != set(CONFIGS) or
            not isinstance(pubmed_index, dict) or set(pubmed_index) != set(CONFIGS)):
        raise RemediationError("INVALID_SOURCE_CONFIG_SET")
    if (not isinstance(old_assignment, dict) or
            not 3 <= len(old_assignment) <= MAX_ROWS or
            type(seed) is not int or not 0 <= seed < 2**32):
        raise RemediationError("INVALID_PRIVATE_ASSIGNMENTS_OR_SEED")
    if (not isinstance(official_original_500, set) or len(official_original_500) != 500 or
            any(not isinstance(x, str) or not x.isdigit() for x in official_original_500)):
        raise RemediationError("EXPECTED_500_OFFICIAL_TEST_IDS_REQUIRED")

    records = []
    reasons = Counter()
    by_config = defaultdict(Counter)
    previous = Counter()
    original_test_by_old_split = Counter()
    for key, prior_split in sorted(old_assignment.items()):
        if not isinstance(key, tuple) or len(key) != 2 or prior_split not in SPLITS:
            raise RemediationError("BAD_PRIVATE_CANDIDATE_ASSIGNMENT")
        config, index = key
        if (config not in CONFIGS or type(index) is not int or index < 0 or
                index >= len(rows_by_config[config])):
            raise RemediationError("PRIVATE_ASSIGNMENT_INDEX_INVALID")
        row = rows_by_config[config][index]
        if (not isinstance(row, dict) or not isinstance(row.get("Question"), str) or
                not isinstance(row.get("Knowledge"), list) or
                any(not isinstance(x, str) for x in row["Knowledge"]) or
                not _norm(row["Question"])):
            raise RemediationError("PRIVATE_ROW_INVALID")
        previous[prior_split] += 1
        q = fingerprint(row["Question"])
        same = pubmed_index[config].get(q, [])
        other = pubmed_index[CONFIGS[1] if config == CONFIGS[0] else CONFIGS[0]].get(q, [])
        matches = same + other
        source_ids = {x[0] for x in matches}
        reason = None
        if not source_ids:
            reason = "no_pubmedqa_pubid"
        elif len(source_ids) != 1:
            reason = "multiple_pubmedqa_pubids"
        else:
            pubid = next(iter(source_ids))
            knowledge_hash = _context(row)
            contexts = {x[1] for x in matches if x[1]}
            if not knowledge_hash or knowledge_hash not in contexts:
                reason = "source_context_not_exact"
            elif str(pubid) in official_original_500:
                reason = "original_official_500_test_heldout"
                original_test_by_old_split[prior_split] += 1

        if reason is not None:
            reasons[reason] += 1
            by_config[config][reason] += 1
            continue
        records.append({
            "key": key,
            "config": config,
            "pubid": pubid,
            "question_hash": q,
            "knowledge_hash": knowledge_hash,
            "category": row["Category of Hallucination"],
            "old_split": prior_split,
        })

    if len(records) < 3:
        raise RemediationError("TOO_FEW_SOURCE_REVIEWED_ELIGIBLE_ROWS")

    # Union PMID + exact-text proxies. Group IDs do not leave private memory.
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def merge(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    cache = {}
    for i, row in enumerate(records):
        for field in ("pubid", "question_hash", "knowledge_hash"):
            value = row[field]
            if value is None:
                continue
            groupkey = (field, value)
            if groupkey in cache:
                merge(i, cache[groupkey])
            else:
                cache[groupkey] = i

    groups = defaultdict(list)
    for i, row in enumerate(records):
        groups[find(i)].append(row)
    if len(groups) < 3:
        raise RemediationError("INSUFFICIENT_SOURCE_INDEPENDENT_GROUPS")

    # Deterministic group ordering; prevent accidental row-level random split.
    chunks = sorted(
        groups.values(),
        key=lambda group: (
            digest(str(seed) + "|" + str(min(r["pubid"] for r in group))),
            min(r["pubid"] for r in group),
        ),
    )
    n = len(chunks)
    train_groups = min(n - 2, max(1, round(n * 0.80)))
    dev_groups = min(n - train_groups - 1, max(1, round(n * 0.10)))
    cutoffs = (train_groups, dev_groups, n-train_groups-dev_groups)
    if min(cutoffs) < 1:
        raise RemediationError("EMPTY_SOURCE_GROUP_SPLIT")
    new_assignments = {}
    group_counts = Counter()
    split_counts = Counter()
    config_split = {c: Counter() for c in CONFIGS}
    category_counts = {c: {s: Counter() for s in SPLITS} for c in CONFIGS}
    offset = 0
    for split, count in zip(SPLITS, cutoffs):
        for group in chunks[offset:offset+count]:
            group_counts[split] += 1
            for r in group:
                if r["key"] in new_assignments:
                    raise RemediationError("DUPLICATE_PRIVATE_PARENT_INDEX")
                new_assignments[r["key"]] = split
                split_counts[split] += 1
                config_split[r["config"]][split] += 1
                category_counts[r["config"]][split][r["category"]] += 1
        offset += count
    if len(new_assignments) != len(records):
        raise RemediationError("INCOMPLETE_NEW_SOURCE_GROUP_ASSIGNMENT")

    split_pubids, split_questions, split_knowledge = defaultdict(set), defaultdict(set), defaultdict(set)
    for r in records:
        new_split = new_assignments[r["key"]]
        split_pubids[r["pubid"]].add(new_split)
        split_questions[r["question_hash"]].add(new_split)
        split_knowledge[r["knowledge_hash"]].add(new_split)
    if (any(len(x) > 1 for x in split_pubids.values()) or
            any(len(x) > 1 for x in split_questions.values()) or
            any(len(x) > 1 for x in split_knowledge.values())):
        raise RemediationError("POST_SPLIT_SOURCE_GROUP_LEAKAGE_DETECTED")

    mapping = [[c, i, new_assignments[(c, i)]] for c, i in sorted(new_assignments)]
    mapping_digest = digest(json.dumps({"seed": seed, "assignment": mapping},
                                        sort_keys=True, separators=(",", ":")))
    summary = {
        "schema_version": 1,
        "audit_kind": "e1_original_pubmedqa_test_exclusion_provisional_source_group_splits",
        "seed": seed,
        "previous_candidate_parent_records": len(old_assignment),
        "previous_candidate_parent_split_counts": {s: previous[s] for s in SPLITS},
        "original_official_500_heldout_by_previous_split": {s: original_test_by_old_split[s] for s in SPLITS},
        "excluded_parent_records_by_reason": {
            s: reasons[s] for s in (
                "original_official_500_test_heldout",
                "multiple_pubmedqa_pubids",
                "no_pubmedqa_pubid",
                "source_context_not_exact",
            )
        },
        "source_reviewed_candidate_parent_records": len(records),
        "source_reviewed_candidate_answer_examples": 2 * len(records),
        "provisional_parent_split_counts": {s: split_counts[s] for s in SPLITS},
        "provisional_answer_example_split_counts": {s: 2 * split_counts[s] for s in SPLITS},
        "source_question_knowledge_proxy_groups_by_split": {s: group_counts[s] for s in SPLITS},
        "source_configuration_by_split": {
            c: {s: config_split[c][s] for s in SPLITS} for c in CONFIGS
        },
        "category_frequency_sorted_without_names_by_config_split": {
            c: {s: sorted(category_counts[c][s].values(), reverse=True) for s in SPLITS}
            for c in CONFIGS
        },
        "exact_pubid_groups_spanning_revised_splits": 0,
        "exact_question_groups_spanning_revised_splits": 0,
        "exact_knowledge_groups_spanning_revised_splits": 0,
        "private_mapping_sha256": mapping_digest,
        "original_official_500_test_removed_from_all_new_candidate_splits": reasons["original_official_500_test_heldout"] == 500,
        "original_pubmedqa_500_test_independently_reserved": True,
        "scientific_binary_labels_adjudicated": False,
        "complete_semantic_leakage_excluded": False,
        "final_train_dev_test_approved": False,
        "scientific_e1_experiment_executed": False,
        "raw_ids_questions_or_answers_exported": False,
        "status": "PROVISIONAL_SOURCE_GROUP_REMEDIATION_NOT_FINAL_SCIENCE",
    }
    verify_summary(summary)
    return summary, new_assignments


def verify_summary(s: dict) -> None:
    names = {
        "schema_version", "audit_kind", "seed",
        "previous_candidate_parent_records", "previous_candidate_parent_split_counts",
        "original_official_500_heldout_by_previous_split",
        "excluded_parent_records_by_reason", "source_reviewed_candidate_parent_records",
        "source_reviewed_candidate_answer_examples", "provisional_parent_split_counts",
        "provisional_answer_example_split_counts",
        "source_question_knowledge_proxy_groups_by_split", "source_configuration_by_split",
        "category_frequency_sorted_without_names_by_config_split",
        "exact_pubid_groups_spanning_revised_splits",
        "exact_question_groups_spanning_revised_splits",
        "exact_knowledge_groups_spanning_revised_splits",
        "private_mapping_sha256",
        "original_official_500_test_removed_from_all_new_candidate_splits",
        "original_pubmedqa_500_test_independently_reserved",
        "scientific_binary_labels_adjudicated", "complete_semantic_leakage_excluded",
        "final_train_dev_test_approved", "scientific_e1_experiment_executed",
        "raw_ids_questions_or_answers_exported", "status",
    }
    if not isinstance(s, dict) or set(s) != names:
        raise RemediationError("UNSAFE_SUMMARY_FIELD_SET")
    if (type(s["schema_version"]) is not int or s["schema_version"] != 1 or
            s["audit_kind"] != "e1_original_pubmedqa_test_exclusion_provisional_source_group_splits" or
            s["status"] != "PROVISIONAL_SOURCE_GROUP_REMEDIATION_NOT_FINAL_SCIENCE"):
        raise RemediationError("FALSE_SUMMARY_TYPE_OR_STATUS")
    for flag in (
        "scientific_binary_labels_adjudicated", "complete_semantic_leakage_excluded",
        "final_train_dev_test_approved", "scientific_e1_experiment_executed",
        "raw_ids_questions_or_answers_exported",
    ):
        if s[flag] is not False:
            raise RemediationError("FALSE_SCIENTIFIC_APPROVAL")
    if (s["original_pubmedqa_500_test_independently_reserved"] is not True or
            type(s["original_official_500_test_removed_from_all_new_candidate_splits"]) is not bool):
        raise RemediationError("HELDOUT_RESERVATION_FLAG_INVALID")
    for k in (
        "exact_pubid_groups_spanning_revised_splits",
        "exact_question_groups_spanning_revised_splits",
        "exact_knowledge_groups_spanning_revised_splits",
    ):
        if type(s[k]) is not int or s[k] != 0:
            raise RemediationError("SOURCE_GROUP_OVERLAP_REPORTED")
    if (type(s["seed"]) is not int or not 0 <= s["seed"] < 2**32 or
            not isinstance(s["private_mapping_sha256"], str) or
            len(s["private_mapping_sha256"]) != 64 or
            any(c not in "0123456789abcdef" for c in s["private_mapping_sha256"])):
        raise RemediationError("SEED_OR_PRIVATE_DIGEST_INVALID")
    if (type(s["previous_candidate_parent_records"]) is not int or
            not 3 <= s["previous_candidate_parent_records"] <= MAX_ROWS or
            type(s["source_reviewed_candidate_parent_records"]) is not int or
            not 3 <= s["source_reviewed_candidate_parent_records"] <= MAX_ROWS):
        raise RemediationError("REMEDIATION_ROW_COUNT_OUT_OF_POLICY")
    prior = s["previous_candidate_parent_split_counts"]
    old_holdout = s["original_official_500_heldout_by_previous_split"]
    split = s["provisional_parent_split_counts"]
    example = s["provisional_answer_example_split_counts"]
    groups = s["source_question_knowledge_proxy_groups_by_split"]
    for m in (prior, old_holdout, split, example, groups):
        if not isinstance(m, dict) or set(m) != set(SPLITS):
            raise RemediationError("BAD_SPLIT_COUNT_FIELDS")
        if any(type(v) is not int or v < 0 or v > 2*MAX_ROWS for v in m.values()):
            raise RemediationError("BAD_SPLIT_COUNT")
    excluded = s["excluded_parent_records_by_reason"]
    if (not isinstance(excluded, dict) or set(excluded) != {
        "original_official_500_test_heldout", "multiple_pubmedqa_pubids",
        "no_pubmedqa_pubid", "source_context_not_exact",
    } or any(type(v) is not int or v < 0 for v in excluded.values())):
        raise RemediationError("BAD_EXCLUSION_REASON_COUNTS")
    if (sum(prior.values()) != s["previous_candidate_parent_records"] or
            sum(prior.values()) != sum(excluded.values()) + sum(split.values()) or
            sum(split.values()) != s["source_reviewed_candidate_parent_records"] or
            2 * sum(split.values()) != s["source_reviewed_candidate_answer_examples"] or
            any(example[p] != 2*split[p] for p in SPLITS) or
            any(groups[p] < 1 or groups[p] > split[p] for p in SPLITS) or
            sum(old_holdout.values()) != excluded["original_official_500_test_heldout"] or
            excluded["original_official_500_test_heldout"] > 500 or
            s["original_official_500_test_removed_from_all_new_candidate_splits"] != (excluded["original_official_500_test_heldout"] == 500)):
        raise RemediationError("SUMMARY_COUNTERS_NOT_CONSISTENT")
    conf = s["source_configuration_by_split"]
    cat = s["category_frequency_sorted_without_names_by_config_split"]
    if (not isinstance(conf, dict) or set(conf) != set(CONFIGS) or
            not isinstance(cat, dict) or set(cat) != set(CONFIGS)):
        raise RemediationError("SOURCE_CONFIGURATION_MAP_INVALID")
    for config in CONFIGS:
        if not isinstance(conf[config], dict) or set(conf[config]) != set(SPLITS):
            raise RemediationError("CONFIG_SPLIT_MAP_INVALID")
        if not isinstance(cat[config], dict) or set(cat[config]) != set(SPLITS):
            raise RemediationError("CONFIG_CATEGORY_FREQUENCY_INVALID")
        for p in SPLITS:
            count = conf[config][p]
            frequencies = cat[config][p]
            if (type(count) is not int or count < 0 or
                not isinstance(frequencies, list) or
                any(type(v) is not int or v < 0 for v in frequencies) or
                sum(frequencies) != count):
                raise RemediationError("CONFIG_CATEGORY_TOTAL_INVALID")
    for p in SPLITS:
        if sum(conf[config][p] for config in CONFIGS) != split[p]:
            raise RemediationError("CONFIG_SPLIT_TOTAL_MISMATCH")
    if len(json.dumps(s, sort_keys=True).encode("utf-8")) > MAX_SUMMARY_BYTES:
        raise RemediationError("REMEDIATION_AGGREGATE_TOO_LARGE")
