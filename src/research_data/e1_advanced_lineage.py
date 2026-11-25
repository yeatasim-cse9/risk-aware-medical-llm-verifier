"""Private, aggregate-only PubMedQA source-linkage and advanced split leakage screen.

No dataset/network access here. This module accepts already approved in-memory
MedHallu and PubMedQA data only. It never serializes article identifiers,
fingerprints, questions, medical text, or the row-level assignment.

Exact question + PubMedQA pubid is a PROVISIONAL lineage link; a verified
origin, complete semantic nonoverlap, independent gold splits, and permission to
train E1 still require separate review.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json

from .medhallu_native_audit import _norm

CONFIGS = ("pqa_labeled", "pqa_artificial")
SPLITS = ("train", "dev", "test")
MAX_SOURCE_ROWS = 250_000
MAX_MEDHALLU_ROWS = 20_000
MAX_CANDIDATE_PAIRS = 300_000
MAX_REPORT_BYTES = 12_000


class LineageAuditError(ValueError):
    """Fixed non-sensitive error codes only."""


def fingerprint(s: str) -> str:
    return sha256(_norm(s).encode("utf-8")).hexdigest()


def context_fingerprint(context: object) -> str | None:
    """Normalize native PubMedQA context fields without exposing their values."""
    parts = []
    if isinstance(context, dict):
        parts = context.get("contexts", [])
    elif isinstance(context, list):
        parts = context
    if not isinstance(parts, list) or not all(isinstance(x, str) for x in parts):
        return None
    joined = _norm(" ".join(parts))
    return sha256(joined.encode("utf-8")).hexdigest() if joined else None


def pubmed_index(source_rows: dict[str, list[dict]]) -> dict:
    """Index source by hashed question; values remain private PMID/context tuples."""
    if not isinstance(source_rows, dict) or set(source_rows) != set(CONFIGS):
        raise LineageAuditError("INVALID_PUBMEDQA_CONFIG_SET")
    index = {}
    for config in CONFIGS:
        data = source_rows[config]
        if not isinstance(data, list) or not 0 < len(data) <= MAX_SOURCE_ROWS:
            raise LineageAuditError("INVALID_PUBMEDQA_CONFIG_ROW_COUNT")
        lookup = defaultdict(list)
        for item in data:
            if not isinstance(item, dict) or not {"pubid", "question", "context"} <= set(item):
                raise LineageAuditError("PUBMEDQA_SOURCE_SCHEMA_CHANGED")
            pubid, question = item["pubid"], item["question"]
            if (type(pubid) is not int or not 0 < pubid <= 2**32 or
                    not isinstance(question, str) or not 0 < len(question) < 5_000 or
                    not _norm(question)):
                raise LineageAuditError("PUBMEDQA_SOURCE_ID_OR_QUESTION_INVALID")
            lookup[fingerprint(question)].append((pubid, context_fingerprint(item["context"])))
        index[config] = dict(lookup)
    return index


def _shingles(text: str, n: int) -> frozenset[str]:
    normalized = _norm(text)
    if not normalized:
        return frozenset()
    words = normalized.split()
    if len(words) >= n:
        return frozenset(" ".join(words[i:i+n]) for i in range(len(words)-n+1))
    return frozenset({normalized})


def _chargrams(text: str, n: int = 5) -> frozenset[str]:
    normalized = _norm(text)
    if not normalized:
        return frozenset()
    if len(normalized) < n:
        return frozenset({normalized})
    return frozenset(normalized[i:i+n] for i in range(len(normalized)-n+1))


def _jaccard(a: frozenset, b: frozenset) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def advanced_near_screen(
    source_rows: dict[str, list[dict]],
    assignment: dict[tuple[str, int], str],
    *,
    candidate_limit: int = MAX_CANDIDATE_PAIRS,
) -> dict:
    """Broader rare-trigram plus short-question char-gram screening.

    Candidate-indexed (NOT exhaustive). Block common tokens; report skipped
    buckets, candidate cap and unmatched short text instead of false assurance.
    """
    if type(candidate_limit) is not int or not 1 <= candidate_limit <= MAX_CANDIDATE_PAIRS:
        raise LineageAuditError("INVALID_NEAR_SCREEN_PAIR_BUDGET")
    questions = []
    for config in CONFIGS:
        for i, row in enumerate(source_rows[config]):
            split = assignment.get((config, i))
            if split is None:
                continue
            text = row["Question"]
            if not isinstance(text, str) or not _norm(text) or split not in SPLITS:
                raise LineageAuditError("INVALID_PRIVATE_QUESTION_OR_SPLIT")
            words = _norm(text).split()
            blocks = _shingles(text, 3) if len(words) >= 3 else _chargrams(text)
            questions.append({
                "split": split,
                "blocks": blocks,
                "words": _shingles(text, 3),
                "chars": _chargrams(text),
                "short": len(words) < 3,
            })
    posting = defaultdict(list)
    for i, item in enumerate(questions):
        for block in item["blocks"]:
            posting[block].append(i)
    seen_pairs = set()
    counts = Counter()
    cap = False
    for block in sorted(posting):
        bucket = posting[block]
        if len(bucket) > 60:
            counts["common_blocks_skipped"] += 1
            continue
        for a_pos in range(len(bucket)):
            a = bucket[a_pos]
            for b_pos in range(a_pos + 1, len(bucket)):
                b = bucket[b_pos]
                if questions[a]["split"] == questions[b]["split"]:
                    continue
                pair = (min(a, b), max(a, b))
                if pair in seen_pairs:
                    continue
                if len(seen_pairs) >= candidate_limit:
                    cap = True
                    break
                seen_pairs.add(pair)
                qa, qb = questions[a], questions[b]
                token_sim = _jaccard(qa["words"], qb["words"])
                char_sim = _jaccard(qa["chars"], qb["chars"])
                token_hit = token_sim >= 0.72
                char_hit = char_sim >= 0.82
                if token_hit:
                    counts["high_word_trigram_pairs"] += 1
                if char_hit:
                    counts["high_char_fivegram_pairs"] += 1
                if token_hit or char_hit:
                    counts["high_overlap_unique_pairs"] += 1
                if qa["short"] or qb["short"]:
                    counts["short_question_pairs_compared"] += 1
            if cap:
                break
        if cap:
            break
    return {
        "method": "bounded_rare_word_trigram_and_short_question_char5",
        "candidate_pairs_compared": len(seen_pairs),
        "candidate_limit": candidate_limit,
        "candidate_limit_reached": cap,
        "high_word_trigram_pairs": counts["high_word_trigram_pairs"],
        "high_char_fivegram_pairs": counts["high_char_fivegram_pairs"],
        "high_overlap_unique_pairs": counts["high_overlap_unique_pairs"],
        "short_questions_in_scope": sum(q["short"] for q in questions),
        "short_question_pairs_compared": counts["short_question_pairs_compared"],
        "common_blocks_skipped": counts["common_blocks_skipped"],
        "complete_pairwise_search": False,
        "semantic_paraphrase_search_performed": False,
    }


def audit_lineage(
    rows: dict[str, list[dict]],
    assignments: dict[tuple[str, int], str],
    source_index: dict,
    original_test_pubids: set[str],
) -> dict:
    if not isinstance(rows, dict) or set(rows) != set(CONFIGS):
        raise LineageAuditError("INVALID_MEDHALLU_CONFIG_SET")
    if not isinstance(assignments, dict) or not assignments or len(assignments) > MAX_MEDHALLU_ROWS:
        raise LineageAuditError("INVALID_CANDIDATE_ASSIGNMENT")
    if not isinstance(source_index, dict) or set(source_index) != set(CONFIGS):
        raise LineageAuditError("INVALID_SOURCE_INDEX")
    if (not isinstance(original_test_pubids, set) or
            len(original_test_pubids) != 500 or
            any(not isinstance(x, str) or not x.isdigit() for x in original_test_pubids)):
        raise LineageAuditError("ORIGINAL_PUBMEDQA_TEST_SET_INVALID")
    source_rows_by_config = Counter()
    matched_splits_by_pubid = defaultdict(set)
    matched_parent_counts_by_pubid = Counter()
    source_match = Counter()
    source_by_medhallu_config = defaultdict(Counter)
    official_test_by_split = Counter()
    exact_question_hash_by_split = defaultdict(set)
    exact_knowledge_hash_by_split = defaultdict(set)
    eligible_by_split = Counter()
    for (config, idx), split in assignments.items():
        if config not in CONFIGS or type(idx) is not int or not 0 <= idx < len(rows[config]) or split not in SPLITS:
            raise LineageAuditError("ASSIGNMENT_NOT_WITHIN_SOURCE_ROWS")
        row = rows[config][idx]
        if not isinstance(row, dict) or not isinstance(row.get("Question"), str) or not isinstance(row.get("Knowledge"), list):
            raise LineageAuditError("MEDHALLU_ROW_SCHEMA_INVALID")
        eligible_by_split[split] += 1
        q_hash = fingerprint(row["Question"])
        exact_question_hash_by_split[q_hash].add(split)
        knowledge = _norm(" ".join(row["Knowledge"]))
        if knowledge:
            exact_knowledge_hash_by_split[sha256(knowledge.encode("utf-8")).hexdigest()].add(split)
        same = source_index[config].get(q_hash, [])
        alternate = source_index[CONFIGS[1] if config == CONFIGS[0] else CONFIGS[0]].get(q_hash, [])
        candidates = same or alternate
        pmids = {pubid for pubid, _ in candidates}
        if not pmids:
            source_match["unmatched"] += 1
            source_by_medhallu_config[config]["unmatched"] += 1
        elif len(pmids) > 1:
            source_match["ambiguous_multiple_pubids"] += 1
            source_by_medhallu_config[config]["ambiguous_multiple_pubids"] += 1
        else:
            pubid = next(iter(pmids))
            label = "same_configuration" if same else "cross_configuration_fallback"
            source_match[label] += 1
            source_by_medhallu_config[config][label] += 1
            context_hash = sha256(knowledge.encode("utf-8")).hexdigest() if knowledge else None
            contexts = {ctx for _, ctx in candidates if ctx}
            if context_hash and context_hash in contexts:
                source_match["exact_context_agreement"] += 1
            elif contexts:
                source_match["context_not_exact"] += 1
            else:
                source_match["context_unavailable"] += 1
            matched_splits_by_pubid[pubid].add(split)
            matched_parent_counts_by_pubid[pubid] += 1
            if str(pubid) in original_test_pubids:
                official_test_by_split[split] += 1
    total = len(assignments)
    matched = source_match["same_configuration"] + source_match["cross_configuration_fallback"]
    if total != (matched + source_match["unmatched"] + source_match["ambiguous_multiple_pubids"]):
        raise LineageAuditError("SOURCE_MATCH_COUNT_MISMATCH")
    source_cross = {pubid for pubid, splits in matched_splits_by_pubid.items() if len(splits) > 1}
    report = {
        "audit_kind": "e1_pubmedqa_candidate_pubid_and_leakage_screen",
        "schema_version": 1,
        "eligible_parent_rows_checked": total,
        "matched_unique_pubid_parent_rows": matched,
        "matched_same_configuration": source_match["same_configuration"],
        "matched_cross_configuration_fallback": source_match["cross_configuration_fallback"],
        "unmatched_parent_rows": source_match["unmatched"],
        "ambiguous_multiple_pubids_parent_rows": source_match["ambiguous_multiple_pubids"],
        "pubmedqa_context_exact_agreement_rows": source_match["exact_context_agreement"],
        "pubmedqa_context_not_exact_rows": source_match["context_not_exact"],
        "pubmedqa_context_unavailable_rows": source_match["context_unavailable"],
        "matched_unique_pubids": len(matched_parent_counts_by_pubid),
        "source_pubids_spanning_candidate_partitions": len(source_cross),
        "parent_rows_affected_by_cross_split_source_pubid": sum(matched_parent_counts_by_pubid[x] for x in source_cross),
        "official_pubmedqa_500_test_overlap_parent_rows_by_candidate_split": {s: official_test_by_split[s] for s in SPLITS},
        "exact_question_fingerprint_groups_spanning_splits": sum(len(x) > 1 for x in exact_question_hash_by_split.values()),
        "exact_knowledge_fingerprint_groups_spanning_splits": sum(len(x) > 1 for x in exact_knowledge_hash_by_split.values()),
        "medhallu_source_match_count_by_config": {
            config: {k: source_by_medhallu_config[config][k]
                     for k in ("same_configuration", "cross_configuration_fallback",
                               "unmatched", "ambiguous_multiple_pubids")}
            for config in CONFIGS
        },
        "eligible_parent_rows_by_candidate_split": {s: eligible_by_split[s] for s in SPLITS},
        "source_pubmedqa_pubids_exported": False,
        "original_medical_questions_answers_exported": False,
        "original_pubmedqa_500_test_ids_exported": False,
        "complete_article_provenance_certified": False,
        "final_leakage_safe_splits_approved": False,
        "scientific_experiments_executed": False,
        "status": "LINEAGE_EVIDENCE_SCREEN_ONLY_NO_FINAL_E1_APPROVAL",
    }
    return report


def validate_report(report: dict) -> None:
    if not isinstance(report, dict) or report.get("audit_kind") != "e1_pubmedqa_candidate_pubid_and_leakage_screen":
        raise LineageAuditError("LINEAGE_REPORT_TYPE_INVALID")
    if report.get("schema_version") != 1 or report.get("status") != "LINEAGE_EVIDENCE_SCREEN_ONLY_NO_FINAL_E1_APPROVAL":
        raise LineageAuditError("FALSE_LINEAGE_READY_STATUS")
    for k in (
        "source_pubmedqa_pubids_exported", "original_medical_questions_answers_exported",
        "original_pubmedqa_500_test_ids_exported", "complete_article_provenance_certified",
        "final_leakage_safe_splits_approved", "scientific_experiments_executed",
    ):
        if report.get(k) is not False:
            raise LineageAuditError("FALSE_APPROVAL_OR_PRIVATE_EXPORT_CLAIM")
    numeric = (
        "eligible_parent_rows_checked", "matched_unique_pubid_parent_rows",
        "matched_same_configuration", "matched_cross_configuration_fallback",
        "unmatched_parent_rows", "ambiguous_multiple_pubids_parent_rows",
        "pubmedqa_context_exact_agreement_rows", "pubmedqa_context_not_exact_rows",
        "pubmedqa_context_unavailable_rows", "matched_unique_pubids",
        "source_pubids_spanning_candidate_partitions",
        "parent_rows_affected_by_cross_split_source_pubid",
        "exact_question_fingerprint_groups_spanning_splits",
        "exact_knowledge_fingerprint_groups_spanning_splits",
    )
    if any(type(report.get(k)) is not int or not 0 <= report[k] <= MAX_MEDHALLU_ROWS for k in numeric):
        raise LineageAuditError("SOURCE_LINEAGE_COUNTER_INVALID")
    n = report["eligible_parent_rows_checked"]
    matched = report["matched_unique_pubid_parent_rows"]
    if (n != matched + report["unmatched_parent_rows"] + report["ambiguous_multiple_pubids_parent_rows"] or
            matched != report["matched_same_configuration"] + report["matched_cross_configuration_fallback"] or
            matched != sum(report[k] for k in ("pubmedqa_context_exact_agreement_rows",
                                              "pubmedqa_context_not_exact_rows",
                                              "pubmedqa_context_unavailable_rows")) or
            n != sum(report.get("eligible_parent_rows_by_candidate_split", {}).values()) or
            report["parent_rows_affected_by_cross_split_source_pubid"] > matched):
        raise LineageAuditError("LINEAGE_REPORT_COUNTERS_INCONSISTENT")
    for m in (report.get("eligible_parent_rows_by_candidate_split"),
              report.get("official_pubmedqa_500_test_overlap_parent_rows_by_candidate_split")):
        if not isinstance(m, dict) or set(m) != set(SPLITS) or any(type(v) is not int or v < 0 or v > MAX_MEDHALLU_ROWS for v in m.values()):
            raise LineageAuditError("LINEAGE_SPLIT_MAP_INVALID")
    if sum(report["official_pubmedqa_500_test_overlap_parent_rows_by_candidate_split"].values()) > matched:
        raise LineageAuditError("ORIGINAL_TEST_OVERLAP_COUNTER_INVALID")
    by_cfg = report.get("medhallu_source_match_count_by_config")
    if not isinstance(by_cfg, dict) or set(by_cfg) != set(CONFIGS):
        raise LineageAuditError("LINEAGE_CONFIG_MAP_INVALID")
    allowed_counters = {"same_configuration", "cross_configuration_fallback", "unmatched", "ambiguous_multiple_pubids"}
    for config, m in by_cfg.items():
        if not isinstance(m, dict) or set(m) != allowed_counters:
            raise LineageAuditError("LINEAGE_CONFIG_COUNTERS_INVALID")
        if any(type(v) is not int or v < 0 for v in m.values()):
            raise LineageAuditError("LINEAGE_CONFIG_NEGATIVE_COUNTER")
    if sum(sum(m.values()) for m in by_cfg.values()) != n:
        raise LineageAuditError("LINEAGE_CONFIG_TOTAL_MISMATCH")
    if len(json.dumps(report, sort_keys=True)) > MAX_REPORT_BYTES:
        raise LineageAuditError("LINEAGE_REPORT_OVERSIZED")
