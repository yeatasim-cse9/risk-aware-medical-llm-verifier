"""01F-R prospective metadata-only native split planner.
No HTTP, downloads, row reads, inference or execution authorization.
Input is independently reviewed normalized publisher metadata; not a source of trust.
"""
import hashlib
import json
import re

CANDIDATE = "LAMA-WeST/medhal-lf_qafil_bal"
SPLITS = ("train", "validation", "test")
PATH = re.compile(r"data/(train|validation|test)-([0-9]{5})-of-([0-9]{5})[.]parquet")
SHA = re.compile(r"[0-9a-f]{64}")
REV = re.compile(r"[0-9a-f]{40}")


def require(ok, code):
    if not ok:
        raise ValueError(code)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def train_plan(metadata, expected_revision, expected_metadata_digest):
    """Only plan native train shard access; no content or network operations.
    Fail closed on ambiguous naming. New layouts require prospective review.
    Each normalized shard record carries publisher LFS content SHA256 and size.
    Caller must independently validate the original inventory and pin.
    """
    require(type(expected_revision) is str and REV.fullmatch(expected_revision),
            "REVISION_REQUIRED")
    require(type(expected_metadata_digest) is str and SHA.fullmatch(expected_metadata_digest),
            "METADATA_PIN_REQUIRED")
    require(type(metadata) is dict and set(metadata) == {"repository", "revision", "shards"}
            and metadata["repository"] == CANDIDATE
            and metadata["revision"] == expected_revision, "METADATA_SCOPE")
    require(digest(metadata) == expected_metadata_digest, "METADATA_PIN_MISMATCH")
    rows = metadata["shards"]
    require(type(rows) is list and 3 <= len(rows) <= 10000, "SHARD_INVENTORY")
    seen, hashes = set(), set()
    splits = {s: [] for s in SPLITS}
    for row in rows:
        require(type(row) is dict and set(row) == {"path", "sha256", "size"}, "SHARD_SCHEMA")
        path, sha, size = row["path"], row["sha256"], row["size"]
        require(type(path) is str and PATH.fullmatch(path), "SHARD_PATH")
        require(type(sha) is str and SHA.fullmatch(sha) and sha not in hashes
                and path not in seen and type(size) is int and size > 0, "SHARD_IDENTITY")
        seen.add(path)
        hashes.add(sha)
        split, ordinal, total = PATH.fullmatch(path).groups()
        splits[split].append((int(ordinal), int(total), row))
    for seq in splits.values():
        require(bool(seq) and all(x[1] == len(seq) for x in seq)
                and sorted(x[0] for x in seq) == list(range(len(seq))), "INCOMPLETE_SHARDS")
    return {"repository": CANDIDATE, "revision": expected_revision,
            "train_files": [dict(x[2]) for x in sorted(splits["train"])],
            "metadata_sha256": expected_metadata_digest,
            "actual_file_bytes_verified": False, "rights_approved": False,
            "source_disjointness_verified": False, "execution_authorized": False}


def group_split(groups, n_fit, n_selection):
    """Deterministic label-blind development pilot assignment of unique groups.
    Group tokens must already represent independently qualified source components.
    No claims of source provenance or representative sampling follow from hashing.
    """
    require(type(groups) is list and all(type(g) is str and SHA.fullmatch(g)
                                        for g in groups), "GROUP_SCHEMA")
    require(len(set(groups)) == len(groups), "DUPLICATE_GROUP")
    require(type(n_fit) is int and type(n_selection) is int and
            n_fit > 0 and n_selection > 0 and n_fit + n_selection <= len(groups),
            "GROUP_BUDGET")
    order = sorted(groups, key=lambda g: (hashlib.sha256(("01fr-v1:" + g).encode()).digest(), g))
    return {"fit": order[:n_fit], "selection": order[n_fit:n_fit+n_selection]}


def transitions(initial, revised):
    """Structured correction accounting: labels from an independent locked scorer.
    No clinical or free-text quality judgment. One outcome per paired item.
    """
    allowed = {"correct", "wrong", "abstain", "invalid"}
    require(type(initial) is list and type(revised) is list and
            len(initial) == len(revised) and bool(initial)
            and all(type(x) is str and x in allowed for x in initial + revised),
            "PAIRED_OUTCOMES")
    n = len(initial)
    wrong = initial.count("wrong")
    repaired = sum(a == "wrong" and b == "correct" for a,b in zip(initial,revised))
    harmed = sum(a == "correct" and b == "wrong" for a,b in zip(initial,revised))
    return {"items": n, "initial_wrong": wrong, "wrong_to_correct": repaired,
            "correct_to_wrong": harmed, "revised_correct": revised.count("correct"),
            "revised_wrong": revised.count("wrong"), "revised_abstain": revised.count("abstain"),
            "revised_invalid": revised.count("invalid"),
            "correction_rate": repaired/wrong if wrong else None,
            "net_valid_correct_change": (revised.count("correct")-initial.count("correct"))/n}
