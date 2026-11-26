"""R3: offline train-only audit, no acquisition or model execution.
Publisher labels are benchmark factuality, not clinical truth.
All outputs are private unless separately reviewed.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata
import e1_master01f_r2_native_v1 as native
import e1_master01f_r_protocol_v1 as p

REV = "3500188ec0691c1d83e2738592ec01a16e7dbcb5"
PIN = "1eb70ba0c1c1d06549d37c41596e5bdd19d0f2c1953493ebc9de2bb26c36f17a"
FIELDS = ("id", "inner_id", "source", "context", "statement", "label", "synthetic")
SOURCES = {"sumpubmed", "acm", "medmcqa", "medqa"}
MAX_ROWS = 1_000_000
require = p.require


def norm(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def h(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def terms(text):
    return set(re.findall(r"\w+", norm(text)))


def ap(labels, scores):
    """Tie-aware average precision; positive = nonfactual."""
    require(len(labels) == len(scores) and len(labels) > 0
            and 0 < sum(labels) < len(labels), "BOTH_CLASSES_REQUIRED")
    bins = defaultdict(lambda: [0, 0])
    for y, s in zip(labels, scores):
        bins[s][0] += y
        bins[s][1] += 1
    tp = total = 0
    result = 0.0
    for s in sorted(bins, reverse=True):
        pos, count = bins[s]
        tp += pos
        total += count
        result += (pos / sum(labels)) * (tp / total)
    return result


def audit(rows, nfit=256, nselection=128):
    """Group all eligible rows by connected source-ID and normalized exact
    context/statement proxies. Not a complete near-duplicate or article audit.
    Choose representatives and splits without labels; no explanation features.
    """
    parent, links, selected, seen = {}, {}, {}, set()
    counts = Counter()
    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a,b):
        a,b = root(a),root(b)
        if a != b:
            parent[max(a,b)] = min(a,b)
    for i,row in enumerate(rows):
        require(i < MAX_ROWS and type(row) is dict and set(row) == set(FIELDS),
                "ROW_SCHEMA_OR_LIMIT")
        rid, inner, source, context, statement, label, synthetic = (row[k] for k in FIELDS)
        require(type(rid) is str and 0 < len(rid) <= 200 and rid not in seen
                and type(source) is str and source in SOURCES
                and type(label) is bool and type(synthetic) is bool, "ROW_ID_SOURCE_LABEL")
        seen.add(rid)
        counts["rows_seen"] += 1
        if source != "sumpubmed":
            counts["other_source_excluded"] += 1
            continue
        if not (type(inner) is str and 0 < len(inner.strip()) <= 200
                and type(context) is str and type(statement) is str
                and norm(context) and norm(statement)
                and len(context)+len(statement) <= 30000):
            counts["missing_evidence_or_id_or_budget"] += 1
            continue
        key = h("id:" + rid)
        require(key not in parent, "ID_HASH_COLLISION")
        parent[key] = key
        # Exact proxies are conservative: common statements can overmerge.
        for token in ("source:" + source + ":" + inner,
                      "context:" + h(norm(context)), "statement:" + h(norm(statement))):
            if token in links:
                union(key, links[token])
            else:
                links[token] = key
        ct, st = terms(context), terms(statement)
        if not st:
            counts["empty_statement_tokens"] += 1
            continue
        features = (len(statement), len(context), len(st & ct)/len(st), int(synthetic))
        selected[key] = (int(not label), features)
    reps = {}
    for key in selected:
        group = root(key)
        if group not in reps or (h("representative:"+key),key) < (h("representative:"+reps[group]),reps[group]):
            reps[group] = key
    groups = p.group_split(list(reps), nfit, nselection)
    fit = [selected[reps[g]] for g in groups["fit"]]
    sel = [selected[reps[g]] for g in groups["selection"]]
    fy, sy = [x[0] for x in fit], [x[0] for x in sel]
    require(0 < sum(fy) < len(fy) and 0 < sum(sy) < len(sy), "BOTH_CLASSES_REQUIRED")
    diagnostics = {}
    for j,name in enumerate(("statement_length","context_length","lexical_overlap","synthetic_flag_diagnostic_only")):
        f = [x[1][j] for x in fit]
        s = [x[1][j] for x in sel]
        direction = 1 if ap(fy,f) >= ap(fy,[-v for v in f]) else -1
        diagnostics[name] = {"fit_selected_direction": direction,
                             "selection_ap": ap(sy,[direction*v for v in s])}
    return {"counts":dict(counts), "eligible_rows":len(selected),
            "proxy_groups":len(reps), "fit_groups":len(fit), "selection_groups":len(sel),
            "fit_nonfactual":sum(fy), "selection_nonfactual":sum(sy),
            "selection_prevalence":sum(sy)/len(sy), "shortcut_diagnostics":diagnostics,
            "exact_proxy_only":True, "article_independence_verified":False,
            "near_duplicate_search_complete":False, "evidence_semantics_verified":False,
            "clinical_validity_verified":False, "execution_authorized":False,
            "public_export_approved":False}


def verified_rows(root, metadata, approval, approval_pin, *, allow, label_semantics_confirmed):
    require(allow is True and label_semantics_confirmed is True, "AUDIT_AUTHORIZATION_REQUIRED")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and os.environ.get("HF_HUB_OFFLINE") == "1", "OFFLINE_CPU_REQUIRED")
    plan = native.approved_plan(metadata,REV,PIN,approval,approval_pin)
    require(root.is_dir() and not root.is_symlink(), "TRAIN_DIRECTORY")
    entries = {Path(x["path"]).name:x for x in plan["train_files"]}
    require({x.name for x in root.iterdir()} == set(entries), "TRAIN_INVENTORY")
    # Verify ALL bytes before any row reader imported; no validation/test files.
    for name,entry in entries.items():
        require(native.file_identity(root/name,entry["size"]) == entry["sha256"], "TRAIN_INTEGRITY")
    import pyarrow.parquet as pq
    for name in sorted(entries):
        parquet = pq.ParquetFile(root/name)
        require(len(set(parquet.schema_arrow.names)) == len(parquet.schema_arrow.names)
                and set(FIELDS) <= set(parquet.schema_arrow.names), "PARQUET_SCHEMA")
        for batch in parquet.iter_batches(batch_size=128, columns=list(FIELDS)):
            yield from batch.to_pylist()


def main():
    a=argparse.ArgumentParser()
    for k in ("train","metadata","approval","out"):
        a.add_argument("--"+k,type=Path,required=True)
    a.add_argument("--approval-sha256",required=True)
    a.add_argument("--authorized-private-audit",action="store_true")
    a.add_argument("--label-semantics-confirmed",action="store_true")
    x=a.parse_args()
    try:
        require(x.authorized_private_audit and x.label_semantics_confirmed,"AUDIT_AUTHORIZATION_REQUIRED")
        temp=Path("/kaggle/temp").resolve()
        require(temp in x.train.resolve().parents and temp in x.out.parent.resolve().parents,
                "KAGGLE_EPHEMERAL_DESTINATION")
        native.private_parent(x.out.parent)
        require(not x.out.exists() and not x.out.is_symlink(),"DESTINATION_EXISTS")
        report=audit(verified_rows(x.train,native.read_json(x.metadata),native.read_json(x.approval),
                                   x.approval_sha256,allow=True,label_semantics_confirmed=True))
        fd=os.open(x.out,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,"w") as stream:
            json.dump(report,stream,sort_keys=True,allow_nan=False)
        print("MASTER01FR3_PRIVATE_EXPLORATORY_AUDIT_COMPLETE_REVIEW_REQUIRED")
    except Exception:
        print("MASTER01FR3_PRIVATE_AUDIT_BLOCKED")
        raise SystemExit(1) from None


if __name__=="__main__":
    main()
