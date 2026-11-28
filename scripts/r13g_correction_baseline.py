"""Exploratory structured correction baseline. No trained models or gold evidence."""
import collections
import hashlib
import json
import random
import re
import time
from pathlib import Path

LABELS = ("significantly decreased", "no significant difference", "significantly increased")
LICENSES = {"CC BY", "CC BY-SA", "CC0"}
SEED = 1307

def require(ok, reason):
    if not ok:
        raise ValueError(reason)

def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()

def norm(s):
    return " ".join(s.lower().split()) if isinstance(s,str) else ""

def reference(row):
    label = norm(row["label"])
    answers = row["answer"]
    if label not in LABELS or not isinstance(answers,list) or not answers:
        return None
    return label if all(norm(a)==label for a in answers) else None

def eligible(r):
    return (r.get("screen")=="preliminary_rights_candidate"
            and r.get("license_code") in LICENSES
            and r.get("is_pmc_openaccess") is True
            and r.get("is_retracted") is False
            and r.get("version_count")==1
            and bool(r.get("doi") and r.get("pmid") and r.get("title_sha256")))

def group_map(inventory):
    # Connect all article components and title-derived trial IDs before filtering.
    parent={r["pmcid"]:r["pmcid"] for r in inventory}
    def root(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]]
            a=parent[a]
        return a
    seen={}
    for r in inventory:
        a=r["pmcid"]
        keys=[("component", str(r["provisional_exact_match_component"]))]
        keys += [("trial",norm(t)) for t in r.get("title_trial_ids",[]) if norm(t)]
        for k in keys:
            if k in seen:
                x,y=root(a),root(seen[k])
                parent[max(x,y)]=min(x,y)
            else:
                seen[k]=a
    return {a:root(a) for a in parent}

def evaluation(group):
    return int(hashlib.sha256(("r13g-v1:"+str(group)).encode()).hexdigest(),16)%4==0

def tokens(s):
    return set(re.findall(r"[a-z][a-z0-9-]{2,}",norm(s)))-{"the","and","with","for","from","than","were","was","group","groups"}

def predict(ico, article):
    # Frozen intentionally simple lexical baseline, not a clinical entailment model.
    sets=[tokens(v) for v in ico]
    if not sets[2] or not isinstance(article,str):
        return None
    scored=[]
    for i,s in enumerate(re.split(r"(?<=[.!?])\s+|\n+",article)):
        t=tokens(s)
        if not (t & sets[2]):
            continue
        score=2*len(t&sets[2])/len(sets[2])+sum(len(t&a)/max(1,len(a)) for a in sets[:2])
        scored.append((-score,i,s.lower()))
    votes=set()
    for _,_,s in sorted(scored)[:3]:
        if re.search(r"\b(?:no significant (?:difference|change|effect)|not statistically significant|did not differ significantly)\b",s):
            votes.add(LABELS[1])
        elif re.search(r"\bsignifican(?:t|tly|ce)\b|\bp\s*[<≤]\s*0\.0[15]\b",s):
            down=bool(re.search(r"\b(?:decreas\w*|reduc\w*|lower|less)\b",s))
            up=bool(re.search(r"\b(?:increas\w*|higher|greater)\b",s))
            if down!=up:
                votes.add(LABELS[0] if down else LABELS[2])
    return next(iter(votes)) if len(votes)==1 else None

def summarize(groups):
    # Each group holds [n, clean preserved, corrupt corrected, changed clean,
    # prediction coverage count] for each arm. Two wrong labels per prompt.
    totals={}
    for arm in ("unchanged","majority","lexical"):
        n,c,w,e,k=[sum(v[arm][i] for v in groups.values()) for i in range(5)]
        totals[arm]={"prompts":n,"clean_preserved":c,"corrupt_corrected":w,
                     "introduced_clean_errors":e,"prediction_available":k,
                     "clean_preservation":c/n,"correction_success":w/(2*n),
                     "introduced_error_rate":e/n,"prediction_coverage":k/n}
    vals=list(groups.values())
    rng=random.Random(SEED)
    draws=[]
    for _ in range(2000):
        ns=diff=0
        for _ in vals:
            v=vals[rng.randrange(len(vals))]
            ns+=v["lexical"][0]
            diff+=v["lexical"][2]-v["majority"][2]
        draws.append(diff/(2*ns))
    draws.sort()
    return totals, {"contrast":"lexical_minus_majority_correction_success",
                    "difference":totals["lexical"]["correction_success"]-totals["majority"]["correction_success"],
                    "percentile_95_ci":[draws[49],draws[1949]],"replicates":2000,
                    "resampling_unit":"metadata_component"}

def run(config):
    import pyarrow.dataset as ds
    start=time.monotonic()
    paths=list(Path("/kaggle/input").rglob("train-00000-of-00001.parquet"))
    require(len(paths)==1,"TRAIN_ASSET_COUNT")
    p=paths[0]
    inv=p.parent/"R13E_PRIVATE_METADATA_INVENTORY.json"
    require(inv.is_file() and sha(p)==config["dataset_sha256"] and sha(inv)==config["inventory_sha256"],"INPUT_INTEGRITY")
    require(p.stat().st_size==config["dataset_bytes"],"INPUT_SIZE")
    inventory=json.loads(inv.read_text())
    by={r["pmcid"]:r for r in inventory}
    require(len(by)==len(inventory),"INVENTORY_DUPLICATE")
    data=ds.dataset(p,format="parquet")
    ids=data.to_table(columns=["id","pmcid","document_id"]).to_pylist()
    count=collections.Counter(r["pmcid"] for r in ids)
    require(len({r["id"] for r in ids})==len(ids) and len(ids)==config["rows"],"ID_COUNT")
    require(count=={a:r["rows"] for a,r in by.items()},"ARTICLE_COUNTS")
    require(all(str(r["document_id"]) in by[r["pmcid"]]["document_ids"] for r in ids),"DOCUMENT_ID")
    groups=group_map(inventory)
    selected=sorted(a for a,r in by.items() if eligible(r))
    require(len(selected)==config["candidate_articles"] and sum(count[a] for a in selected)==config["candidate_rows"],"RIGHTS_COHORT")
    split={a:evaluation(groups[a]) for a in selected}
    # Freeze metadata partition prior to any label access; no tuning on outcomes.
    partition_sha=hashlib.sha256(json.dumps(split,sort_keys=True).encode()).hexdigest()
    filter_=ds.field("pmcid").isin(selected)
    columns=["id","pmcid","intervention","comparator","outcome","label","answer"]
    rows=data.to_table(columns=columns,filter=filter_).to_pylist()
    require(len(rows)==config["candidate_rows"],"PROJECTION_COUNT")
    buckets=collections.defaultdict(list)
    rejected=collections.Counter()
    for r in rows:
        ico=tuple(norm(r[k]) for k in ("intervention","comparator","outcome"))
        if not all(ico):
            rejected["missing_ico"]+=1
            continue
        buckets[(r["pmcid"],ico)].append(r)
    accepted=[]
    for key,rs in buckets.items():
        refs={reference(r) for r in rs}
        if None in refs or len(refs)!=1:
            rejected["invalid_or_conflicting_reference"]+=len(rs)
            continue
        r=min(rs,key=lambda r:r["id"])
        accepted.append((r["id"],key[0],key[1],next(iter(refs))))
        rejected["duplicate_ico_extra"]+=len(rs)-1
    require(len(accepted)+sum(rejected.values())==len(rows),"RECONCILE")
    dev=[r for r in accepted if not split[r[1]]]
    test=[r for r in accepted if split[r[1]]]
    require(bool(dev) and len({groups[r[1]] for r in test})>=30,"GROUPS_INSUFFICIENT")
    dist=collections.Counter(r[3] for r in dev)
    require(set(dist)==set(LABELS),"DEV_CLASSES_MISSING")
    majority=max(LABELS,key=lambda l:dist[l])
    # Only the authorized evaluation cohort's Article_Content and IDs, no
    # annotated evidence, context, question, label, or answer passed to predictor.
    testids=[r[0] for r in test]
    textrows=data.to_table(columns=["id","pmcid","Article_Content"],
                          filter=filter_ & ds.field("id").isin(testids)).to_pylist()
    articles={r["id"]:(r["pmcid"],r["Article_Content"]) for r in textrows}
    require(set(articles)==set(testids) and len(textrows)==len(testids),"TEXT_PROJECTION")
    result={}
    for rid,a,ico,gold in sorted(test):
        require(articles[rid][0]==a,"TEXT_ID_MISMATCH")
        pred=predict(ico,articles[rid][1])
        scores=result.setdefault(groups[a],{arm:[0]*5 for arm in ("unchanged","majority","lexical")})
        # All two alternative corruptions plus clean control. If no rule vote,
        # preserve incoming claim (explicit no-edit policy).
        for arm,out in (("unchanged",None),("majority",majority),("lexical",pred)):
            z=scores[arm]
            z[0]+=1
            z[1]+=int(out is None or out==gold)
            z[2]+=2*int(out==gold)
            z[3]+=int(out is not None and out!=gold)
            z[4]+=int(out is not None)
    metrics,contrast=summarize(result)
    receipt={"status":"EXPLORATORY_BASELINE_COMPLETE","run_id":config["run_id"],
             "source_commit":config["source_commit"],"dataset_sha256":config["dataset_sha256"],
             "inventory_sha256":config["inventory_sha256"],"partition_sha256":partition_sha,
             "candidate_rows":len(rows),"excluded_rows":dict(rejected),
             "development_prompts":len(dev),"evaluation_prompts":len(test),
             "evaluation_groups":len(result),"development_label_counts":dict(dist),
             "evaluation_label_counts":dict(collections.Counter(r[3] for r in test)),
             "metrics":metrics,"paired_contrast":contrast,
             "runtime_seconds":time.monotonic()-start,
             "reference_provenance_established":False,"historical_overlap_checked":False,
             "study_independence_established":False,"clinical_correctness_claim":False,
             "protected_split_read":False,"gpu_used":False,"model_weights_used":False,
             "public_export_approved":False}
    out=Path("/kaggle/working/R13G_RECEIPT.json")
    out.write_text(json.dumps(receipt,sort_keys=True,indent=2))
    print("R13G_EXPLORATORY_CPU_COMPLETE",flush=True)
