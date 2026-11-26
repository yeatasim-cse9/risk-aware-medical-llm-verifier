"""MASTER 01D V2: offline, in-memory benchmark integrity preparation.

No dataset loading, GPU, network, writes, automatic approval or clinical labels.
Inputs for a future real audit require separate private processing authorization.
All row-level outputs are private; only explicitly allowlisted aggregates may
leave an authorized environment. Synthetic callers cannot certify real data.
"""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import re
import unicodedata

import e1_master01c_core as c
import e1_master01c_private as prior
import e1_master01c_validate as validator

require=c.require
CHECKPOINT="83de2eb19a6ab64331a88a4ac1a7fe6b26f440fb8430acd0ff8d277e20871a5d"
MAPPING="025d4323b4db15435b21c7479bfc3f63d9dbbef087b2976a26c542959e66c9cb"
PACKAGE_COMMIT="b2dab6ab6cce514c59c5d5df98a83bc70f7355c4"
PACKAGE_HASHES={
 "e1_master01_validation_kaggle":prior.BASE_SHA,
 "e1_master01b_kaggle":prior.PRIOR_SHA,
 "e1_master01c_core":"a30e9116430a568e8fdb3d2aaa06dd791225246fe8b73b7f580b9c010d7a9602",
 "e1_master01c_private":"6b3f764fa79c6f839cc6705c4cc05c3178d3c4432d52c762ecffa25a8c100cbc"}
ORIGINAL={"train":7579,"dev":947,"test":948,"official_holdout":500}
RETAINED={"train":7427,"dev":890,"test":887,"official_holdout":500}
CONFIGS={"pqa_artificial","pqa_labeled"}

def verify_checkpoint(report):
    """Bind to the independently checked 01C canonical aggregate, not just a
    caller-supplied hash or self-consistent replacement. Does not reauthenticate
    the historical Kaggle run or original private pair membership."""
    require(type(report) is dict and report.get("integrity_sha256")==CHECKPOINT,
            "HISTORICAL_CHECKPOINT")
    validator.validate(report,PACKAGE_COMMIT,PACKAGE_HASHES)
    q=report["quarantine"]
    require(q["mapping_sha256"]==MAPPING and
            {s:q["counts"][s]["retained"] for s in c.SPLITS}==RETAINED,
            "HISTORICAL_RETENTION")
    return {"checkpoint_sha256":CHECKPOINT,"mapping_sha256":MAPPING,
            "historical_aggregate_verified":True,"historical_run_reexecuted":False}

def verify_prior_source_files(root=None):
    """Read code bytes only, no data. Detect drift from historical source pins."""
    root=Path(__file__).resolve().parent if root is None else Path(root)
    require(all(hashlib.sha256((root/(k+".py")).read_bytes()).hexdigest()==h
                for k,h in PACKAGE_HASHES.items()),"PRIOR_SOURCE_PIN")

def normalize(text):
    """Supplemental conservative exact proxy; not semantic paraphrase search."""
    require(type(text) is str and bool(text.strip()),"TEXT_SCHEMA")
    v=unicodedata.normalize("NFKC",text).casefold()
    v=" ".join(re.findall(r"\w+",v,flags=re.UNICODE))
    require(bool(v),"EMPTY_NORMALIZED_TEXT")
    return v

def source_components(records):
    """Use all original rows, even excluded ones, to preserve transitive paths.
    PMID/source, normalized question, normalized supplied context are proxies.
    Returns private key -> representative mapping. No proof of full provenance.
    """
    c.validate(records,{p:set() for p in c.PAIRS})
    parent={r["key"]:r["key"] for s in c.SPLITS for r in records[s]}
    def find(k):
        while parent[k]!=k:
            parent[k]=parent[parent[k]];k=parent[k]
        return k
    cache={}
    for s in c.SPLITS:
        for r in records[s]:
            k=r["key"]
            for field in ("source","question","context"):
                v=r[field].strip() if field=="source" else normalize(r[field])
                proxy=(field,v)
                if proxy in cache:
                    a,b=find(k),find(cache[proxy])
                    if a!=b: parent[max(a,b)]=min(a,b)
                else: cache[proxy]=k
    return {k:find(k) for k in parent}

def reconstruct_candidate(records,flags,prior_normalizer):
    """Run historical quarantine and independent supplemental exact isolation.
    Generic low-level routine for synthetic tests or authorized private use.
    No human dispositions, no reinstatement of excluded candidates.
    """
    kept,q=c.quarantine(records,flags,prior_normalizer)
    groups=source_components(records)
    occupied=defaultdict(set)
    for s in c.SPLITS:
        for r in kept[s]: occupied[groups[r["key"]]].add(s)
    require(all(len(v)==1 for v in occupied.values()),"SUPPLEMENTAL_PROXY_CONFLICT")
    mapping=sorted([*r["key"],s] for s in c.SPLITS[:3] for r in kept[s])
    c.freeze_preparation(mapping,q["mapping_sha256"],dict.fromkeys(c.GATES,False))
    require(kept["official_holdout"]==records["official_holdout"],"HOLDOUT_CHANGED")
    return kept,mapping,{"quarantine":q,"balance":c.balance(records,kept),
        "source_group_counts":{s:len({groups[r["key"]] for r in kept[s]}) for s in c.SPLITS},
        "supplemental_exact_conflicts":0}

def verify_holdout_identity(records,official_test_bytes):
    """Private original official file, SHA then ID equality; no use of labels."""
    require(type(official_test_bytes) is bytes and
            hashlib.sha256(official_test_bytes).hexdigest()==prior.DATA_PINS["official_500_sha256"],
            "HOLDOUT_FILE_PIN")
    raw=json.loads(official_test_bytes)
    require(type(raw) is dict and len(raw)==500 and
            all(type(k) is str and k.isdigit() for k in raw),"HOLDOUT_SCHEMA")
    ids=[r["source"] for r in records["official_holdout"]]
    require(len(ids)==500 and len(set(ids))==500 and set(ids)==set(raw),"HOLDOUT_IDENTITY")

def prepare_pinned_candidate(report,records,flags,prior_normalizer,*,screen_evidence,
                             official_test_bytes):
    """For a FUTURE authorized private reconstruction only.
    screen_evidence is a recomputed 01B aggregate from pinned encoder routine;
    validates aggregate equivalence, not individual original pair identity.
    Caller must bind records, flags and evidence to one actual run; this pure
    function cannot authenticate caller provenance. Never returns approval.
    """
    verify_checkpoint(report);verify_prior_source_files()
    require({s:len(records[s]) for s in c.SPLITS}==ORIGINAL,"ORIGINAL_COUNTS")
    verify_holdout_identity(records,official_test_bytes)
    require(all(r["config"] in CONFIGS for s in c.SPLITS[:3] for r in records[s]),
            "NATIVE_CONFIG")
    require(type(screen_evidence) is dict and set(screen_evidence)==
            {"encoders","review","affected","diagnostics","exact"},"SCREEN_EVIDENCE")
    prior.verify_screen(**screen_evidence)
    for a,b in c.PAIRS:
        require(len(flags[a,b])==prior.EXPECTED_UNION[a+"__"+b],"FLAG_COUNT")
    kept,mapping,agg=reconstruct_candidate(records,flags,prior_normalizer)
    require(agg["quarantine"]==report["quarantine"] and
            agg["balance"]==report["balance"],"RECONSTRUCTION_MISMATCH")
    require(c.digest({"seed":c.SEED,"assignment":mapping})==MAPPING,"MAPPING_PIN")
    return kept,mapping,agg

def shortcut_audit(kept,answers):
    """Train/dev only: paired answer-role structural/shortcut diagnostics.
    No test or official holdout answers accepted. Roles are dataset metadata,
    not medical truth. Descriptive length-ranking and exact-copy signals;
    no fitting, no threshold selection, no post-hoc filtering.
    """
    require(type(answers) is dict and set(answers)==
            {r["key"] for s in ("train","dev") for r in kept[s]},"TRAIN_DEV_ANSWER_SCOPE")
    out={}
    for s in ("train","dev"):
        counts=Counter()
        for r in kept[s]:
            a=answers[r["key"]]
            require(type(a) in (tuple,list) and len(a)==2 and
                    all(type(x) is str and bool(x.strip()) for x in a),"PAIRED_ROLE_SCHEMA")
            x,y=map(normalize,a);ctx=normalize(r["context"])
            counts["parents"]+=1
            counts["identical_normalized_roles"]+=int(x==y)
            counts["hallucinated_role_longer"]+=int(len(y.split())>len(x.split()))
            counts["ground_truth_role_longer"]+=int(len(x.split())>len(y.split()))
            counts["length_tie"]+=int(len(x.split())==len(y.split()))
            counts["ground_truth_exact_context_copy"]+=int(x==ctx)
            counts["hallucinated_exact_context_copy"]+=int(y==ctx)
        out[s]={k:counts[k] for k in ("parents","identical_normalized_roles",
            "hallucinated_role_longer","ground_truth_role_longer","length_tie",
            "ground_truth_exact_context_copy","hallucinated_exact_context_copy")}
    return out

def gate_report():
    """Offline readiness report. No caller boolean may create approval."""
    return {"status":"BENCHMARK_ONLY_PREPARATION_NOT_APPROVED",
        "historical_human_review_path_preserved":True,
        "missing_requirements":["fresh_private_processing_rights",
            "authorized_pinned_reconstruction","reference_label_provenance_and_scope",
            "train_dev_shortcut_pilot","locked_analysis_and_model_protocol",
            "authorized_private_manifest_freeze","separate_execution_authorization"],
        "clinical_labels_approved":False,"human_review_completed":False,
        "complete_semantic_leakage_excluded":False,"final_split_approved":False,
        "final_split_frozen":False,"e1_execution_authorized":False,
        "scientific_experiment_executed":False}
