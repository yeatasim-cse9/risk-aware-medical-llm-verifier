"""Authorized private reconstruction using the unchanged historical screen.
No execution on import. Only the controller's explicitly authorized package
may call run. No human review attestation, no E1 or revision inference.
"""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import random
import urllib.request
import re
import e1_master01c_core as c
import e1_master01c_private as p
import e1_master01d_v2_integrity as d
import e1_master01d_v2_cohort as cohort

require=c.require

def helper_at(root):
    path=root/"helper.py"
    require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==
            p.base.PINS["scripts/e1_pubmedqa_lineage_kaggle.py"],"HELPER_PIN")
    spec=importlib.util.spec_from_file_location("master01dv3_helper",path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    return helper

def checked_local(root,spec,pq,columns):
    relative,size,sha,count=spec
    path=root/relative
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve())
            and path.stat().st_size==size,"LOCAL_PARQUET")
    with path.open("rb") as f:
        h=hashlib.file_digest(f,"sha256").hexdigest()
    require(h==sha,"LOCAL_PARQUET_PIN")
    reader=pq.ParquetFile(path)
    require(reader.metadata.num_rows==count and set(columns)<=set(reader.schema_arrow.names),
            "LOCAL_PARQUET_SCHEMA")
    rows=reader.read(columns=columns).to_pylist()
    require(len(rows)==count,"LOCAL_PARQUET_COUNT")
    return rows

def holdout_bytes(helper):
    # Exact already authorized/pinned official file, no other acquisition.
    url=("https://raw.githubusercontent.com/pubmedqa/pubmedqa/"+
         helper.PUBMEDQA_OFFICIAL_GIT_REVISION+"/data/test_ground_truth.json")
    with urllib.request.urlopen(urllib.request.Request(url,headers={
            "User-Agent":"master01dv3-private-audit"}),timeout=45) as response:
        raw=response.read(20001)
    require(len(raw)<=20000 and hashlib.sha256(raw).hexdigest()==
            p.DATA_PINS["official_500_sha256"],"HOLDOUT_PIN")
    return raw

def pubmed_inventory(report,records,raw,rows):
    """Already pinned PubMedQA labeled source only. No BioASQ acquisition.
    Gold values only checked for vocabulary/availability on nonholdout rows;
    never exported or used to determine splits. Official holdout is protected.
    """
    protected=cohort.protected_from_01c(report,records,raw)
    ids=set(json.loads(raw));seen=set();candidates=[]
    require(type(rows) is list and len(rows)==1000,"LABELED_COUNT")
    for row in rows:
        pid=str(row["pubid"])
        require(re.fullmatch(r"[1-9][0-9]*",pid) is not None and pid not in seen,
                "LABELED_PMID")
        seen.add(pid)
        if pid in ids: continue
        ctx=row["context"];parts=ctx.get("contexts",[]) if isinstance(ctx,dict) else ctx
        require(type(parts) is list and all(type(v) is str for v in parts),"CONTEXT_SCHEMA")
        context=" ".join(parts)
        candidates.append({"dataset":"pubmedqa","release":p.DATA_PINS["pubmedqa_revision"],
            "id":pid,"task":"yes_no_maybe","native_split":"development",
            "sources":("pmid:"+pid,),"question":row["question"],"context":context,
            "evidence_available":bool(context.strip()),
            "reference_available":row["final_decision"] in ("yes","no","maybe"),
            "provenance_known":True})
    require(ids<=seen and len(candidates)==500,"NONHOLDOUT_COUNT")
    _,summary=cohort.plan(protected+candidates,dataset="pubmedqa",
                    release=p.DATA_PINS["pubmedqa_revision"],task="yes_no_maybe")
    # No private manifest or new digest exported. Only count-level proxy inventory.
    return {k:summary[k] for k in ("candidate_rows","excluded_rows",
            "eligible_group_representatives")}

def run(root,report):
    require(os.environ.get("MASTER01DV3_AUTHORIZED_ONE_GPU_AUDIT")=="YES","AUTHORIZATION_REQUIRED")
    require(isinstance(root,Path) and root.is_dir() and not root.is_symlink()
            and not any(root.iterdir()),"PRIVATE_SCRATCH")
    d.verify_checkpoint(report);d.verify_prior_source_files()
    os.environ["MASTER01C_PRIVATE_RIGHTS_CONFIRMED"]="YES"
    # Never set MASTER01C_PRIVATE_REVIEW_CONFIRMED or request private reviewer packets.
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
    os.environ["HF_HOME"]=str(root/"hf")
    import numpy as np
    import torch
    import pyarrow.parquet as pq
    require(torch.cuda.is_available(),"GPU_REQUIRED")
    versions={k:importlib.metadata.version(k) for k in p.OLD_VERSIONS}
    require(versions==p.OLD_VERSIONS,"RUNTIME_VERSION_DRIFT")
    random.seed(c.SEED);torch.manual_seed(c.SEED);torch.cuda.manual_seed_all(c.SEED)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    device=torch.device("cuda:0")
    for key in p.prior.MODELS:
        tok,model,files=p.prior.model_load(key,root,device)
        require(c.digest(files)==p.EXPECTED_ASSETS[key],"MODEL_ASSET_PIN")
        groups,cov=p.prior.encode_windows(["synthetic alpha beta","invented example "*400],
                         tok,model,device,label="SMOKE_"+key.upper())
        require(cov["total_tokens"]==cov["covered_tokens"] and cov["multiwindow_parents"]==1,
                "GPU_SMOKE_COVERAGE")
        require(p.prior.compare_windows(groups[:1],groups[1:],np.zeros((1,1),dtype=bool))[
                "parent_pairs"]==1,"GPU_SMOKE_PAIR")
        del groups,tok,model
        torch.cuda.empty_cache()
    records,norm,diagnostics,exact,unused=p.reconstruct_private(root,return_private=False)
    require(unused=={},"UNEXPECTED_PRIVATE_ANSWERS")
    flags,maxima=p.screen(records,norm,diagnostics,exact,root,device)
    # screen already verifies all pinned 01B aggregates and numerical maxima.
    for a,b in c.PAIRS:
        require(len(flags[a,b])==p.EXPECTED_UNION[a+"__"+b],"FLAG_COUNT")
    kept,mapping,agg=d.reconstruct_candidate(records,flags,norm)
    require(agg["quarantine"]==report["quarantine"] and agg["balance"]==report["balance"]
            and c.digest({"seed":c.SEED,"assignment":mapping})==d.MAPPING,"CHECKPOINT_DRIFT")
    helper=helper_at(root);raw=holdout_bytes(helper)
    d.verify_holdout_identity(records,raw)
    answers={}
    for cfg in sorted(d.CONFIGS):
        rows=checked_local(root/"med",helper.SOURCE_SHA256[cfg],pq,
                           ["Ground Truth","Hallucinated Answer"])
        for s in ("train","dev"):
            for r in kept[s]:
                if r["config"]==cfg:
                    row=rows[r["key"][1]]
                    answers[r["key"]]=(row["Ground Truth"],row["Hallucinated Answer"])
        del rows
    shortcuts=d.shortcut_audit(kept,answers);del answers
    rows=checked_local(root/"pub",helper.PUBMEDQA_SHA256["pqa_labeled"],pq,
                       ["pubid","question","context","final_decision"])
    inventory=pubmed_inventory(report,records,raw,rows);del rows
    power=cohort.prospective_power(inventory["eligible_group_representatives"]["calibration"])
    return {"schema_version":1,"status":"PRIVATE_BENCHMARK_AUDIT_ONLY",
        "checkpoint_sha256":d.CHECKPOINT,"mapping_sha256":d.MAPPING,
        "retained":d.RETAINED,"source_group_counts":agg["source_group_counts"],
        "residual_flagged_pairs":0,"supplemental_exact_conflicts":0,
        "source_proxy_conflicts":0,"maxima":maxima,"versions":versions,
        "device":torch.cuda.get_device_name(0),"shortcut_diagnostics":shortcuts,
        "pubmedqa_inventory":inventory,"hypothetical_power":power,
        "bioasq_status":"NOT_ACQUIRED_RELEASE_AND_RIGHTS_PENDING",
        "inventory_scope":"EXACT_PROXY_ONLY_PRIOR_01C_ORIGINALS_PROTECTED",
        "human_review_completed":False,"clinical_labels_approved":False,
        "complete_semantic_leakage_excluded":False,"iid_groups_verified":False,
        "new_calibration_cohort_approved":False,"final_split_frozen":False,
        "e1_execution_authorized":False,"scientific_experiment_executed":False,
        "private_rows_exported":False}
