"""MASTER 01C private reconstruction/screening adapter, no direct execution.

Uses unchanged pinned 01B encoder routines. Private session returns memory-only
objects; cloud aggregate mode must discard them, never write them to output.
"""
from __future__ import annotations
import copy
import importlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import random
import sys
import math

import e1_master01_validation_kaggle as base
import e1_master01b_kaggle as prior
import e1_master01c_core as core

BASE_SHA="11d803b06167689df6a3caa290be69a8b9b519ee9e83ec0d5de619e95a43761e"
PRIOR_SHA="89bdecce4109aa0cc4dc90f1f8a537ce085f13a707c9c0df9258d36625f62e2f"
PRIOR_EVIDENCE="cac57aa852abc685f4125b276f6080cb5019bee73b5c5e97f93934c104ab321a"
EXPECTED_ASSETS={
    "minilm":"08160d52c713bb7828f661d19ec55be7db23475e3b6c1c1c6013e38922a47810",
    "pubmedbert":"fe18be60b397be8c010fa55a1fc8060c34a4fdd4211cb4255de2c964280eacdf"}
EXPECTED_ENCODERS="d16d1ef50b07cd2c0a8db605b12ca4f4974bff487ed60a20b0ea7e1853f5f73e"
EXPECTED_REVIEW="9f2e43b0d345c1d744b09bd0313ad01d5044f70cd4b72215ad2e84360993934d"
EXPECTED_AFFECTED="d1d09139376682770b3b333b67587e1d72803aa3b9960c0001bfe7d06261d156"
EXPECTED_DIAGNOSTICS="bcab21b763189cfa0c62680062d02ef3e18b89ce62c98d0fccfa08dc4c55e3a3"
EXPECTED_EXACT="4cef385bb61b55cc548cc42729a7c245ed05e12d469195665639b2124c50324d"
MAX_KEYS=("dev__official_holdout","dev__test","test__official_holdout",
          "train__dev","train__official_holdout","train__test")
EXPECTED_MAX={
    "minilm":{"context":(.8149875,.805079,.8148848,.8555893,.8425628,.8781399),
              "question":(.7679858,.7864698,.7360821,.9094671,.8618726,.9268393)},
    "pubmedbert":{"context":(.8465978,.8504726,.8123976,.8757358,.8847558,.9094917),
                  "question":(.7607752,.8390777,.7430628,.8627837,.8386818,.8889172)}}
MAX_TOL=1e-5
EXPECTED_UNION=dict(zip(MAX_KEYS,(5,3,2,60,28,73)))
EXPECTED_ENDPOINTS={"train":152,"dev":57,"test":61,"official_holdout":27}
OLD_VERSIONS={"torch":"2.11.0+cu130","transformers":"5.20.2",
              "huggingface_hub":"1.33.0","numpy":"2.1.3","pyarrow":"23.0.1"}
STAGE="BOOT"
require=core.require

def stage(name):
    global STAGE
    STAGE=name
    print("MASTER01C_STAGE="+name,flush=True)

def compare_maxima(actual):
    require(type(actual) is dict and set(actual)==set(EXPECTED_MAX),"MAXIMA_SCHEMA")
    for k,views in EXPECTED_MAX.items():
        require(type(actual[k]) is dict and set(actual[k])==set(views),"MAXIMA_SCHEMA")
        for v,expected in views.items():
            a=actual[k][v]
            require(type(a) is dict and set(a)==set(MAX_KEYS),"MAXIMA_SCHEMA")
            for pair,e in zip(MAX_KEYS,expected):
                x=a[pair]
                require(type(x) in (int,float) and math.isfinite(x) and
                        -1<=x<=1 and abs(x-e)<=MAX_TOL,"NUMERICAL_REPRODUCTION")

def verify_screen(encoders,review,affected,diagnostics,exact):
    # Exact hashes of the actual verified 01B artifact's aggregate subobjects,
    # excluding the 24 rounded maxima, which have a prespecified tolerance.
    e=copy.deepcopy(encoders)
    maxima={}
    for k,enc in e.items():
        maxima[k]={}
        for v,view in enc["views"].items():
            maxima[k][v]={p:c.pop("maximum_cosine") for p,c in view["comparisons"].items()}
    compare_maxima(maxima)
    require(core.digest(e)==EXPECTED_ENCODERS,"ENCODER_AGGREGATE_REPRODUCTION")
    for x,h in ((review,EXPECTED_REVIEW),(affected,EXPECTED_AFFECTED),
                (diagnostics,EXPECTED_DIAGNOSTICS),(exact,EXPECTED_EXACT)):
        require(core.digest(x)==h,"AGGREGATE_REPRODUCTION")
    return maxima

def reconstruct_private(root):
    """Preserve prior ordering, verified mapping and unique PMID/context lineage.

    Only question/context and source identity are used from the official holdout.
    Candidate labels/answers never enter semantic screening.
    """
    pkg=root/"research_data"
    pkg.mkdir()
    (pkg/"__init__.py").write_text("")
    for p in base.PINS:
        if p.startswith("src/"):
            (pkg/Path(p).name).write_bytes(base.download_module(p))
    sys.path.insert(0,str(root))
    p=root/"helper.py"
    p.write_bytes(base.download_module("scripts/e1_pubmedqa_lineage_kaggle.py"))
    spec=importlib.util.spec_from_file_location("master01c_helper",p)
    helper=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    candidate=importlib.import_module("research_data.e1_candidate_splits")
    advanced=importlib.import_module("research_data.e1_advanced_lineage")
    hard=importlib.import_module("research_data.e1_pubmedqa_test_exclusion")
    snap,pq=helper.libraries()
    med=helper.checked_parquet(snap,pq,helper.MEDHALLU,helper.MEDHALLU_REVISION,
                                helper.SOURCE_SHA256,root/"med",is_source=False)
    pub=helper.checked_parquet(snap,pq,helper.PUBMEDQA,helper.PUBMEDQA_REVISION,
                                helper.PUBMEDQA_SHA256,root/"pub",is_source=True)
    ids,holdout_digest=helper.official_test_ids()
    require(holdout_digest=="939fe566f09017d13b1ca64d2ddfee0bc2374b366048152997669cccedc44d51",
            "HOLDOUT_PIN")
    old_summary,old=candidate.candidate_plan(med)
    require(old_summary["candidate_assignment_sha256"]==helper.CANDIDATE_ASSIGNMENT_SHA256,
            "OLD_MAPPING_PIN")
    index=advanced.pubmed_index(pub)
    summary,assignment=hard.candidate_remediation(med,old,index,ids,seed=base.SEED)
    hard.verify_summary(summary)
    require(summary["private_mapping_sha256"]==base.EXPECTED_MAPPING,"MAPPING_PIN")
    require(core.digest({"seed":base.SEED,"assignment":
                         [[c,i,assignment[c,i]] for c,i in sorted(assignment)]})
            ==base.EXPECTED_MAPPING,"INDEPENDENT_MAPPING_PIN")
    diagnostics=base.grouped_diagnostics(med,assignment)
    records={s:[] for s in core.SPLITS}
    for (c,i),s in sorted(assignment.items()):
        row=med[c][i]
        matches=[m for cfg in index for m in index[cfg].get(advanced.fingerprint(row["Question"]),[])]
        pids={m[0] for m in matches}
        require(len(pids)==1 and hard._context(row) in {m[1] for m in matches},"SOURCE_PROXY")
        pid=str(next(iter(pids)))
        require(pid not in ids,"HOLDOUT_SOURCE_CONTAMINATION")
        # Validate both paired answer roles exist, without using text for screen.
        require(all(type(row.get(k)) is str and row[k].strip()
                    for k in ("Ground Truth","Hallucinated Answer")),"PAIRED_ANSWER_SCHEMA")
        records[s].append({"key":(c,i),"source":pid,"question":row["Question"],
            "context":" ".join(row["Knowledge"]),"config":c,
            "category":row["Category of Hallucination"],"difficulty":row["Difficulty Level"]})
    held={}
    for c in sorted(pub):
        for row in pub[c]:
            pid=str(row["pubid"])
            if pid in ids:
                ctx=row["context"]
                parts=ctx.get("contexts",[]) if isinstance(ctx,dict) else ctx
                item={"question":row["question"],"context":" ".join(parts)}
                require(pid not in held or held[pid]==item,"AMBIGUOUS_HOLDOUT")
                held[pid]=item
    require(set(held)==ids,"HOLDOUT_COVERAGE")
    records["official_holdout"]=[{"key":("official_holdout",i),"source":pid,
        **held[pid],"config":"official_holdout","category":"not_used","difficulty":"not_used"}
        for i,pid in enumerate(sorted(held))]
    require({s:len(v) for s,v in records.items()}==prior.SIZES,"PARENT_COUNTS")
    exact=base.exact_screen(records,advanced._norm)
    require(core.digest(exact)==EXPECTED_EXACT and core.digest(diagnostics)==EXPECTED_DIAGNOSTICS,
            "SOURCE_REPRODUCTION")
    core.validate(records,{p:set() for p in core.PAIRS})
    # med contains candidate answers and is not returned by the semantic adapter.
    return records,advanced._norm,diagnostics,exact

def screen(records,norm,diagnostics,exact,root,device):
    import numpy as np
    import torch
    union={a+"__"+b:np.zeros((prior.SIZES[a],prior.SIZES[b]),dtype=bool)
           for a,b in core.PAIRS}
    results={}
    for key in prior.MODELS:
        tok,model,files=prior.model_load(key,root,device)
        require(core.digest(files)==EXPECTED_ASSETS[key],"MODEL_ASSET_PIN")
        result={"model":prior.MODELS[key][0],"revision":prior.MODELS[key][1],
                "weights_sha256":prior.MODELS[key][2],"file_sha256":files,"views":{}}
        for view in ("question","context"):
            groups={};coverage={}
            for s in core.SPLITS:
                groups[s],coverage[s]=prior.encode_windows([r[view] for r in records[s]],
                    tok,model,device,label=key.upper()+"_"+view.upper()+"_"+s.upper())
            comparisons={}
            for a,b in core.PAIRS:
                name=a+"__"+b
                stage(key.upper()+"_"+view.upper()+"_COMPARE_"+name.upper())
                comparisons[name]=prior.compare_windows(groups[a],groups[b],union[name])
            result["views"][view]={"coverage":coverage,"comparisons":comparisons}
            del groups
            torch.cuda.empty_cache()
        results[key]=result
        del tok,model
        torch.cuda.empty_cache()
    flags={};review={};affected={s:set() for s in core.SPLITS}
    for idx,(a,b) in enumerate(core.PAIRS):
        mask=union[a+"__"+b]
        flags[a,b]={(int(i),int(j)) for i,j in np.argwhere(mask)}
        affected[a].update(i for i,j in flags[a,b])
        affected[b].update(j for i,j in flags[a,b])
        review[a+"__"+b]={"union_pairs":int(mask.sum()),"left_rows":int(mask.any(axis=1).sum()),
            "right_rows":int(mask.any(axis=0).sum()),
            "lexical_sample":prior.lexical_sample(records[a],records[b],mask,norm,base.SEED+idx)}
    maxima=verify_screen(results,review,{s:len(v) for s,v in affected.items()},diagnostics,exact)
    return flags,maxima

def prepare_private(root, *, return_private=False):
    """Authorized private GPU session only. Never print or serialize private return.

    return_private=True additionally requires explicit private reviewer access
    attestation. This is not proof of licensing or authenticated reviewer identity.
    """
    require(os.environ.get("MASTER01C_PRIVATE_RIGHTS_CONFIRMED")=="YES","RIGHTS_REQUIRED")
    require(type(return_private) is bool,"REVIEW_MODE")
    if return_private:
        require(os.environ.get("MASTER01C_PRIVATE_REVIEW_CONFIRMED")=="YES","PRIVATE_REVIEW_REQUIRED")
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
    os.environ["HF_HOME"]=str(root/"hf")
    import numpy as np
    import torch
    require(torch.cuda.is_available(),"KAGGLE_GPU_REQUIRED")
    random.seed(base.SEED)
    torch.manual_seed(base.SEED)
    torch.cuda.manual_seed_all(base.SEED)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    device=torch.device("cuda:0")
    for key in prior.MODELS:
        tok,model,files=prior.model_load(key,root,device)
        require(core.digest(files)==EXPECTED_ASSETS[key],"MODEL_ASSET_PIN")
        groups,cov=prior.encode_windows(["synthetic alpha beta","invented example "*400],
                                      tok,model,device,label="SMOKE_"+key.upper())
        require(cov["total_tokens"]==cov["covered_tokens"] and cov["multiwindow_parents"]==1,
                "MODEL_SMOKE_COVERAGE")
        check=prior.compare_windows(groups[:1],groups[1:],np.zeros((1,1),dtype=bool))
        require(check["parent_pairs"]==1,"MODEL_SMOKE_PAIR")
        del groups,tok,model
        torch.cuda.empty_cache()
    stage("PINNED_SOURCE_RECONSTRUCTION")
    records,norm,diagnostics,exact=reconstruct_private(root)
    flags,maxima=screen(records,norm,diagnostics,exact,root,device)
    stage("PRIVATE_REVIEW_AND_QUARANTINE")
    queue,strata=core.review_plan(records,flags,norm)
    retained,quarantine=core.quarantine(records,flags,norm)
    mapping=sorted([*r["key"],s] for s in core.SPLITS[:3] for r in retained[s])
    gates=core.freeze_preparation(mapping,quarantine["mapping_sha256"],dict.fromkeys(core.GATES,False))
    versions={k:importlib.metadata.version(k) for k in OLD_VERSIONS}
    report={"schema_version":1,"status":"PROVISIONAL_REMEDIATION_REVIEW_REQUIRED",
        "previous_evidence_sha256":PRIOR_EVIDENCE,"previous_mapping_sha256":base.EXPECTED_MAPPING,
        "seed":base.SEED,"encoder_aggregate_sha256":EXPECTED_ENCODERS,
        "maxima":maxima,"strata":strata,"review_queue_count":len(queue),
        "quarantine":quarantine,"balance":core.balance(records,retained),
        "versions":versions,"device":torch.cuda.get_device_name(0),
        "exact_original_pair_membership_verified":False,"human_review_completed":False,
        "clinical_labels_approved":False,"full_article_provenance_certified":False,
        "complete_semantic_leakage_excluded":False,"private_rows_exported":False,
        "private_review_queue_persisted":False,"gates":gates}
    if return_private:
        # Private, authorized caller only. Do not display this tuple in notebooks.
        return report,{"records":records,"review_queue":queue,"retained_mapping":mapping}
    return report
