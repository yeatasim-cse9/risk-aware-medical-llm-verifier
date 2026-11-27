"""R7 development-only lexical experiments. No protected split or GPU access.
R6 grouping algorithm reconstructed; only aggregate results leave Kaggle.
"""
import base64
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import tempfile
import warnings

import e1_master01f_r3_audit_v1 as r3
import e1_master01f_r2_native_v1 as native
import e1_master01fr6r_checkpoint_v1 as cp

require = r3.require
BASE_METRICS = ["ap","auroc","brier"]
FULL_METRICS = BASE_METRICS + ["removed_ap","permuted_ap","statement_ap"] + [
    prefix+suffix for prefix in ("vs_statement","vs_removed","vs_permuted")
    for suffix in ("_delta","_lo","_hi")]
STAGES = [{"name":n,"metrics":BASE_METRICS[:]} for n in
          ("statement_word","statement_char","statement_combined","context_only")] + [
              {"name":"evidence_ablation","metrics":FULL_METRICS[:]}]
R6_ARCHIVE = "88ee04917b72166180cde65f40a289eb7bca29d7"
R6_PATH = "evidence/master01f-r6/report.json"
NBOOT = 2000
SEED = 61707

def reconstruct(row_factory, nfit=256, nselection=128):
    """Exact R3 grouping, representative and allocation rule. Two passes,
    first retains only hashes/labels; second retains selected text in RAM.
    No raw data or membership list is exported."""
    parent, links, selected, seen = {}, {}, {}, set()
    counts = Counter()
    def root(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]
            x=parent[x]
        return x
    def union(a,b):
        a,b=root(a),root(b)
        if a!=b: parent[max(a,b)]=min(a,b)
    for i,row in enumerate(row_factory()):
        require(i<r3.MAX_ROWS and type(row) is dict and set(row)==set(r3.FIELDS),"ROW_SCHEMA_OR_LIMIT")
        rid,inner,source,context,statement,label,synthetic=(row[k] for k in r3.FIELDS)
        require(type(rid) is str and 0<len(rid)<=200 and rid not in seen
                and type(source) is str and source in r3.SOURCES
                and type(label) is bool and type(synthetic) is bool,"ROW_ID_SOURCE_LABEL")
        seen.add(rid); counts["rows_seen"]+=1
        if source!="sumpubmed":
            counts["other_source_excluded"]+=1
            continue
        if not (type(inner) is str and 0<len(inner.strip())<=200 and
                type(context) is str and type(statement) is str and
                r3.norm(context) and r3.norm(statement) and len(context)+len(statement)<=30000):
            counts["missing_evidence_or_id_or_budget"]+=1
            continue
        key=r3.h("id:"+rid)
        require(key not in parent,"ID_HASH_COLLISION")
        parent[key]=key
        for token in ("source:"+source+":"+inner,"context:"+r3.h(r3.norm(context)),
                      "statement:"+r3.h(r3.norm(statement))):
            if token in links: union(key,links[token])
            else: links[token]=key
        if not r3.terms(statement):
            counts["empty_statement_tokens"]+=1
            continue
        selected[key]=int(not label)
    reps={}
    for key in selected:
        g=root(key)
        if g not in reps or (r3.h("representative:"+key),key)<(r3.h("representative:"+reps[g]),reps[g]):
            reps[g]=key
    parts=r3.p.group_split(list(reps),nfit,nselection)
    ids={part:[reps[g] for g in groups] for part,groups in parts.items()}
    targets=set(ids["fit"]+ids["selection"])
    found={}
    for row in row_factory():
        key=r3.h("id:"+row["id"])
        if key in targets:
            require(key not in found and row["source"]=="sumpubmed"
                    and int(not row["label"])==selected[key],"COHORT_DRIFT")
            found[key]=(row["statement"],row["context"],selected[key])
    require(set(found)==targets,"COHORT_MEMBERSHIP")
    fit,sel=([found[k] for k in ids[part]] for part in ("fit","selection"))
    fy,sy=([v[2] for v in rows] for rows in (fit,sel))
    require(0<sum(fy)<len(fy) and 0<sum(sy)<len(sy),"BOTH_CLASSES_REQUIRED")
    return fit,sel,{"counts":dict(counts),"eligible_rows":len(selected),"proxy_groups":len(reps),
                    "fit_groups":len(fit),"selection_groups":len(sel),
                    "fit_nonfactual":sum(fy),"selection_nonfactual":sum(sy),
                    "selection_prevalence":sum(sy)/len(sy)}

def check_r6(summary, report):
    require(type(report) is dict and all(k in report and report[k]==v for k,v in summary.items()),
            "R6_AGGREGATE_RECONSTRUCTION_MISMATCH")
    require(summary["fit_groups"]==256 and summary["selection_groups"]==128,
            "R7_COHORT_SIZE")

def read_r6(store, expected_hash):
    store.verify()
    require(cp.match(cp.HEX,expected_hash),"R6_REPORT_PIN")
    obj=cp.request_json(cp.API+"/contents/"+R6_PATH+"?ref="+R6_ARCHIVE,store.headers)
    require(type(obj) is dict and obj.get("type")=="file" and obj.get("path")==R6_PATH
            and obj.get("encoding")=="base64" and type(obj.get("content")) is str,"R6_ARCHIVE_SCHEMA")
    raw=base64.b64decode("".join(obj["content"].splitlines()),validate=True)
    require(len(raw)<=cp.MAX and len(raw)==obj.get("size") and cp.blob(raw)==obj.get("sha")
            and cp.digest(raw)==expected_hash,"R6_REPORT_INTEGRITY")
    return json.loads(raw)

def runtime():
    names=("numpy","scipy","scikit-learn","pyarrow","threadpoolctl","joblib")
    return {"python":platform.python_version(),"machine":platform.machine(),
            **{n:importlib.metadata.version(n) for n in names}}

def make_plan(commit,run_id,r6_hash,approval_pin,recipe_blobs,runtime_info):
    require(cp.match(cp.HEX,r6_hash) and cp.match(cp.HEX,approval_pin),"INPUT_PINS")
    require(type(recipe_blobs) is dict and bool(recipe_blobs)
            and all(cp.match(cp.COMMIT,v) for v in recipe_blobs.values()),"RECIPE_BLOBS")
    plan={"schema":1,"run_id":run_id,"scope":"private_development_aggregate",
          "source_commit":commit,
          "input_sha256":cp.digest(cp.canonical({"revision":r3.REV,"metadata":r3.PIN,
                                                 "r6_report":r6_hash,"approval":approval_pin})),
          "recipe_sha256":cp.digest(cp.canonical({"blobs":recipe_blobs,"runtime":runtime_info,
                                                  "seed":SEED,"bootstrap":NBOOT})),
          "stages":json.loads(cp.canonical(STAGES))}
    return cp.validate_plan(plan,cp.digest(cp.canonical(plan)))

def split(rows):
    return [r[0] for r in rows],[r[1] for r in rows],[r[2] for r in rows]

def vector(analyzer):
    from sklearn.feature_extraction.text import TfidfVectorizer
    opts={"ngram_range":(1,2),"token_pattern":r"(?u)\b\w+\b","max_features":20000} if analyzer=="word" else {
        "ngram_range":(3,5),"max_features":30000}
    return TfidfVectorizer(analyzer=analyzer,sublinear_tf=True,min_df=1,**opts)

def block(train,test,analyzers):
    from scipy.sparse import hstack
    pairs=[]
    for analyzer in analyzers:
        v=vector(analyzer)
        pairs.append((v.fit_transform(train),v.transform(test),v))
    return hstack([p[0] for p in pairs],format="csr"),hstack([p[1] for p in pairs],format="csr"),[p[2] for p in pairs]

def transformed(vectors, texts):
    from scipy.sparse import hstack
    return hstack([v.transform(texts) for v in vectors],format="csr")

def classifier(x,y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter("error",ConvergenceWarning)
        model=LogisticRegression(C=1.0,solver="liblinear",random_state=SEED,max_iter=1000)
        model.fit(x,y)
    require(list(model.classes_)==[0,1],"CLASS_ORDER")
    return model

def prob(model,x):
    return model.predict_proba(x)[:,1]

def measures(y,s):
    from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
    return {"ap":float(average_precision_score(y,s)),
            "auroc":float(roc_auc_score(y,s)),"brier":float(brier_score_loss(y,s))}

def numeric(statements,contexts):
    # No label, synthetic flag, source ID or explanation features.
    import numpy as np
    out=[]
    for s,c in zip(statements,contexts):
        st,ct=r3.terms(s),r3.terms(c)
        out.append((len(s),len(c),len(st&ct)/max(1,len(st))))
    return np.asarray(out,dtype=float)

def permute(contexts):
    # Distinct representative group at every position (cyclic rotation).
    require(len(contexts)>1,"PERMUTATION_SIZE")
    return contexts[1:]+contexts[:1]

def paired(y,full,comparators,nboot=NBOOT):
    import numpy as np
    y=np.asarray(y); full=np.asarray(full)
    require(type(nboot) is int and nboot>0 and len(y)==len(full) and
            set(y.tolist())=={0,1},"BOOTSTRAP_INPUT")
    comps={k:np.asarray(v) for k,v in comparators.items()}
    require(all(len(v)==len(y) for v in comps.values()),"PAIRED_MEMBERSHIP")
    rng=np.random.default_rng(SEED)
    deltas={k:[] for k in comps}
    def ap(a,b): return r3.ap(a.tolist(),b.tolist())
    for _ in range(nboot):
        idx=rng.integers(0,len(y),size=len(y))
        yy=y[idx]
        if yy.min()==yy.max(): continue
        a=ap(yy,full[idx])
        for k,v in comps.items(): deltas[k].append(a-ap(yy,v[idx]))
    require(all(len(v)>=max(1,int(.95*nboot)) for v in deltas.values()),"BOOTSTRAP_DEGENERATE")
    result={}
    for k,v in comps.items():
        lo,hi=np.percentile(deltas[k],[2.5,97.5])
        result[k+"_delta"]=ap(y,full)-ap(y,v)
        result[k+"_lo"]=float(lo); result[k+"_hi"]=float(hi)
    return result

def experiment(stage,fit,selection,nboot=NBOOT):
    from scipy.sparse import csr_matrix,hstack
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    sf,cf,yf=split(fit); ss,cs,ys=split(selection)
    with threadpool_limits(limits=1):
        if stage in ("statement_word","statement_char","statement_combined","context_only"):
            modes={"statement_word":["word"],"statement_char":["char_wb"],
                   "statement_combined":["word","char_wb"],"context_only":["word","char_wb"]}
            x,z,_=block(cf if stage=="context_only" else sf,
                        cs if stage=="context_only" else ss,modes[stage])
            return measures(ys,prob(classifier(x,yf),z))
        require(stage=="evidence_ablation","STAGE_NAME")
        a,b,_=block(sf,ss,["word","char_wb"])
        d,e,cv=block(cf,cs,["word","char_wb"])
        scaler=StandardScaler().fit(numeric(sf,cf))
        def pack(s,c,sv,cvmat):
            return hstack([sv,cvmat,csr_matrix(scaler.transform(numeric(s,c)))],format="csr")
        model=classifier(pack(sf,cf,a,d),yf)
        full=prob(model,pack(ss,cs,b,e))
        removed=[""]*len(cs)
        perm=permute(cs)
        rem=prob(model,pack(ss,removed,b,transformed(cv,removed)))
        per=prob(model,pack(ss,perm,b,transformed(cv,perm)))
        stmt=prob(classifier(a,yf),b) # separately fitted removal comparator
        result=measures(ys,full)
        result.update(removed_ap=measures(ys,rem)["ap"],permuted_ap=measures(ys,per)["ap"],
                      statement_ap=measures(ys,stmt)["ap"])
        result.update(paired(ys,full,{"vs_statement":stmt,"vs_removed":rem,"vs_permuted":per},nboot))
        return result

def execute(plan,pin,store,metadata,receipt,approval_pin,r6_hash,*,authorized=False,
            label_reviewed=False,aggregate_export_reviewed=False,max_new=None,
            commit=None,recipe_blobs=None):
    """Lazy cohort reconstruction: already-complete runs don't read data.
    No persistent raw checkpoint; only aggregates. No GPU/model download."""
    require(authorized is True and label_reviewed is True and aggregate_export_reviewed is True,
            "OPERATOR_REVIEW_REQUIRED")
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="" and
            os.environ.get("HF_HUB_OFFLINE")=="1","CPU_OFFLINE_LIBRARY_FLAGS")
    require(plan["scope"]=="private_development_aggregate" and plan["stages"]==STAGES,"R7_PLAN")
    expected=make_plan(commit,plan["run_id"],r6_hash,approval_pin,recipe_blobs,runtime())
    require(plan==expected and cp.digest(cp.canonical(expected))==pin,"R7_RUNTIME_PLAN_BINDING")
    native.approved_plan(metadata,r3.REV,r3.PIN,receipt,approval_pin)
    report=read_r6(store,r6_hash)
    cache=None
    def compute(stage,prior):
        nonlocal cache
        if cache is None:
            temp=Path("/kaggle/temp")
            temp.mkdir(exist_ok=True)
            root=Path(tempfile.mkdtemp(prefix="master01fr7-",dir=temp))
            root.chmod(0o700)
            try:
                native.acquire(metadata,r3.REV,r3.PIN,receipt,approval_pin,root/"train",allow=True)
                factory=lambda:r3.verified_rows(root/"train",metadata,receipt,approval_pin,
                                                allow=True,label_semantics_confirmed=True)
                fit,sel,summary=reconstruct(factory)
                check_r6(summary,report)
                cache=(fit,sel)
            finally:
                shutil.rmtree(root)
        return experiment(stage,*cache)
    return cp.run(plan,pin,compute,store,authorized=True,max_new=max_new)
