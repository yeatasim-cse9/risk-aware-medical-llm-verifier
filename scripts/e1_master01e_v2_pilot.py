"""Train/dev-only E1 engineering pilot. No data loading, submission or public export.
Private caller must reconstruct pinned MASTER01C mapping and authenticate rows.
This module alone cannot prove source provenance or authorize processing.
"""
import hashlib,math,time
from e1_master01d_v2_integrity import require
import e1_master01e_baselines as b
from e1_master01e_v2_neural import SLOTS,MAX_LEN
SEED=20261126
LIMITS={"train":128,"dev":64}
CONDITIONS=("full","removed","permuted")
def order(p):
    return hashlib.sha256((str(SEED)+":"+p["parent"]).encode()).digest()
def sample(pairs,partition):
    b.validate_pairs(pairs,partition)
    require(partition in LIMITS,"PILOT_PARTITION")
    seen=set();out=[]
    for p in sorted(pairs,key=lambda x:(order(x),x["parent"])):
        if p["source_group"] not in seen:
            seen.add(p["source_group"]);out.append(p)
        if len(out)==LIMITS[partition]:break
    require(len(out)==LIMITS[partition],"PILOT_GROUP_COUNT")
    return out

def fit_head(examples,*,steps=100):
    require(type(examples) is list and bool(examples),"HEAD_ROWS")
    m=len(examples[0][0]);n=len(examples)
    require(m>0 and all(len(x)==m and y in (0,1) and all(math.isfinite(v) for v in x)
            for x,y in examples),"HEAD_SCHEMA")
    mean=[sum(x[j] for x,y in examples)/n for j in range(m)]
    sd=[math.sqrt(sum((x[j]-mean[j])**2 for x,y in examples)/n) or 1 for j in range(m)]
    z=[([1.]+[(v-a)/s for v,a,s in zip(x,mean,sd)],y) for x,y in examples]
    w=[0.]*(m+1)
    for _ in range(steps):
        g=[0.]*(m+1)
        for x,y in z:
            e=b.sigmoid(sum(a*t for a,t in zip(x,w)))-y
            for j,v in enumerate(x):g[j]+=e*v
        for j in range(m+1):w[j]-=.1*(g[j]/n+(.01*w[j] if j else 0))
    return mean,sd,w
def predict_head(head,x):
    mean,sd,w=head
    require(len(x)==len(mean)==len(sd)==len(w)-1,"HEAD_DIM")
    return b.sigmoid(w[0]+sum((v-a)/s*t for v,a,s,t in zip(x,mean,sd,w[1:])))

def validate_private_inputs(train,dev):
    b.disjoint(train,dev)
    # Full retained train/dev counts; not sufficient to authenticate mapping.
    require(len(train)==7427 and len(dev)==890,"RETAINED_PILOT_COUNTS")

def collect(engine,pairs,partition):
    sh=b.context_permutation(pairs,partition,SEED)
    donor={p["parent"]:p["context"] for p in sh}
    out={k:[] for k in CONDITIONS}
    cost={k:dict(forward_examples=0,padded_positions=0,active_tokens=0,real_windows=0)
        for k in CONDITIONS}
    # Fail before inference if any chosen answer/context exceeds fixed budget.
    for p in pairs:
        engine.windows(p["context"]);engine.windows(donor[p["parent"]])
        for a in p["answers"].values():
            engine.encode([],engine.tokens(a))
    for p in pairs:
        full,_=engine.windows(p["context"]);perm,_=engine.windows(donor[p["parent"]])
        for k in CONDITIONS:
            fs=[]
            for role in ("ground_truth","hallucinated"):
                f,c=engine.infer(perm if k=="permuted" else full,p["answers"][role],
                        removed=k=="removed")
                require(set(c)==set(cost[k]) and c["forward_examples"]==SLOTS and
                        c["padded_positions"]==SLOTS*MAX_LEN,"BUDGET_CONTRACT")
                for key,value in c.items():cost[k][key]+=value
                fs.append(f)
            out[k].append((p["source_group"],p["parent"],tuple(fs)))
    require(len({x["padded_positions"] for x in cost.values()})==1,"BUDGET_MISMATCH")
    return out,cost

def scored(rows,fn):
    return [(g,p,tuple(fn(f) for f in fs)) for g,p,fs in rows]
def training(rows):
    return [(f,y) for g,p,fs in rows for y,f in enumerate(fs)]

def length_strata(pairs,scores):
    lookup={p["parent"]:p for p in pairs};groups={k:[] for k in ("gt_shorter","tie","gt_longer_1_5","gt_longer_6_plus")}
    for row in scores:
        p=lookup[row[1]]
        gap=len(b.words(p["answers"]["ground_truth"]))-len(b.words(p["answers"]["hallucinated"]))
        key="gt_shorter" if gap<0 else "tie" if gap==0 else "gt_longer_1_5" if gap<=5 else "gt_longer_6_plus"
        groups[key].append(row)
    # In-memory descriptive output only; future egress validator must review small cells.
    return {k:{"parents":len(v),"metrics":b.metrics(v) if len(v)>=10 else None} for k,v in groups.items()}

def run(engine,train,dev,*,authorized):
    require(authorized is True,"PRIVATE_AUTHORIZATION_REQUIRED")
    validate_private_inputs(train,dev)
    train=sample(train,"train");dev=sample(dev,"dev")
    started=time.monotonic()
    tr,tc=collect(engine,train,"train");dv,dc=collect(engine,dev,"dev")
    full=fit_head(training(tr["full"]));removed=fit_head(training(tr["removed"]))
    results={"neural_full":scored(dv["full"],lambda f:predict_head(full,f)),
        "neural_removed_frozen":scored(dv["removed"],lambda f:predict_head(full,f)),
        "neural_permuted_frozen":scored(dv["permuted"],lambda f:predict_head(full,f)),
        "neural_removed_trained":scored(dv["removed"],lambda f:predict_head(removed,f)),
        "nli_contradiction":scored(dv["full"],lambda f:f[1]),
        "nli_non_entailment":scored(dv["full"],lambda f:1-f[0])}
    for mode in b.MODES:
        results[mode]=b.score_pairs(b.fit(train,mode),dev,"dev")
    # All comparisons exploratory, no best model lock based on pilot.
    metrics={k:b.metrics(v) for k,v in results.items()}
    comparisons={k:b.paired_bootstrap(results["neural_full"],v,replicates=1000)
                 for k,v in results.items() if k!="neural_full"}
    return {"status":"EXPLORATORY_TRAIN_DEV_PILOT_ONLY","train_parents":len(train),
        "dev_parents":len(dev),"metrics":metrics,"exploratory_ap_differences":comparisons,
        "length_strata":{k:length_strata(dev,v) for k,v in results.items()},
        "inference_budget":{"train":tc,"dev":dc},"elapsed_seconds":time.monotonic()-started,
        "clinical_validation":False,"test_used":False,"final_split_frozen":False,
        "e1_execution_authorized":False,"public_export_approved":False}
