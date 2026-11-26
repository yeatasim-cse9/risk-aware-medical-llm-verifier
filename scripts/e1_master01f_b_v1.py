"""MASTER 01F-B v1. Pure functions; no data acquisition, GPU, test answers or exports.
All scores target dataset answer role, not clinical truth.
Caller must separately authenticate source-component membership and private inputs.
"""
import hashlib
import math
import random
from dataclasses import dataclass
import e1_master01e_baselines as b
from e1_master01e_v2_pilot import fit_head,predict_head

VERSION="master01fb-v1"
SEED=20261126
PARTITIONS=("fit","selection","calibration")
CONDITIONS=("full","removed","permuted")
MODES=("legacy_lexical","evidence_only","question_answer_overlap","length","style",
       "answer_only","neural","hybrid","hybrid_legacy")
ROLES=("ground_truth","hallucinated")
def require(ok,code):
    if not ok:raise ValueError(code)
def key(x):
    return hashlib.sha256((VERSION+":"+str(SEED)+":"+x).encode()).digest()
def prob(x):
    return type(x) in (int,float) and math.isfinite(x) and 0<=x<=1

@dataclass(frozen=True)
class Plan:
    # Private in-memory identifiers only. Never serialize publicly.
    members:tuple
    exposed_fit_groups:frozenset

def metadata(rows,partition):
    require(type(rows) is list and bool(rows),"METADATA_ROWS")
    out=[]
    for x in rows:
        require(type(x) is dict and set(x)=={"parent","source_group","partition"}
                and x["partition"]==partition
                and all(type(x[k]) is str and bool(x[k].strip()) for k in ("parent","source_group")),
                "METADATA_SCHEMA")
        out.append((x["parent"],x["source_group"]))
    require(len({p for p,g in out})==len(out),"DUPLICATE_PARENT")
    return out

def prior_pilot_groups(train_meta):
    """Reproduce the exact MASTER01E V4 train sample from metadata alone."""
    tr=metadata(train_meta,"train")
    seen=set()
    for parent,g in sorted(tr,key=lambda x:(hashlib.sha256(
            (str(SEED)+":"+x[0]).encode()).digest(),x[0])):
        seen.add(g)
        if len(seen)==128:break
    require(len(seen)==128,"PRIOR_PILOT_GROUP_COUNT")
    return frozenset(seen)

def plan(train_meta,dev_meta,*,calibration_groups,additional_exposed_fit_groups):
    """Split existing train only; existing dev remains observed development selection.
    Group assignment label-blind and independent of lengths/scores/answers.
    No test or holdout metadata accepted by this API.
    """
    tr=metadata(train_meta,"train");dv=metadata(dev_meta,"dev")
    groups={g for p,g in tr}
    require(not(groups&{g for p,g in dv} or {p for p,g in tr}&{p for p,g in dv}),
            "SOURCE_OVERLAP")
    require(type(additional_exposed_fit_groups) is frozenset
            and additional_exposed_fit_groups<=groups,"EXPOSURE_INVENTORY")
    exposed=prior_pilot_groups(train_meta)|additional_exposed_fit_groups
    eligible=groups-exposed
    require(type(calibration_groups) is int and 2<=calibration_groups<=min(len(eligible),len(groups)-2),
            "CALIBRATION_GROUP_COUNT")
    cal=set(sorted(eligible,key=lambda g:(key(g),g))[:calibration_groups])
    result=Plan(tuple(sorted([(p,g,"calibration" if g in cal else "fit") for p,g in tr]
                            +[(p,g,"selection") for p,g in dv])),exposed)
    validate_plan(result)
    return result

def validate_plan(p):
    require(type(p) is Plan and type(p.members) is tuple and bool(p.members),"PLAN")
    seen=set();groups={};counts={k:set() for k in PARTITIONS}
    for x in p.members:
        require(type(x) is tuple and len(x)==3 and all(type(v) is str and v for v in x)
                and x[2] in PARTITIONS,"PLAN_MEMBER")
        parent,g,k=x
        require(parent not in seen and groups.get(g,k)==k,"PLAN_OVERLAP")
        seen.add(parent);groups[g]=k;counts[k].add(g)
    require(all(len(v)>=2 for v in counts.values()),"PLAN_GROUP_COUNT")
    require(type(p.exposed_fit_groups) is frozenset and p.exposed_fit_groups<=counts["fit"],
            "EXPOSED_CALIBRATION_GROUP")

def check_rows(p,rows,partition):
    validate_plan(p)
    require(partition in PARTITIONS,"PARTITION")
    b.validate_pairs(rows,partition)
    expected={(a,g) for a,g,k in p.members if k==partition}
    require({(x["parent"],x["source_group"]) for x in rows}==expected,"PARTITION_MEMBERSHIP")

def evidence_only(answer,context):
    a=set(b.words(answer));c=set(b.words(context)) if context.strip() else set()
    require(bool(a),"EMPTY_ANSWER")
    return [len(a&c)/len(a),len(a&c)/max(1,len(a|c))]

def lexical(row,mode):
    require(type(row) is dict and set(row)=={"question","answer","context"}
            and all(type(v) is str for v in row.values()),"LEXICAL_ROW")
    if mode=="evidence_only":return evidence_only(row["answer"],row["context"])
    if mode=="question_answer_overlap":
        a=set(b.words(row["answer"]));q=set(b.words(row["question"]))
        return [len(a&q)/max(1,len(a))]
    if mode=="legacy_lexical":return b.features(row,"evidence_lexical")
    require(mode in ("length","style","answer_only"),"LEXICAL_MODE")
    return b.features(row,mode)

def neural_map(rows,expected):
    require(type(rows) is list and len(rows)==len(expected),"NEURAL_ALIGNMENT")
    out={}
    for row in rows:
        require(type(row) in (tuple,list) and len(row)==3,"NEURAL_ROW")
        g,p,fs=row
        require(type(g) is str and type(p) is str and (g,p) in expected and (g,p) not in out
                and type(fs) in (tuple,list) and len(fs)==2,"NEURAL_ALIGNMENT")
        require(all(type(f) in (tuple,list) and len(f)==6 and all(prob(v) for v in f)
                    for f in fs),"NEURAL_FEATURES")
        out[g,p]=fs
    require(set(out)==expected,"NEURAL_ALIGNMENT")
    return out

def design(p,rows,partition,condition,mode,neural=None):
    check_rows(p,rows,partition)
    require(condition in CONDITIONS and mode in MODES,"DESIGN")
    expected={(x["source_group"],x["parent"]) for x in rows}
    needs=mode in ("neural","hybrid","hybrid_legacy")
    if needs:
        require(type(neural) is dict and set(neural)==set(CONDITIONS),"NEURAL_CONDITIONS")
        nm=neural_map(neural[condition],expected)
    else:require(neural is None,"UNEXPECTED_NEURAL")
    donors={}
    if condition=="permuted":
        donors={x["parent"]:x["context"] for x in b.context_permutation(rows,partition,SEED)}
    out=[]
    for x in sorted(rows,key=lambda x:(x["source_group"],x["parent"])):
        g,parent=x["source_group"],x["parent"]
        context=x["context"] if condition=="full" else "" if condition=="removed" else donors[parent]
        fs=[]
        for i,role in enumerate(ROLES):
            row={"question":x["question"],"answer":x["answers"][role],"context":context}
            f=list(nm[g,parent][i]) if needs else lexical(row,mode)
            if mode=="hybrid":f+=lexical(row,"evidence_only")
            if mode=="hybrid_legacy":f+=lexical(row,"legacy_lexical")
            require(bool(f) and all(math.isfinite(v) for v in f),"DESIGN_FEATURES")
            fs.append(f)
        out.append((g,parent,tuple(fs)))
    return out

@dataclass(frozen=True)
class Model:
    mode:str
    trained_condition:str
    plan:Plan
    head:tuple

def fit(p,rows,mode,*,condition="full",neural=None):
    d=design(p,rows,"fit",condition,mode,neural)
    head=fit_head([(f,y) for g,parent,fs in d for y,f in enumerate(fs)])
    return Model(mode,condition,p,tuple(tuple(x) for x in head))

def score(model,p,rows,partition,*,condition="full",neural=None):
    require(type(model) is Model and model.plan==p and partition in ("selection","calibration"),
            "SCORE_SCOPE")
    d=design(p,rows,partition,condition,model.mode,neural)
    return [(g,parent,tuple(predict_head(model.head,f) for f in fs)) for g,parent,fs in d]

def equal_length_pairs(rows,scores):
    """Descriptive prespecified within-parent exact normalized word-count tie only.
    Never use subset labels to fit/select a model. No small-cell public export.
    """
    expected={(x["source_group"],x["parent"]) for x in rows}
    aligned_scores(scores,expected)
    keep={(x["source_group"],x["parent"]) for x in rows
          if len(b.words(x["answers"][ROLES[0]]))==len(b.words(x["answers"][ROLES[1]]))}
    return [x for x in scores if x[:2] in keep]

def aligned_scores(scores,expected):
    require(type(scores) is list and len(scores)==len(expected),"SCORE_ALIGNMENT")
    seen=set()
    for x in scores:
        require(type(x) is tuple and len(x)==3 and type(x[0]) is str and type(x[1]) is str
                and x[:2] in expected and x[:2] not in seen
                and type(x[2]) is tuple and len(x[2])==2 and all(prob(v) for v in x[2]),"SCORE_ALIGNMENT")
        seen.add(x[:2])
    return {x[:2]:x for x in scores}

def compare_family(p,partition,scores,contrasts,*,replicates=1000):
    """Fixed aligned paired contrasts; common source-cluster draws across family.
    Descriptive percentile intervals only; no confirmatory p-values or winner.
    Positive differences favor first method for AP/ranking; negative for Brier.
    """
    validate_plan(p);require(partition=="selection","COMPARISON_SCOPE")
    expected={(g,a) for a,g,k in p.members if k==partition}
    require(type(scores) is dict and bool(scores) and type(contrasts) is tuple and bool(contrasts)
            and len(set(contrasts))==len(contrasts) and all(type(x) is tuple and len(x)==2
            and x[0]!=x[1] and all(k in scores for k in x) for x in contrasts),"CONTRASTS")
    require(type(replicates) is int and 100<=replicates<=10000,"REPLICATES")
    maps={k:aligned_scores(v,expected) for k,v in scores.items()}
    groups={}
    for g,a in sorted(expected):groups.setdefault(g,[]).append((g,a))
    keys=sorted(groups);require(len(keys)>=2,"GROUPS")
    metrics=("average_precision","paired_ranking","brier")
    def evaluate(ids):
        m={k:b.metrics([v[i] for i in ids]) for k,v in maps.items()}
        return [[m[a][z]-m[c][z] for z in metrics] for a,c in contrasts]
    obs=evaluate(sorted(expected))
    dist=[[[] for z in metrics] for x in contrasts]
    rng=random.Random(SEED)
    for _ in range(replicates):
        d=evaluate([i for g in rng.choices(keys,k=len(keys)) for i in groups[g]])
        for i,row in enumerate(d):
            for j,v in enumerate(row):dist[i][j].append(v)
    out=[]
    for i,(a,c) in enumerate(contrasts):
        m={}
        for j,z in enumerate(metrics):
            v=sorted(dist[i][j]);m[z]={"difference":obs[i][j],
                   "percentile_95":[v[int(.025*(replicates-1))],v[int(.975*(replicates-1))]]}
        out.append({"first":a,"second":c,"metrics":m})
    return {"groups":len(keys),"parents":len(expected),"replicates":replicates,
            "contrasts":out,"exploratory":True,"multiplicity_adjusted":False,
            "public_export_approved":False}
