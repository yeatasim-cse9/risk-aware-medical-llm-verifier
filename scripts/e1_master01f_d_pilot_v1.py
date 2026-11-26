"""01F-D v1: explicit two-partition suite; no calibration dependency or calls.
Reuses 01F-B's fixed heads and feature semantics with a separate two-partition plan.
"""
import math,random,time
from dataclasses import dataclass
import e1_master01f_b_v1 as f
import e1_master01f_d_inputs_v1 as inp
import e1_master01e_v4_budget as budget
import e1_master01e_v4_neural as n
from e1_master01e_v2_pilot import collect,fit_head,predict_head
b=f.b
require=f.require
MODES=f.MODES
CONDITIONS=f.CONDITIONS
ROLES=f.ROLES
SEED=f.SEED
neural_map=f.neural_map
lexical=f.lexical
prob=f.prob
aligned_scores=f.aligned_scores

@dataclass(frozen=True)
class Plan:
    members:tuple

def validate_plan(p):
    require(type(p) is Plan and type(p.members) is tuple,"PLAN")
    parents=set();groups=set();counts=dict.fromkeys(inp.LIMITS,0)
    for x in p.members:
        require(type(x) is tuple and len(x)==3 and all(inp.valid_hash(v) for v in x[:2])
                and x[2] in counts,"PLAN_MEMBER")
        a,g,k=x
        require(a not in parents and g not in groups,"PLAN_OVERLAP")
        parents.add(a);groups.add(g);counts[k]+=1
    require(counts==inp.LIMITS,"PLAN_COUNTS")

def check_rows(p,rows,partition):
    validate_plan(p)
    require(partition in inp.LIMITS,"PARTITION")
    b.validate_pairs(rows,partition)
    require({(x["parent"],x["source_group"]) for x in rows}==
            {(a,g) for a,g,k in p.members if k==partition},"PARTITION_MEMBERSHIP")

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
    require(type(model) is Model and model.plan==p and partition=="selection",
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

# Candidate list and contrasts fixed before the next development run.
# Existing dev has already been observed; this family remains exploratory.
SUITE_CONTRASTS=(
    ("hybrid/full","neural/full"),
    ("hybrid/full","legacy_lexical/full"),
    ("hybrid/full","full_lexical/full"),
    ("hybrid/full","evidence_only/full"),
    ("hybrid_legacy/full","hybrid/full"),
    ("neural/full","neural/removed"),
    ("neural/full","neural/permuted"),
    ("evidence_only/full","evidence_only/removed"),
    ("evidence_only/full","evidence_only/permuted"),
    ("legacy_lexical/full","legacy_lexical/removed"),
    ("legacy_lexical/full","legacy_lexical/permuted"),
    ("hybrid/full","hybrid/removed"),
    ("hybrid/full","hybrid/permuted"),
    ("neural/full","neural/removed_retrained"),
    ("legacy_lexical/full","legacy_lexical/removed_retrained"),
    ("hybrid/full","hybrid/removed_retrained"),
)


def audit(engine,p,rows,partition):
    check_rows(p,rows,partition)
    donor={x["parent"]:x["context"] for x in b.context_permutation(rows,partition,SEED)}
    vocab=engine.model.config.vocab_size
    require(type(vocab) is int and vocab>0 and engine.special==3,"MODEL_CONTRACT")
    costs={k:dict(forward_examples=0,padded_positions=0,active_tokens=0,real_windows=0)
           for k in CONDITIONS}
    for row in rows:
        full=budget.checked_windows(engine,row["context"],vocab)
        perm=budget.checked_windows(engine,donor[row["parent"]],vocab)
        for role in ROLES:
            answer=row["answers"][role];hyp=engine.tokens(answer)
            for k,parts in (("full",full),("removed",[]),("permuted",perm)):
                for mode in MODES:
                    if mode not in ("neural","hybrid","hybrid_legacy"):
                        vals=lexical({"question":row["question"],"answer":answer,
                            "context":row["context"] if k=="full" else donor[row["parent"]] if k=="permuted" else ""},mode)
                        require(all(math.isfinite(v) for v in vals),"FEATURES")
                slots=parts+[[] for _ in range(n.SLOTS-len(parts))]
                active=sum(budget.checked_pair(engine,x,hyp,vocab) for x in slots)
                c=costs[k];c["forward_examples"]+=n.SLOTS;c["padded_positions"]+=n.SLOTS*n.MAX_LEN
                c["active_tokens"]+=active;c["real_windows"]+=len(parts)
    return costs

def suite(p,tr,dv,tn,dn,replicates):
    models={};scores={}
    for mode in MODES:
        needs=mode in ("neural","hybrid","hybrid_legacy")
        model=fit(p,tr,mode,neural=tn if needs else None);models[mode]=model
        conditions=CONDITIONS if mode in ("neural","hybrid","hybrid_legacy",
                    "evidence_only","legacy_lexical","full_lexical") else ("full",)
        for k in conditions:
            scores[mode+"/"+k]=score(model,p,dv,"selection",condition=k,neural=dn if needs else None)
        if mode in ("neural","hybrid","legacy_lexical"):
            refit=fit(p,tr,mode,condition="removed",neural=tn if needs else None)
            scores[mode+"/removed_retrained"]=score(refit,p,dv,"selection",condition="removed",neural=dn if needs else None)
    for name,fn in (("nli_contradiction",lambda x:x[1]),("nli_non_entailment",lambda x:1-x[0])):
        scores[name+"/full"]=[(g,a,tuple(fn(x) for x in fs)) for g,a,fs in dn["full"]]
    comparisons=compare_family(p,"selection",scores,SUITE_CONTRASTS,replicates=replicates)
    tied={k:equal_length_pairs(dv,v) for k,v in scores.items()}
    length={k:b.metrics(v) for k,v in tied.items()} if len(next(iter(tied.values())))>=10 else None
    return {"metrics":{k:b.metrics(v) for k,v in scores.items()},
            "comparisons":comparisons,"equal_length_metrics":length}

class BoundedEngine:
    def __init__(self,engine,deadline,maximum):
        self.engine=engine;self.deadline=deadline;self.maximum=maximum;self.used=0
    def __getattr__(self,k):return getattr(self.engine,k)
    def infer(self,*a,**kw):
        require(time.monotonic()<self.deadline and self.used+n.SLOTS<=self.maximum,"COMPUTE_LIMIT")
        self.used+=n.SLOTS
        return self.engine.infer(*a,**kw)

def run(engine,p,blocks,*,authorized,deadline,mark=lambda _:None,replicates=1000):
    require(authorized is True,"AUTHORIZATION_REQUIRED")
    require(type(blocks) is dict and set(blocks)==set(inp.LIMITS),"PARTITION_SCOPE")
    require(replicates==1000,"REPLICATES")
    tr,dv=blocks["fit"],blocks["selection"]
    check_rows(p,tr,"fit");check_rows(p,dv,"selection")
    mark("AUDIT_FIT");tc=audit(engine,p,tr,"fit")
    mark("AUDIT_SELECTION");dc=audit(engine,p,dv,"selection")
    limit=sum(inp.LIMITS.values())*2*3*n.SLOTS
    e=BoundedEngine(engine,deadline,limit)
    mark("INFER_FIT");tn,actual=collect(e,tr,"fit")
    require(actual==tc,"COST_RECONCILIATION")
    mark("INFER_SELECTION");dn,actual=collect(e,dv,"selection")
    require(actual==dc and e.used==limit,"COST_RECONCILIATION")
    for rows,features in ((tr,tn),(dv,dn)):
        expected={(x["source_group"],x["parent"]) for x in rows}
        for k in CONDITIONS:neural_map(features[k],expected)
    require(time.monotonic()<deadline,"COMPUTE_LIMIT")
    mark("METRICS");result=suite(p,tr,dv,tn,dn,replicates)
    require(time.monotonic()<deadline,"COMPUTE_LIMIT")
    return {**result,"budget":{"fit":tc,"selection":dc}}
