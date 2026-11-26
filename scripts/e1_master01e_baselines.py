"""Deterministic E1 benchmark-role baselines, no data I/O or model downloads.
Positive = hallucinated answer ROLE, not clinically confirmed factual error.
Synthetic-tested pure Python prototypes; train-only fitting.
"""
from collections import Counter
import hashlib,math,re,random
from e1_master01d_v2_integrity import require,normalize

MODES=("length","style","answer_only","question_only","question_answer","evidence_lexical","full_lexical")
DIM=256
def words(s):
    require(type(s) is str and bool(s.strip()),"TEXT")
    return normalize(s).split()
def hashed(tokens,prefix):
    v=[0.]*DIM
    for t in tokens:
        h=hashlib.sha256((prefix+":"+t).encode()).digest()
        v[int.from_bytes(h[:4],"big")%DIM]+=1 if h[4]&1 else -1
    norm=math.sqrt(sum(x*x for x in v)) or 1
    return [x/norm for x in v]
def lexical(q,a,c):
    q,a,c=set(q),set(a),set(c)
    return [len(a&c)/max(1,len(a)),len(q&c)/max(1,len(q)),
            len(a&q)/max(1,len(a)),len(a&c)/max(1,len(a|c))]
def features(x,mode):
    require(type(x) is dict and set(x)=={"question","answer","context"} and mode in MODES,"INPUT_SCHEMA")
    q,a,c=map(words,(x["question"],x["answer"],x["context"]))
    length=[math.log1p(len(a))]
    style=length+[math.log1p(len(x["answer"])),len(set(a))/len(a),
        sum(t.isdigit() for t in a)/len(a),sum(t in {"no","not","never","none"} for t in a)/len(a),
        sum(ch in ".,;:!?" for ch in x["answer"])/max(1,len(x["answer"]))]
    ans=hashed(a,"a")+hashed([u+" "+v for u,v in zip(a,a[1:])],"ab")
    if mode=="length":return length
    if mode=="style":return style
    if mode=="answer_only":return ans
    if mode=="question_only":return hashed(q,"q")
    qa=hashed(q,"q")+ans
    if mode=="question_answer":return qa
    ev=lexical(q,a,c)
    return ev if mode=="evidence_lexical" else qa+style+ev
def sigmoid(x):
    return 1/(1+math.exp(-x)) if x>=0 else math.exp(x)/(1+math.exp(x))
def validate_pairs(pairs,partition):
    require(type(pairs) is list and bool(pairs),"EMPTY_PAIRS")
    seen=set()
    for p in pairs:
        require(type(p) is dict and set(p)=={"parent","source_group","partition","question","context","answers"} and
            all(type(p[k]) is str and bool(p[k].strip()) for k in ("parent","source_group","question","context"))
            and p["partition"]==partition and p["parent"] not in seen,"PAIR_SCHEMA")
        seen.add(p["parent"])
        require(type(p["answers"]) is dict and set(p["answers"])=={"ground_truth","hallucinated"} and
                all(type(a) is str and bool(a.strip()) for a in p["answers"].values()),"ANSWER_ROLES")
def rows(pairs):
    for p in pairs:
        for role in ("ground_truth","hallucinated"):
            yield {"question":p["question"],"context":p["context"],"answer":p["answers"][role]},int(role=="hallucinated")
def fit(pairs,mode,*,steps=100,rate=.1,l2=.01):
    validate_pairs(pairs,"train")
    require(mode in MODES and type(steps) is int and 1<=steps<=1000 and
        0<rate<=1 and 0<=l2<=1,"FIT_PARAMS")
    data=[(features(x,mode),y) for x,y in rows(pairs)]
    n=len(data);m=len(data[0][0])
    mean=[sum(x[j] for x,y in data)/n for j in range(m)]
    sd=[math.sqrt(sum((x[j]-mean[j])**2 for x,y in data)/n) or 1 for j in range(m)]
    data=[([1.]+[(v-a)/b for v,a,b in zip(x,mean,sd)],y) for x,y in data]
    w=[0.]*(m+1)
    for _ in range(steps):
        g=[0.]*len(w)
        for x,y in data:
            error=sigmoid(sum(a*b for a,b in zip(w,x)))-y
            for j,z in enumerate(x):g[j]+=error*z
        for j in range(len(w)):w[j]-=rate*(g[j]/n+(l2*w[j] if j else 0))
    return {"mode":mode,"mean":mean,"sd":sd,"weights":w}
def predict(model,x):
    f=features(x,model["mode"])
    require(len(f)==len(model["mean"])==len(model["sd"])==len(model["weights"])-1,"MODEL_SCHEMA")
    z=[1.]+[(v-a)/b for v,a,b in zip(f,model["mean"],model["sd"])]
    return sigmoid(sum(a*b for a,b in zip(z,model["weights"])))
def disjoint(train,dev,test=None):
    blocks=[(train,"train"),(dev,"dev")]+([(test,"test")] if test is not None else [])
    groups=set();parents=set()
    for p,s in blocks:
        validate_pairs(p,s)
        g={x["source_group"] for x in p};k={x["parent"] for x in p}
        require(not(g&groups or k&parents),"CROSS_PARTITION_OVERLAP")
        groups.update(g);parents.update(k)
def context_permutation(pairs,partition,seed=20261126):
    """Label-blind deterministic cyclic derangement. Same-partition only.
    Fails if no cyclic shift avoids same source; not proof unrelated evidence.
    """
    validate_pairs(pairs,partition)
    a=sorted(pairs,key=lambda p:hashlib.sha256((str(seed)+":"+p["parent"]).encode()).digest())
    n=len(a)
    shift=next((s for s in range(1,n) if all(a[i]["source_group"]!=a[(i+s)%n]["source_group"] for i in range(n))),None)
    require(shift is not None,"NO_CROSS_SOURCE_PERMUTATION")
    return [{**p,"context":a[(i+shift)%n]["context"]} for i,p in enumerate(a)]
def score_pairs(model,pairs,partition):
    validate_pairs(pairs,partition)
    return [(p["source_group"],tuple(predict(model,x) for x,y in rows([p]))) for p in pairs]
def metrics(scored):
    """AP positive hallucinated role. Tie-grouped AP, paired ranking and Brier."""
    require(bool(scored),"EMPTY_SCORES")
    xy=[];rank=0
    for group,(gt,ha) in scored:
        require(all(type(x) in (int,float) and math.isfinite(x) and 0<=x<=1 for x in (gt,ha)),"SCORE")
        xy.extend([(gt,0),(ha,1)]);rank+=int(ha>gt)+.5*int(ha==gt)
    xy.sort(reverse=True);i=tp=0;ap=0.;n=len(scored)
    while i<len(xy):
        j=i
        while j<len(xy) and xy[j][0]==xy[i][0]:j+=1
        positives=sum(y for x,y in xy[i:j]);tp+=positives
        ap+=positives/n*tp/j;i=j
    return {"parents":n,"average_precision":ap,"paired_ranking":rank/n,
            "brier":sum((x-y)**2 for x,y in xy)/(2*n)}
def paired_bootstrap(proposed,baseline,*,replicates=1000,seed=20261126):
    """Aligned scores in same private parent order; resample source groups.
    Percentile CI descriptive unless protocol assumptions established.
    """
    require(type(replicates) is int and 100<=replicates<=10000 and len(proposed)==len(baseline)>0,"BOOTSTRAP_INPUT")
    groups={}
    for a,b in zip(proposed,baseline):
        require(a[0]==b[0],"GROUP_ALIGNMENT")
        groups.setdefault(a[0],[]).append((a,b))
    keys=sorted(groups);require(len(keys)>=2,"INSUFFICIENT_GROUPS")
    rng=random.Random(seed);diff=[]
    for _ in range(replicates):
        pairs=[p for k in rng.choices(keys,k=len(keys)) for p in groups[k]]
        diff.append(metrics([p[0] for p in pairs])["average_precision"]-metrics([p[1] for p in pairs])["average_precision"])
    diff.sort()
    return {"groups":len(keys),"replicates":replicates,
        "ap_difference":metrics(proposed)["average_precision"]-metrics(baseline)["average_precision"],
        "percentile_95":[diff[int(.025*(replicates-1))],diff[int(.975*(replicates-1))]]}
