"""Matched-input baseline policies and dependence stress tests; synthetic only."""
from pirc_v2_method import joint, independent_joint, choose_action, require, probability

def feasible_joint(p0,p1,harm):
    require(all(probability(v) for v in (p0,p1,harm)),"MARGINALS")
    require(max(0.,p0-p1)-1e-12<=harm<=min(p0,1-p1)+1e-12,"FRECHET")
    return {k:max(0.,v) for k,v in {"00":1-p1-harm,"01":p1-p0+harm,
        "10":harm,"11":p0-harm}.items()}

def baseline_actions(p,*,changed,confidence_threshold=.7,release_penalty=1.,
                     preservation_penalty=3.,coverage_bonus=0.):
    p=joint(p)
    require(type(changed) is bool and probability(confidence_threshold),"BASELINE_INPUT")
    require(changed or (p["01"]==0 and p["10"]==0),"UNCHANGED")
    p0=p["10"]+p["11"];p1=p["01"]+p["11"]
    args=dict(release_penalty=release_penalty,preservation_penalty=preservation_penalty,
              coverage_bonus=coverage_bonus)
    independent=independent_joint(p0,p1) if changed else p
    return {"always_accept":"accept","always_revise":"revise",
        "confidence_abstain":"accept" if p0>=confidence_threshold else "abstain",
        "marginal_selector":"revise" if p1>p0 else "accept",
        "independence_policy":choose_action(independent,**args)[0],
        "joint_policy":choose_action(p,**args)[0]}

def stress_test(p0,p1,**args):
    """Endpoint comparison at Frechet bounds; same marginals, different dependence.
    An algebraic counterexample, not an estimate of empirical improvement.
    """
    low=max(0.,p0-p1);high=min(p0,1-p1)
    a=feasible_joint(p0,p1,low);b=feasible_joint(p0,p1,high)
    return {"low_harm":low,"high_harm":high,
        "low_action":choose_action(a,**args)[0],
        "high_action":choose_action(b,**args)[0]}

def calibration_inventory(groups):
    """Count metadata-only synthetic inventory, reject cross-partition source overlap.
    Does not authenticate actual source identities, rights, or independence.
    """
    require(type(groups) is list,"INVENTORY")
    seen={};counts=dict.fromkeys(("train","tune","calibration","test"),0)
    for g in groups:
        require(type(g) is dict and set(g)=={"source_group","partition","eligible"}
            and type(g["source_group"]) is str and bool(g["source_group"])
            and g["partition"] in counts and type(g["eligible"]) is bool,"INVENTORY_ROW")
        key=g["source_group"]
        require(key not in seen,"DUPLICATE_SOURCE_GROUP")
        seen[key]=g["partition"]
        counts[g["partition"]]+=int(g["eligible"])
    return {"eligible_group_counts":counts,"rights_verified":False,
            "independence_verified":False,"execution_authorized":False}


def _sigmoid(z):
    from math import exp
    return 1/(1+exp(-z)) if z>=0 else exp(z)/(1+exp(z))

def _binary_fit(xs,ys,steps,rate,l2):
    require(bool(xs) and len(xs)==len(ys),"EMPTY_TRAIN")
    w=[0.0]*len(xs[0])
    for _ in range(steps):
        grad=[0.0]*len(w)
        for x,y in zip(xs,ys):
            err=_sigmoid(sum(a*b for a,b in zip(w,x)))-y
            for j,v in enumerate(x): grad[j]+=err*v
        for j in range(len(w)): w[j]-=rate*(grad[j]/len(xs)+(l2*w[j] if j else 0))
    return w

def fit_comparators(rows,targets,*,steps=400,rate=.1,l2=.01):
    """Independent marginal + direct harm heads, conditional factorization and
    constant empirical transition control. Same training features and labels
    as joint softmax; tune hyperparameters on tune only. No split authorization.
    """
    from pirc_v2_method import features,STATES
    from math import isfinite
    require(type(rows) is list and type(targets) is list and len(rows)==len(targets)>0
            and all(t in STATES for t in targets),"TRAIN_ALIGNMENT")
    require(type(steps) is int and 1<=steps<=10000 and type(rate) in (float,int)
            and isfinite(rate) and 0<rate<=1 and type(l2) in (float,int)
            and isfinite(l2) and 0<=l2<=1,"TRAIN_PARAMETERS")
    xs=[[1.]+features(r) for r in rows]
    require(all(r["answer_changed"] or t in ("00","11") for r,t in zip(rows,targets)),
            "UNCHANGED_TRANSITION")
    # Conditional head predicts revised correctness given original correctness;
    # not supplied the actual original correctness at inference.
    changed=[i for i,r in enumerate(rows) if r["answer_changed"]]
    def head(indices,fn):
        require(bool(indices),"MISSING_TRAIN_STRATUM")
        return _binary_fit([xs[i] for i in indices],[int(fn(targets[i])) for i in indices],
                           steps,rate,l2)
    all_i=list(range(len(xs)))
    correct=[i for i in changed if targets[i][0]=="1"]
    wrong=[i for i in changed if targets[i][0]=="0"]
    return {"p0":head(all_i,lambda t:t[0]=="1"),
            "p1":head(all_i,lambda t:t[1]=="1"),
            "harm":head(changed,lambda t:t=="10"),
            "conditional_repair":head(wrong,lambda t:t[1]=="1"),
            "conditional_preserve":head(correct,lambda t:t[1]=="1"),
            "constant_repair":sum(targets[i]=="01" for i in wrong)/len(wrong),
            "constant_preserve":sum(targets[i]=="11" for i in correct)/len(correct)}

def predict_comparators(model,row):
    from pirc_v2_method import features
    x=[1.]+features(row)
    def pred(k): return _sigmoid(sum(a*b for a,b in zip(model[k],x)))
    p0=pred("p0")
    if not row["answer_changed"]:
        q={"00":1-p0,"01":0.,"10":0.,"11":p0}
        return {k:q.copy() for k in ("independent_trained","direct_harm","conditional_factorization","constant_transition")}
    p1=pred("p1")
    harm=max(max(0.,p0-p1),min(min(p0,1-p1),pred("harm")))
    def cond(repair,preserve):
        return {"00":(1-p0)*(1-repair),"01":(1-p0)*repair,
                "10":p0*(1-preserve),"11":p0*preserve}
    return {"independent_trained":independent_joint(p0,p1),
            "direct_harm":feasible_joint(p0,p1,harm),
            "conditional_factorization":cond(pred("conditional_repair"),pred("conditional_preserve")),
            "constant_transition":cond(model["constant_repair"],model["constant_preserve"])}

def enforce_valid_action(action,*,original_valid,revised_valid):
    """Malformed answer is never released. No gold used."""
    require(action in ("accept","revise","abstain") and type(original_valid) is bool
            and type(revised_valid) is bool,"VALID_ACTION")
    if (action=="accept" and not original_valid) or (action=="revise" and not revised_valid):
        return "abstain"
    return action
