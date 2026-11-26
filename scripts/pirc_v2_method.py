"""PIRC V2 candidate decision rule and fixed-family feasibility tests.

Synthetic/design-only. No dataset access, no generator, no clinical guarantee.
Joint probabilities must be learned on source-disjoint training data only.
Calibration is for fixed policies on iid source representatives; neither
exchangeability alone nor a selected benchmark establishes deployment validity.
"""
from math import isfinite, exp, log
from pirc_calibration_prototype import policy_summary

STATES=("00","01","10","11")
FEATURES=("original_support","original_contradiction","original_evidence_coverage",
          "original_support_drop","original_confidence","revised_support",
          "revised_contradiction","revised_evidence_coverage","revised_support_drop",
          "revised_confidence","answer_changed","evidence_available")

def require(ok,code):
    if not ok: raise ValueError(code)

def probability(x):
    return type(x) in (int,float) and isfinite(x) and 0<=x<=1

def joint(x):
    require(type(x) is dict and set(x)==set(STATES) and
            all(probability(v) for v in x.values()) and abs(sum(x.values())-1)<=1e-9,
            "JOINT_PROBABILITIES")
    return x

def features(row):
    """Fixed proposed features, no labels. Support drops clipped to [0,1].
    Include contrasts and the product of answer-conditioned support drops.
    Missing evidence must be explicitly flagged, not silently imputed as support.
    """
    require(type(row) is dict and set(row)==set(FEATURES) and
            all(probability(v) for v in row.values()),"FEATURE_SCHEMA")
    require(row["answer_changed"] in (0,1) and row["evidence_available"] in (0,1),
            "FEATURE_INDICATORS")
    if not row["evidence_available"]:
        require(all(row[f]==0 for f in FEATURES if f.endswith(
            ("_support","_contradiction","_evidence_coverage","_support_drop"))),
            "MISSING_EVIDENCE_ENCODING")
    a=[row[f] for f in FEATURES]
    return a+[a[i+5]-a[i] for i in range(5)]+[a[3]*a[8]]

def fit_joint(rows,targets,*,steps=400,rate=.1,l2=.01):
    """Deterministic multinomial logistic prototype. Labels (Y0,Yrev) train only.
    No split enforcement here: upstream source-disjoint protocol is mandatory.
    Use same original/revised candidates for all compared policies.
    """
    require(type(rows) is list and type(targets) is list and
            len(rows)==len(targets)>0 and all(t in STATES for t in targets),"TRAIN_ALIGNMENT")
    require(type(steps) is int and 1<=steps<=10000 and type(rate) in (int,float)
            and isfinite(rate) and 0<rate<=1 and type(l2) in (int,float)
            and isfinite(l2) and 0<=l2<=1,"TRAIN_PARAMETERS")
    X=[[1.0]+features(row) for row in rows];n=len(X);m=len(X[0])
    require(all(row["answer_changed"] or t in ("00","11") for row,t in zip(rows,targets)),
            "UNCHANGED_TRANSITION")
    W=[[0.0]*m for _ in STATES]
    for _ in range(steps):
        grad=[[0.0]*m for _ in STATES]
        for x,t in zip(X,targets):
            p=_softmax([sum(a*b for a,b in zip(w,x)) for w in W],changed=bool(x[11]))
            for k,s in enumerate(STATES):
                err=p[k]-(s==t)
                for j,v in enumerate(x): grad[k][j]+=err*v
        for k in range(4):
            for j in range(m):
                W[k][j]-=rate*(grad[k][j]/n+(l2*W[k][j] if j else 0))
    return W

def _softmax(z,changed=True):
    allowed=(0,1,2,3) if changed else (0,3)
    m=max(z[i] for i in allowed);w=[exp(v-m) if i in allowed else 0.0 for i,v in enumerate(z)];s=sum(w)
    return [v/s for v in w]

def predict_joint(weights,row):
    x=[1.0]+features(row)
    require(type(weights) is list and len(weights)==4 and
            all(type(w) is list and len(w)==len(x) and
                all(type(v) in (int,float) and isfinite(v) for v in w) for w in weights),
            "WEIGHT_SCHEMA")
    return dict(zip(STATES,_softmax([sum(a*b for a,b in zip(w,x)) for w in weights],
                                     changed=bool(row["answer_changed"]))))

def independent_joint(p0,p1):
    """Same marginals, independence-imputed transition baseline."""
    require(probability(p0) and probability(p1),"MARGINAL_PROBABILITIES")
    return {"00":(1-p0)*(1-p1),"01":(1-p0)*p1,
            "10":p0*(1-p1),"11":p0*p1}

def choose_action(p,*,release_penalty,preservation_penalty,coverage_bonus,
                  risk_cap=.10):
    """Lagrangian candidate rule; scores are not certified probabilities.
    Objective candidate: correct release, with release-risk, introduced-error
    and coverage constraints. The constant terms in the last two constraints
    cancel for action selection. Abstention yields no correct release.
    """
    p=joint(p)
    require(all(type(v) in (int,float) and isfinite(v) and v>=0 for v in
        (release_penalty,preservation_penalty,coverage_bonus)) and
        probability(risk_cap) and 0<risk_cap<1,"POLICY_PARAMETERS")
    p0=p["10"]+p["11"];p1=p["01"]+p["11"]
    a=p0-release_penalty*(1-p0-risk_cap)+coverage_bonus
    b=p1-release_penalty*(1-p1-risk_cap)-preservation_penalty*p["10"]+coverage_bonus
    scores={"abstain":0.0,"accept":a,"revise":b}
    # Prefer abstain on exact zero tie, then accept to avoid gratuitous change.
    return max(scores,key=lambda k:scores[k]),scores

def binomial_pmf(n,p):
    """Mode-centered normalized recurrence; stable for planning, not exact arithmetic."""
    require(type(n) is int and 0<=n<=20000 and probability(p),"BINOMIAL_INPUT")
    if p==0: return [1.0]+[0.0]*n
    if p==1: return [0.0]*n+[1.0]
    mode=min(n,int((n+1)*p));a=[0.0]*(n+1);a[mode]=1.0
    for k in range(mode,0,-1):
        a[k-1]=a[k]*k/(n-k+1)*(1-p)/p
    for k in range(mode,n):
        a[k+1]=a[k]*(n-k)/(k+1)*p/(1-p)
    total=sum(a)
    return [v/total for v in a]

def lower_tail(n,x,p):
    require(type(x) is int and 0<=x<=n,"TAIL_INPUT")
    return min(1.0,sum(binomial_pmf(n,p)[:x+1]))

def upper_tail(n,x,p):
    require(type(x) is int and 0<=x<=n,"TAIL_INPUT")
    return min(1.0,sum(binomial_pmf(n,p)[x:]))

def certify_family(original,policies,*,alpha=.05,risk_cap=.10,
                   introduced_cap=.05,coverage_floor=.50):
    """Intersection-union p-test per fixed policy + Holm across policies.

    For each policy null = at least one target violated (boundary included).
    Max of three valid one-sided p-values is valid without independence.
    Holm controls probability of declaring any infeasible policy feasible.
    NOT simultaneous confidence bounds on all three endpoints; retain V1
    Bonferroni intervals when simultaneous numerical bounds are requested.
    """
    require(type(policies) is dict and bool(policies) and
            all(type(k) is str and bool(k) for k in policies),"POLICY_SET")
    require(all(probability(v) and 0<v<1 for v in
                (alpha,risk_cap,introduced_cap,coverage_floor)),"TARGETS")
    out={}
    for name,row in policies.items():
        require(type(row) is dict and set(row)=={"actions","final"},"POLICY_SCHEMA")
        s=policy_summary(original,row["actions"],row["final"])
        n=s["n"];a=s["released"];b=s["original_correct"]
        p_r=lower_tail(a,s["released_errors"],risk_cap) if a else 1.0
        p_i=lower_tail(b,s["introduced_errors"],introduced_cap) if b else 1.0
        p_c=upper_tail(n,a,coverage_floor)
        # Conservative floating-point guard; not a formal numerical proof.
        out[name]={**s,"p_release":p_r,"p_introduced":p_i,"p_coverage":p_c,
                   "p_infeasible":min(1.0,max(p_r,p_i,p_c)+1e-12),"passes":False}
    ordered=sorted(out,key=lambda k:(out[k]["p_infeasible"],k))
    for j,k in enumerate(ordered):
        if out[k]["p_infeasible"]>alpha/(len(ordered)-j): break
        out[k]["passes"]=True
    passing=[k for k in ordered if out[k]["passes"]]
    chosen=min(passing,key=lambda k:(-out[k]["released"],-out[k]["correct_released"],k)) if passing else None
    return {"selected":chosen,"status":"FEASIBLE_UNDER_ASSUMPTIONS" if chosen else
            "NO_CERTIFIED_POLICY","policies":out,"clinical_safety_certified":False,
            "real_experiment_executed":False}
