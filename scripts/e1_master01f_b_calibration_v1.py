"""Fixed-family selective release calibration for benchmark roles only.
One parent representative per source group, chosen independently of labels/scores.
A released group is erroneous if any accepted answer has hallucinated role.
This is group-level risk, not individual clinical error or correction efficacy.
"""
from dataclasses import dataclass
from pirc_calibration_prototype import upper_binomial
import e1_master01f_b_v1 as f

# A small prespecified candidate family. No adaptation to calibration outcomes.
THRESHOLDS=(0.05,0.10,0.20,0.30,0.40,0.50)
@dataclass(frozen=True)
class Policy:
    model:f.Model
    alpha:float
    risk_cap:float
    coverage_floor:float
    thresholds:tuple=THRESHOLDS

def freeze(model,*,alpha=.05,risk_cap=.10,coverage_floor=.50):
    f.require(type(model) is f.Model and model.trained_condition=="full","MODEL")
    f.validate_plan(model.plan)
    f.require(all(f.prob(x) and 0<x<1 for x in (alpha,risk_cap,coverage_floor)),"TARGETS")
    return Policy(model,alpha,risk_cap,coverage_floor)

def representatives(plan):
    f.validate_plan(plan)
    by={}
    for p,g,k in plan.members:
        if k=="calibration":by.setdefault(g,[]).append(p)
    return {(g,min(ps,key=lambda p:(f.key(p),p))) for g,ps in by.items()}

def calibration(policy,plan,rows,*,neural=None):
    """Scores computed with a previously frozen train-only head; no fitting here.
    Fixed family, two one-sided bounds per threshold with Bonferroni.
    Conditional binomial inference assumes iid source-group representatives,
    fixed model and selection independent of calibration scores/labels.
    Those assumptions are not certified by this function.
    """
    f.require(type(policy) is Policy and policy.model.plan==plan
              and policy.thresholds==THRESHOLDS,"FROZEN_POLICY")
    alpha,risk_cap,coverage_floor=policy.alpha,policy.risk_cap,policy.coverage_floor
    f.require(all(f.prob(x) and 0<x<1 for x in (alpha,risk_cap,coverage_floor)),"TARGETS")
    scores=f.score(policy.model,plan,rows,"calibration",condition="full",neural=neural)
    expected={(g,p) for p,g,k in plan.members if k=="calibration"}
    f.aligned_scores(scores,expected)
    chosen=representatives(plan)
    sample=[x[2] for x in scores if x[:2] in chosen]
    n=len(sample)
    f.require(n==len(chosen) and n>=2,"CALIBRATION_REPRESENTATIVES")
    delta=alpha/(2*len(THRESHOLDS))
    result=[]
    for t in THRESHOLDS:
        released=sum(gt<=t or ha<=t for gt,ha in sample)
        errors=sum(ha<=t for gt,ha in sample)
        answers=sum(int(gt<=t)+int(ha<=t) for gt,ha in sample)
        u=upper_binomial(errors,released,delta)
        l=max(0.,1-upper_binomial(n-released,n,delta))
        result.append({"threshold":t,"released_groups":released,"error_groups":errors,
            "accepted_answers":answers,"group_risk_upper":u,"group_coverage_lower":l,
            "answer_coverage_descriptive":answers/(2*n),
            "meets_bounds_under_assumptions":released>0 and u<=risk_cap and l>=coverage_floor})
    passing=[x for x in result if x["meets_bounds_under_assumptions"]]
    best=min(passing,key=lambda x:(-x["released_groups"],x["group_risk_upper"],x["threshold"])) if passing else None
    return {"groups":n,"per_bound_delta":delta,"policies":result,
            "selected_threshold":best["threshold"] if best else None,
            "status":"BOUNDS_MET_UNDER_ASSUMPTIONS" if best else "NO_POLICY_MEETS_BOUNDS",
            "endpoint":"ANY_HALLUCINATED_ROLE_ACCEPTED_PER_RELEASED_GROUP",
            "clinical_safety_certified":False,"revision_evaluated":False,
            "public_export_approved":False}
