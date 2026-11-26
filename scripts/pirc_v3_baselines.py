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
