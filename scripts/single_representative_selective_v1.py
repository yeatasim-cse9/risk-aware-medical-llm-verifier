"""Pure fixed-family single-representative evaluation; no data or model I/O.
Synthetic regression fixtures only in repository CI. Bounds are conditional on
independent representative sampling and a prospectively fixed score/policy.
"""
from math import isfinite
from pirc_calibration_prototype import upper_binomial, require

def evaluate_fixed_family(rows, thresholds, *, alpha=.05, risk_cap=.10,
                          coverage_floor=.50):
    """Rows: (unique group key, nonfactual bool, score in [0,1]).
    Score direction: larger means more nonfactual. No fit, retune, or audit I/O.
    Caller must establish provenance, independence and prospective commitments.
    """
    require(type(rows) in (list,tuple) and len(rows)>0, "ROWS")
    require(type(thresholds) is tuple and bool(thresholds), "THRESHOLDS")
    def prob(x):
        return type(x) in (int,float) and isfinite(x) and 0<=x<=1
    require(all(prob(t) for t in thresholds) and
            tuple(sorted(set(thresholds)))==thresholds, "THRESHOLDS")
    require(all(prob(x) and 0<x<1 for x in
                (alpha,risk_cap,coverage_floor)), "TARGETS")
    groups=set()
    for row in rows:
        require(type(row) in (tuple,list) and len(row)==3, "ROW_SCHEMA")
        g,y,s=row
        require(type(g) is str and bool(g) and g not in groups, "GROUP")
        require(type(y) is bool and prob(s), "LABEL_OR_SCORE")
        groups.add(g)
    n=len(rows); positives=sum(y for _,y,_ in rows)
    delta=alpha/(2*len(thresholds))
    policies=[]
    for t in thresholds:
        a=sum(s<=t for _,_,s in rows)
        e=sum(y and s<=t for _,y,s in rows)
        u=upper_binomial(e,a,delta)
        l=max(0.,1-upper_binomial(n-a,n,delta))
        policies.append(dict(threshold=t,n=n,nonfactual=positives,
            accepted=a,accepted_nonfactual=e,risk=e/a if a else None,
            coverage=a/n,risk_upper=u,coverage_lower=l,
            meets_bounds_under_assumptions=a>0 and u<=risk_cap and l>=coverage_floor))
    passing=[r for r in policies if r["meets_bounds_under_assumptions"]]
    best=min(passing,key=lambda r:(-r["accepted"],r["risk_upper"],r["threshold"])) if passing else None
    return dict(policies=policies,per_bound_delta=delta,
        selected_threshold=best["threshold"] if best else None,
        status="BOUNDS_MET_UNDER_ASSUMPTIONS" if best else "NO_POLICY_MEETS_BOUNDS",
        independence_verified=False,clinical_safety_certified=False,
        formal_risk_guarantee=False)
