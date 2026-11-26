"""Synthetic prototype of fixed-family benchmark intervention risk calibration.

No models, data access, clinical guarantee, or new theorem. One preselected QA
representative per iid source group is required for the statistical interpretation.
Policies and inputs must be fixed independently of calibration labels.
"""
from math import isfinite

def require(ok, code):
    if not ok:
        raise ValueError(code)

def upper_binomial(x, n, delta):
    """One-sided Clopper-Pearson upper bound; empty denominator is uncertifiable."""
    require(type(x) is int and type(n) is int and 0 <= x <= n and
            type(delta) in (int,float) and isfinite(delta) and 0 < delta < 1,
            "BOUND_INPUT")
    if n == 0 or x == n:
        return 1.0
    if x == 0:
        from math import expm1, log, nextafter
        return min(1.0,nextafter(-expm1(log(delta)/n),1.0))
    # Stable binomial CDF via log-sum-exp, avoiding underflow in tail terms.
    from math import lgamma, log, log1p, exp, fsum
    terms=[lgamma(n+1)-lgamma(k+1)-lgamma(n-k+1) for k in range(x+1)]
    def cdf(p):
        a=log(p); b=log1p(-p)
        z=[v+k*a+(n-k)*b for k,v in enumerate(terms)]
        m=max(z)
        return exp(m)*fsum(exp(v-m) for v in z)
    lo,hi=0.0,1.0
    for _ in range(60):
        mid=(lo+hi)/2
        if mid==lo or mid==hi: break
        if cdf(mid)>delta: lo=mid
        else: hi=mid
    # Small outward numerical cushion; not an interval-arithmetic proof.
    return min(1.0,hi+1e-12)

def policy_summary(original, actions, final):
    """All arrays aligned to same independent source representatives.
    final=None only for abstentions. Accept must preserve original correctness.
    """
    n=len(original)
    require(n>0 and len(actions)==len(final)==n,"ALIGNMENT")
    require(all(type(y) is bool for y in original),"ORIGINAL_LABEL")
    released=errors=introduced=repair=correct_abstain=wrong_abstain=0
    correct=sum(original)
    for y,a,z in zip(original,actions,final):
        require(a in ("accept","revise","abstain"),"ACTION")
        if a=="abstain":
            require(z is None,"ABSTENTION_LABEL")
            correct_abstain+=y;wrong_abstain+=not y
            continue
        require(type(z) is bool and (a!="accept" or z==y),"FINAL_LABEL")
        released+=1; errors+=not z; introduced+=y and not z;repair+=not y and z
    return {"n":n,"original_correct":correct,"released":released,"released_errors":errors,
            "introduced_errors":introduced,"repaired_original_errors":repair,
            "original_correct_abstentions":correct_abstain,"original_wrong_abstentions":wrong_abstain,
            "correct_released":released-errors}

def calibrate(original, policies, *, alpha=.05, release_cap=.10, introduced_cap=.05,
              coverage_floor=.50):
    """Fixed K policies, 3 simultaneous one-sided bounds via Bonferroni.
    No policy fitting, test use, clinical safety certification or shift guarantee.
    """
    require(type(policies) is dict and bool(policies) and all(type(k) is str and k for k in policies),
            "POLICY_SET")
    require(all(type(v) in (int,float) and isfinite(v) and 0<v<1 for v in
                (alpha,release_cap,introduced_cap,coverage_floor)),"TARGETS")
    delta=alpha/(3*len(policies)); report={}
    for name in sorted(policies):
        row=policies[name]
        require(type(row) is dict and set(row)=={"actions","final"},"POLICY_SCHEMA")
        s=policy_summary(original,row["actions"],row["final"])
        n=s["n"];a=s["released"];b=s["original_correct"]
        u=upper_binomial(s["released_errors"],a,delta)
        v=upper_binomial(s["introduced_errors"],b,delta)
        l=max(0.0,1-upper_binomial(n-a,n,delta))
        ok=a>0 and b>0 and u<=release_cap and v<=introduced_cap and l>=coverage_floor
        report[name]={**s,"release_error_upper":u,"introduced_error_upper":v,
                      "coverage_lower":l,"certified_under_assumptions":ok}
    passing=[k for k in report if report[k]["certified_under_assumptions"]]
    chosen=min(passing,key=lambda k:(-report[k]["released"],report[k]["release_error_upper"],k)) if passing else None
    return {"status":"CALIBRATION_CRITERIA_MET" if chosen else "NO_CERTIFIED_POLICY",
            "selected":chosen,"policy_count":len(policies),"per_bound_delta":delta,
            "policies":report,"clinical_safety_certified":False,"distribution_shift_guaranteed":False,
            "real_experiment_executed":False}
