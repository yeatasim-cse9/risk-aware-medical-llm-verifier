"""Prospective power calculations using hypothetical parameters only.

No dataset/model access. Exact conditional binomial tests are evaluated
numerically. n is independent source representatives, not rows or answers.
"""
from math import ceil, log, log1p
from pirc_v2_method import binomial_pmf, probability, require

def zero_error_minimum(cap,delta):
    require(probability(cap) and probability(delta) and 0<cap<1 and 0<delta<1,
            "TARGETS")
    n=ceil(log(delta)/log1p(-cap))
    while (1-cap)**n>delta: n+=1
    while n>1 and (1-cap)**(n-1)<=delta: n-=1
    return n

def lower_cutoff(n,p,alpha):
    s=0.0;cut=-1
    for x,mass in enumerate(binomial_pmf(n,p)):
        s+=mass
        if s+1e-12>alpha: break
        cut=x
    return cut

def upper_cutoff(n,p,alpha):
    return n-lower_cutoff(n,1-p,alpha)

def lower_power(n,null,alternative,alpha):
    k=lower_cutoff(n,null,alpha)
    return sum(binomial_pmf(n,alternative)[:k+1])

def upper_power(n,null,alternative,alpha):
    k=upper_cutoff(n,null,alpha)
    return sum(binomial_pmf(n,alternative)[k:])

def feasibility(n,*,policies=12,alpha=.05,release_cap=.10,introduced_cap=.05,
                coverage_floor=.50,true_release_error=.03,true_introduced_error=.01,
                true_coverage=.75,true_original_correct=.70):
    """Marginal certification power + union lower bound for one fixed policy.

    Random conditional denominators integrated over their binomial laws.
    First-step Holm alpha/K sufficient threshold; other policies may help.
    Joint success not a product: endpoint counts are correlated. Input
    marginal scenario need not specify their full joint distribution.
    """
    require(type(n) is int and 1<=n<=2000 and type(policies) is int and policies>0 and
            all(probability(x) and 0<x<1 for x in
                (alpha,release_cap,introduced_cap,coverage_floor)) and
            all(probability(x) for x in (true_release_error,true_introduced_error,
                    true_coverage,true_original_correct)),"PLAN_INPUT")
    require(true_original_correct*true_introduced_error<=true_coverage*true_release_error+1e-12
            and true_original_correct*true_introduced_error<=true_coverage+1e-12,
            "INCONSISTENT_MARGINALS")
    delta=alpha/policies
    r=sum(w*lower_power(m,release_cap,true_release_error,delta)
          for m,w in enumerate(binomial_pmf(n,true_coverage)) if m and w>1e-16)
    i=sum(w*lower_power(m,introduced_cap,true_introduced_error,delta)
          for m,w in enumerate(binomial_pmf(n,true_original_correct)) if m and w>1e-16)
    c=upper_power(n,coverage_floor,true_coverage,delta)
    return {"n":n,"policies":policies,"first_holm_threshold":delta,
            "release_power":r,"introduced_power":i,"coverage_power":c,
            "joint_power_lower_bound":max(0.0,r+i+c-2),
            "joint_power_upper_bound":min(r,i,c)}

def mcnemar_power(n,p01,p10,alpha=.05):
    """One-sided exact paired superiority planning for fixed binary endpoint.

    Condition on discordant count M~Binomial(n,p01+p10). Under null the
    favorable direction probability <=.5. Abstention counts as failure
    for correct-release endpoint, not necessarily for conditional release risk.
    """
    require(type(n) is int and 1<=n<=2000 and probability(p01) and
            probability(p10) and p01+p10<=1 and 0<alpha<1,"MCNEMAR_INPUT")
    d=p01+p10
    if d==0: return 0.0
    return sum(w*upper_power(m,.5,p01/d,alpha)
               for m,w in enumerate(binomial_pmf(n,d)) if m and w>1e-16)

def report():
    import json
    out={"status":"HYPOTHETICAL_PLANNING_NOT_EXPERIMENT",
         "assumptions":{"K":12,"alpha":.05,"caps":[.10,.05,.50],
            "true_rates":[.03,.01,.75,.70],"independent_source_units":True},
         "zero_error_denominators":{str(k):{"release":zero_error_minimum(.10,.05/k),
            "introduced":zero_error_minimum(.05,.05/k)} for k in (1,12,36)},
         "calibration":[feasibility(n) for n in (200,300,500,900,1200)],
         "paired_power":[{"n":n,"p01":a,"p10":b,"power":mcnemar_power(n,a,b,.025)}
                         for n in (300,500,887) for a,b in ((.08,.04),(.10,.04),(.12,.04))]}
    print(json.dumps(out,sort_keys=True,indent=2))

if __name__=="__main__": report()
