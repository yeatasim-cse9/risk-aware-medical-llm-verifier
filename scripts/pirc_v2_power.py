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
    return min(1.0,sum(binomial_pmf(n,alternative)[:k+1]))

def upper_power(n,null,alternative,alpha):
    k=upper_cutoff(n,null,alpha)
    return min(1.0,sum(binomial_pmf(n,alternative)[k:]))

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
    r,i=min(1.0,r),min(1.0,i)
    return {"n":n,"policies":policies,"first_holm_threshold":delta,
            "release_power":r,"introduced_power":i,"coverage_power":c,
            "joint_power_lower_bound":max(0.0,min(r,i,c,r+i+c-2)),
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

def average_precision(labels,scores):
    """Non-interpolated AP, ties grouped; not trapezoidal PR AUC."""
    require(len(labels)==len(scores)>0 and all(type(v) is bool for v in labels)
            and all(probability(v) for v in scores) and 0<sum(labels)<len(labels),"AP_INPUT")
    rows=sorted(zip(scores,labels),reverse=True);seen=pos=0;ap=0.0;i=0;total_pos=sum(labels)
    while i<len(rows):
        j=i;block=0
        while j<len(rows) and rows[j][0]==rows[i][0]:
            block+=rows[j][1];j+=1
        seen+=j-i;pos+=block;ap+=block/total_pos*pos/seen;i=j
    return ap

def pilot_ap_jackknife(groups):
    """Paired source-group jackknife pseudovalue SD; TRAIN pilot only.
    Each group: list of (label, proposed score, baseline score).
    Unequal sizes allowed; target is example-weighted AP for sampled groups.
    Independent groups and asymptotic regularity are assumed for planning only.
    """
    from statistics import stdev
    require(type(groups) is list and len(groups)>=4 and
            all(type(g) is list and g for g in groups),"GROUPS")
    def diff(gs):
        rows=[row for g in gs for row in g]
        y,a,b=zip(*rows)
        return average_precision(y,a)-average_precision(y,b)
    n=len(groups);full=diff(groups)
    pseudo=[n*full-(n-1)*diff(groups[:i]+groups[i+1:]) for i in range(n)]
    return {"pilot_groups":n,"pilot_ap_difference":full,"pseudovalue_sd":stdev(pseudo)}

def ap_planning_groups(sd,effect=.02,alpha=.05,power=.80):
    """Approximate independent-group count, two-sided normal planning.
    Pilot estimates are noisy; report sensitivity and validate by simulation.
    """
    from statistics import NormalDist
    from math import isfinite
    require(type(sd) in (int,float) and isfinite(sd) and sd>0 and
            0<effect<=1 and 0<alpha<1 and .5<power<1,"AP_PLAN_INPUT")
    norm=NormalDist()
    return ceil(((norm.inv_cdf(1-alpha/2)+norm.inv_cdf(power))*sd/effect)**2)

def report():
    import json
    out={"status":"HYPOTHETICAL_PLANNING_NOT_EXPERIMENT",
         "assumptions":{"K":12,"alpha":.05,"caps":[.10,.05,.50],
            "true_rates":[.03,.01,.75,.70],"independent_source_units":True},
         "zero_error_denominators":{str(k):{"release":zero_error_minimum(.10,.05/k),
            "introduced":zero_error_minimum(.05,.05/k)} for k in (1,12,36)},
         "calibration":[feasibility(n) for n in (200,300,500,900,1200)],
         "ap_group_count_sensitivity":[{"hypothetical_sd":sd,"groups_80pct":ap_planning_groups(sd),
             "groups_90pct":ap_planning_groups(sd,power=.90)} for sd in (.15,.25,.35)],
         "calibration_tight_margin":[feasibility(n,true_release_error=.07,true_introduced_error=.03)
                                      for n in (300,500,900,1200)],
         "paired_power":[{"n":n,"p01":a,"p10":b,"power":mcnemar_power(n,a,b,.025)}
                         for n in (300,500,887) for a,b in ((.08,.04),(.10,.04),(.12,.04))]}
    print(json.dumps(out,sort_keys=True,indent=2))

if __name__=="__main__": report()
