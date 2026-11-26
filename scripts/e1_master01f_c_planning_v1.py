"""01F-C prospective power and compute planning. Hypothetical inputs only.
No dataset, score, model or network access. No certified quota or runtime.
"""
from math import fsum, isfinite
from pirc_v2_method import binomial_pmf
from pirc_v2_power import lower_cutoff, upper_cutoff, zero_error_minimum, ap_planning_groups
from e1_master01f_b_calibration_v1 import THRESHOLDS
from e1_master01e_v4_neural import SLOTS,MAX_LEN
from e1_master01f_c_qualification_v1 import require

def probability(x):
    return type(x) in (int,float) and isfinite(x) and 0<x<1

def group_power(n,*,true_risk,true_coverage,risk_cap=.10,coverage_floor=.50,alpha=.05):
    """Joint power for ONE fixed threshold under iid group representative model.
    M~Binomial(n, coverage); E|M~Binomial(M, risk).
    Integrate conditional risk pass probability over released group denominator,
    restricted to denominators passing the exact one-sided coverage test.
    Same Bonferroni allocation as 01F-B, no independence of endpoints assumed.
    Power for at least one threshold in family is >= fixed-threshold power
    under that threshold's scenario; not a prediction of actual performance.
    Numerical binomial computations are not interval-arithmetic proofs.
    """
    require(type(n) is int and 1<=n<=2000 and all(probability(x) for x in
           (true_risk,true_coverage,risk_cap,coverage_floor,alpha)),"POWER_INPUT")
    delta=alpha/(2*len(THRESHOLDS))
    coverage_min=upper_cutoff(n,coverage_floor,delta)
    terms=[]
    for m,w in enumerate(binomial_pmf(n,true_coverage)):
        if m==0 or m<coverage_min or w==0:continue
        cut=lower_cutoff(m,risk_cap,delta)
        if cut>=0:terms.append(w*fsum(binomial_pmf(m,true_risk)[:cut+1]))
    return {"groups":n,"per_bound_delta":delta,"minimum_released_zero_errors":
            zero_error_minimum(risk_cap,delta),"coverage_min_released":coverage_min,
            "joint_power_one_fixed_threshold":min(1.,max(0.,fsum(terms))),
            "hypothetical":True,"iid_verified":False}

def ap_sensitivity(*,effect=.02,alpha=.05,power=.80):
    """Approximate independent-group normal planning; hypothetical SD grid.
    Not an estimate of V4 variance or a confirmed sample size.
    """
    return [{"hypothetical_pseudovalue_sd":sd,
             "groups":ap_planning_groups(sd,effect,alpha,power)}
            for sd in (.15,.25,.35)]

def _seconds(x):
    return type(x) in (float,int) and isfinite(x) and x>0

def compute_plan(*,measured_train_parents,measured_dev_parents,
                 measured_train_inference_seconds,measured_dev_inference_seconds,
                 measured_pre_inference_seconds,measured_post_inference_seconds,
                 fit_parents,selection_parents,calibration_parents,
                 safety_multiplier=1.30):
    """All conditions for all parents (current collection adapter), single cuda:0.
    V4 timing inputs must be authenticated privately; no actual V4 numbers here.
    Conservative larger observed seconds/parent, fixed setup/post reference.
    Postprocessing growth, queueing and quota rounding are NOT modeled.
    """
    require(all(type(n) is int and 0<n<=20000 for n in
                (measured_train_parents,measured_dev_parents,fit_parents,selection_parents))
            and type(calibration_parents) is int and 0<=calibration_parents<=20000,
            "PARENT_COUNT")
    require(all(_seconds(x) for x in (measured_train_inference_seconds,
                measured_dev_inference_seconds,measured_pre_inference_seconds,
                measured_post_inference_seconds)),"RUNTIME_INPUT")
    require(type(safety_multiplier) in (int,float) and isfinite(safety_multiplier)
            and 1<=safety_multiplier<=4,"MARGIN")
    per=max(measured_train_inference_seconds/measured_train_parents,
            measured_dev_inference_seconds/measured_dev_parents)
    total=fit_parents+selection_parents+calibration_parents
    forwards=total*2*3*SLOTS
    runtime=measured_pre_inference_seconds+per*total+measured_post_inference_seconds
    require(isfinite(runtime*safety_multiplier),"RUNTIME_RANGE")
    return {"total_parents":total,"forward_examples":forwards,"padded_positions":forwards*MAX_LEN,
            "seconds_per_parent_reference":per,"projected_seconds":runtime,
            "planning_seconds_with_margin":runtime*safety_multiplier,
            "postprocessing_scaling_modeled":False,"queue_time_modeled":False,
            "quota_certified":False,"actual_runtime_guaranteed":False,
            "public_export_approved":False}

def validate_compute(value,**inputs):
    expected=compute_plan(**inputs)
    require(type(value) is dict and set(value)==set(expected)
            and all(type(value[k]) is type(v) and value[k]==v for k,v in expected.items()),"COMPUTE_SCHEMA")
    return value
