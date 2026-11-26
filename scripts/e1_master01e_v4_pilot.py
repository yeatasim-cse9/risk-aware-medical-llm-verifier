"""Original pilot computation, staged telemetry and budget reconciliation only."""
import time
import e1_master01e_baselines as b
from e1_master01e_v2_pilot import (require,validate_private_inputs,sample,collect,
    fit_head,training,scored,predict_head,length_strata)

def run(engine,train,dev,*,authorized,expected,mark):
    require(authorized is True,"PRIVATE_AUTHORIZATION_REQUIRED")
    validate_private_inputs(train,dev)
    train=sample(train,"train");dev=sample(dev,"dev")
    started=time.monotonic()
    mark("PILOT_TRAIN_INFERENCE")
    tr,tc=collect(engine,train,"train")
    require(tc==expected["train"],"AUDIT_ACTUAL_BUDGET_MISMATCH")
    mark("PILOT_DEV_INFERENCE")
    dv,dc=collect(engine,dev,"dev")
    require(dc==expected["dev"],"AUDIT_ACTUAL_BUDGET_MISMATCH")
    mark("PILOT_HEAD_FIT")
    full=fit_head(training(tr["full"]));removed=fit_head(training(tr["removed"]))
    results={"neural_full":scored(dv["full"],lambda f:predict_head(full,f)),
        "neural_removed_frozen":scored(dv["removed"],lambda f:predict_head(full,f)),
        "neural_permuted_frozen":scored(dv["permuted"],lambda f:predict_head(full,f)),
        "neural_removed_trained":scored(dv["removed"],lambda f:predict_head(removed,f)),
        "nli_contradiction":scored(dv["full"],lambda f:f[1]),
        "nli_non_entailment":scored(dv["full"],lambda f:1-f[0])}
    mark("PILOT_BASELINES")
    for mode in b.MODES:
        results[mode]=b.score_pairs(b.fit(train,mode),dev,"dev")
    # All comparisons exploratory, no best model lock based on pilot.
    mark("PILOT_METRICS")
    metrics={k:b.metrics(v) for k,v in results.items()}
    mark("PILOT_BOOTSTRAP")
    comparisons={k:b.paired_bootstrap(results["neural_full"],v,replicates=1000)
                 for k,v in results.items() if k!="neural_full"}
    return {"status":"EXPLORATORY_TRAIN_DEV_PILOT_ONLY","train_parents":len(train),
        "dev_parents":len(dev),"metrics":metrics,"exploratory_ap_differences":comparisons,
        "length_strata":{k:length_strata(dev,v) for k,v in results.items()},
        "inference_budget":{"train":tc,"dev":dc},"elapsed_seconds":time.monotonic()-started,
        "clinical_validation":False,"test_used":False,"final_split_frozen":False,
        "e1_execution_authorized":False,"public_export_approved":False}
