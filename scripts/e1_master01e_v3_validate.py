"""Allowlisted public pilot aggregate. Scores are exploratory benchmark metrics."""
import math,re
import e1_master01c_core as c
import e1_master01c_private as p
import e1_master01c_validate as v
import e1_master01d_v2_integrity as d
import e1_master01d_v3_validate as prev
import e1_master01e_v3_assets as a
import e1_master01e_baselines as b
require=c.require
MODULES=prev.MODULES+("e1_master01e_baselines","e1_master01e_v2_neural",
    "e1_master01e_v2_pilot","e1_master01e_v3_assets","e1_master01e_v3_private",
    "e1_master01e_v3_validate")
MODES=b.MODES+("neural_full","neural_removed_frozen","neural_permuted_frozen",
    "neural_removed_trained","nli_contradiction","nli_non_entailment")
FALSE=("clinical_validation","test_used","final_split_frozen","e1_execution_authorized",
    "public_export_approved","subgroup_exported","private_rows_exported","scientific_experiment_executed")
def num(x,lo,hi):return type(x) in (int,float) and math.isfinite(x) and lo<=x<=hi
def integer(x,lo,hi):return type(x) is int and lo<=x<=hi
keys=prev.keys
def validate(x,commit,hashes,manifest_digest):
    keys(x,("status","train_parents","dev_parents","metrics","exploratory_ap_differences",
        "inference_budget","elapsed_seconds","schema_version","checkpoint_sha256","mapping_sha256",
        "model_manifest_sha256","model_revision","versions","device","peak_allocated_bytes",
        "peak_reserved_bytes","package_source_commit","package_hashes","integrity_sha256",*FALSE))
    y=x.copy();h=y.pop("integrity_sha256")
    require(v.hexhash(h) and c.digest(y)==h,"REPORT_INTEGRITY")
    keys(hashes,MODULES);keys(x["package_hashes"],MODULES)
    require(x["package_hashes"]==hashes and all(v.hexhash(i) for i in hashes.values())
        and all(hashes[k]==i for k,i in d.PACKAGE_HASHES.items()),"PACKAGE_PIN")
    require(type(commit) is str and re.fullmatch("[0-9a-f]{40}",commit)
        and x["package_source_commit"]==commit,"SOURCE_COMMIT")
    require(type(x["schema_version"]) is int and x["schema_version"]==1
        and x["status"]=="EXPLORATORY_TRAIN_DEV_PILOT_ONLY"
        and x["checkpoint_sha256"]==d.CHECKPOINT and x["mapping_sha256"]==d.MAPPING
        and v.hexhash(manifest_digest) and x["model_manifest_sha256"]==manifest_digest
        and x["model_revision"]==a.REVISION,"PILOT_SCOPE")
    require(all(x[k] is False for k in FALSE),"FALSE_APPROVAL_OR_PRIVACY")
    require(integer(x["train_parents"],128,128) and integer(x["dev_parents"],64,64),"PILOT_COUNTS")
    require(x["versions"]==p.OLD_VERSIONS and x["device"] in v.KNOWN_GPU,"RUNTIME_PIN")
    require(num(x["elapsed_seconds"],0,20000) and integer(x["peak_allocated_bytes"],1,100_000_000_000)
        and integer(x["peak_reserved_bytes"],x["peak_allocated_bytes"],100_000_000_000),"RESOURCE_COUNTS")
    m=x["metrics"];keys(m,MODES)
    for row in m.values():
        keys(row,("parents","average_precision","paired_ranking","brier"))
        require(integer(row["parents"],64,64) and all(num(row[k],0,1)
            for k in ("average_precision","paired_ranking","brier")),"METRIC_RANGE")
    require(m["question_only"]["paired_ranking"]==.5,"QUESTION_ONLY_CONTROL")
    q=x["exploratory_ap_differences"];keys(q,set(MODES)-{"neural_full"})
    for k,row in q.items():
        keys(row,("groups","replicates","ap_difference","percentile_95"))
        require(integer(row["groups"],64,64) and integer(row["replicates"],1000,1000)
            and num(row["ap_difference"],-1,1) and math.isclose(row["ap_difference"],
            m["neural_full"]["average_precision"]-m[k]["average_precision"],abs_tol=1e-12),"AP_DIFFERENCE")
        z=row["percentile_95"]
        require(type(z) is list and len(z)==2 and all(num(t,-1,1) for t in z)
            and z[0]<=z[1],"CI_RANGE")
    budget=x["inference_budget"];keys(budget,("train","dev"))
    for split,n in (("train",128),("dev",64)):
        keys(budget[split],("full","removed","permuted"))
        for k,row in budget[split].items():
            keys(row,("forward_examples","padded_positions","active_tokens","real_windows"))
            require(integer(row["forward_examples"],n*32,n*32)
                and integer(row["padded_positions"],n*32*512,n*32*512)
                and integer(row["active_tokens"],1,n*32*512)
                and integer(row["real_windows"],0 if k=="removed" else n*2,
                        0 if k=="removed" else n*32),"BUDGET_COUNTS")
    return x
