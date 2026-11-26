"""Strict aggregate schema, no arbitrary strings, rows or approval claims."""
import math
import re
import e1_master01c_core as c
import e1_master01c_private as p
import e1_master01c_validate as v
import e1_master01d_v2_integrity as d
import e1_master01d_v2_cohort as cohort

require=c.require
FALSE_FLAGS=("human_review_completed","clinical_labels_approved",
    "complete_semantic_leakage_excluded","iid_groups_verified",
    "new_calibration_cohort_approved","final_split_frozen",
    "e1_execution_authorized","scientific_experiment_executed","private_rows_exported")
MODULES=("e1_master01c_core","e1_master01_validation_kaggle","e1_master01b_kaggle",
    "e1_master01c_private","e1_master01c_validate","e1_master01d_v2_integrity",
    "pirc_calibration_prototype","pirc_v2_method","pirc_v2_power",
    "e1_master01d_v2_cohort","e1_master01d_v3_private","e1_master01d_v3_validate")
SHORTCUTS=("parents","identical_normalized_roles","hallucinated_role_longer",
    "ground_truth_role_longer","length_tie","ground_truth_exact_context_copy",
    "hallucinated_exact_context_copy")
EXCLUSIONS=("prior_protected_overlap","official_test_overlap","missing_eligibility",
    "nonrepresentative_same_group")
def keys(x,k):
    require(type(x) is dict and set(x)==set(k),"AGGREGATE_SCHEMA")
def integer(x,lo,hi):
    return type(x) is int and lo<=x<=hi
def same_numeric(a,b):
    """Strict keys/types, small numerical tolerance on recomputed float power."""
    if type(b) is dict:
        return type(a) is dict and set(a)==set(b) and all(same_numeric(a[k],v) for k,v in b.items())
    if type(b) is list:
        return type(a) is list and len(a)==len(b) and all(same_numeric(x,y) for x,y in zip(a,b))
    if type(b) is float:
        return type(a) in (int,float) and math.isfinite(a) and math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-12)
    return type(a) is type(b) and a==b

def validate(a,commit,hashes):
    keys(a,("schema_version","status","checkpoint_sha256","mapping_sha256",
        "retained","source_group_counts","residual_flagged_pairs","supplemental_exact_conflicts",
        "source_proxy_conflicts","maxima","versions","device","shortcut_diagnostics",
        "pubmedqa_inventory","hypothetical_power","bioasq_status","inventory_scope",
        "package_source_commit","package_hashes","integrity_sha256",*FALSE_FLAGS))
    raw=a.copy();h=raw.pop("integrity_sha256")
    require(v.hexhash(h) and c.digest(raw)==h,"REPORT_INTEGRITY")
    require(type(commit) is str and re.fullmatch("[0-9a-f]{40}",commit) and
            a["package_source_commit"]==commit,"SOURCE_COMMIT")
    keys(hashes,MODULES);keys(a["package_hashes"],MODULES)
    require(all(v.hexhash(x) for x in hashes.values()) and a["package_hashes"]==hashes and
            all(hashes[k]==x for k,x in d.PACKAGE_HASHES.items()),"PACKAGE_PIN")
    require(type(a["schema_version"]) is int and a["schema_version"]==1 and
        a["status"]=="PRIVATE_BENCHMARK_AUDIT_ONLY" and a["checkpoint_sha256"]==d.CHECKPOINT
        and a["mapping_sha256"]==d.MAPPING and
        a["bioasq_status"]=="NOT_ACQUIRED_RELEASE_AND_RIGHTS_PENDING" and
        a["inventory_scope"]=="EXACT_PROXY_ONLY_PRIOR_01C_ORIGINALS_PROTECTED","EVIDENCE_SCOPE")
    require(all(a[k] is False for k in FALSE_FLAGS),"FALSE_APPROVAL_OR_PRIVACY")
    keys(a["retained"],d.RETAINED);keys(a["source_group_counts"],d.RETAINED)
    require(all(type(a["retained"][s]) is int and a["retained"][s]==n and
                integer(a["source_group_counts"][s],1,n) for s,n in d.RETAINED.items()),
                "RETENTION_COUNTS")
    require(all(type(a[k]) is int and a[k]==0 for k in ("residual_flagged_pairs",
                "supplemental_exact_conflicts","source_proxy_conflicts")),"RESIDUAL_CONFLICT")
    p.compare_maxima(a["maxima"])
    require(a["versions"]==p.OLD_VERSIONS and type(a["device"]) is str and
                a["device"] in v.KNOWN_GPU,"RUNTIME_PIN")
    sc=a["shortcut_diagnostics"];keys(sc,("train","dev"))
    for s,x in sc.items():
        keys(x,SHORTCUTS);n=d.RETAINED[s]
        require(all(integer(y,0,n) for y in x.values()) and x["parents"]==n and
            x["hallucinated_role_longer"]+x["ground_truth_role_longer"]+x["length_tie"]==n and
            x["identical_normalized_roles"]<=x["length_tie"],"SHORTCUT_COUNTS")
    inv=a["pubmedqa_inventory"];keys(inv,("candidate_rows","excluded_rows","eligible_group_representatives"))
    keys(inv["candidate_rows"],("development","official_test"))
    require(inv["candidate_rows"]=={"development":500,"official_test":0} and
            all(type(x) is int for x in inv["candidate_rows"].values()),"INVENTORY_COUNT")
    ex=inv["excluded_rows"];g=inv["eligible_group_representatives"]
    keys(ex,EXCLUSIONS);keys(g,cohort.PARTITIONS)
    require(all(integer(x,0,500) for x in (*ex.values(),*g.values())) and
            g["test"]==0 and ex["official_test_overlap"]==0 and
            sum(ex.values())+sum(g.values())==500,"INVENTORY_RECONCILIATION")
    require(same_numeric(a["hypothetical_power"],cohort.prospective_power(g["calibration"])),
            "POWER_RECOMPUTATION")
    return a
