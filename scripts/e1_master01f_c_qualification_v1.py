"""01F-C v1: offline metadata-only qualification; not execution authorization.
No file/network/model access. Never emit identifiers, labels, scores or results.
Historical source membership must be authenticated outside this module.
"""
from dataclasses import dataclass
import math
import e1_master01f_b_v1 as f

VERSION="master01fc-v1"
MIN_REPORT_CELL=10
# Known repository-history scope, not a claim of exhaustive external activity.
HISTORY=(
    ("01_01B_01C","covariate_screen","all_original_sources"),
    ("01D_V3","answer_role_diagnostics","all_retained_train_dev"),
    ("01E_V3_RECOVERY","failed_before_private_pilot","no_verified_fit"),
    ("01E_V4","answer_role_materialization","all_retained_train_dev"),
    ("01E_V4","model_fit","prior_train_sample"),
    ("01E_V4","evaluation","prior_dev_sample"),
    ("01F_01FB","aggregate_adaptation","observed_development"),
)
FALSE_FLAGS=("test_answers_accessed","private_rows_exported","public_export_approved",
             "clinical_validation","iid_groups_verified","provenance_verified",
             "final_protocol_approved","execution_authorized")

def require(ok,code):
    if not ok:raise ValueError(code)

def existing_scope(train_meta,dev_meta):
    """All existing retained train/dev groups had prior role-informed diagnostics.
    Returns sensitive internal sets only, not a report or permission to proceed.
    """
    tr=f.metadata(train_meta,"train");dv=f.metadata(dev_meta,"dev")
    tg={g for p,g in tr};dg={g for p,g in dv}
    require(not(tg&dg or {p for p,g in tr}&{p for p,g in dv}),"SOURCE_OVERLAP")
    return {"label_development_exposed":frozenset(tg|dg),
            "prior_fit":f.prior_pilot_groups(train_meta),
            "observed_selection":frozenset(dg)}

def _ids(v):
    require(type(v) is frozenset and all(type(x) is str and x.strip() for x in v),"GROUP_INVENTORY")
    return v

@dataclass(frozen=True)
class CandidateInventory:
    # In-memory only. Group keys must be canonical connected-component keys
    # computed jointly with all prior originals/holdout covariates.
    candidate_groups:frozenset
    protected_groups:frozenset
    additional_exposed_groups:frozenset
    unknown_provenance_groups:frozenset

def eligible_inventory(inv,existing):
    require(type(inv) is CandidateInventory,"INVENTORY")
    require(type(existing) is dict and set(existing)==
            {"label_development_exposed","prior_fit","observed_selection"},"EXISTING_SCOPE")
    for v in existing.values():_ids(v)
    require(existing["prior_fit"]<=existing["label_development_exposed"]
            and existing["observed_selection"]<=existing["label_development_exposed"],"EXPOSURE_SCOPE")
    candidates,protected,additional,unknown=map(_ids,(inv.candidate_groups,inv.protected_groups,
                                inv.additional_exposed_groups,inv.unknown_provenance_groups))
    require(existing["label_development_exposed"]<=protected,"PROTECTED_SCOPE_INCOMPLETE")
    require(unknown<=candidates,"UNKNOWN_SCOPE")
    # Unknown or overlapping whole components never qualify.
    return candidates-(protected|additional|unknown)

def _bucket(n):
    require(type(n) is int and n>=0,"COUNT")
    return "zero" if n==0 else "small_suppressed" if n<MIN_REPORT_CELL else "at_least_10"

def report(train_meta,dev_meta,inventory,*,target_groups):
    require(type(target_groups) is int and 2<=target_groups<=20000,"TARGET_GROUPS")
    scope=existing_scope(train_meta,dev_meta)
    eligible=eligible_inventory(inventory,scope)
    return {"schema_version":1,"kind":"MASTER01FC_METADATA_QUALIFICATION",
            "existing_confirmatory_calibration":"BLOCKED_PRIOR_LABEL_EXPOSURE",
            "candidate_structural_status":"TARGET_MET_METADATA_ONLY" if len(eligible)>=target_groups
                else "INSUFFICIENT_ELIGIBLE_GROUPS",
            "eligible_count_band":_bucket(len(eligible)),
            "target_groups":target_groups,
            **dict.fromkeys(FALSE_FLAGS,False)}

def validate_report(value,train_meta,dev_meta,inventory,*,target_groups):
    expected=report(train_meta,dev_meta,inventory,target_groups=target_groups)
    require(type(value) is dict and set(value)==set(expected)
            and all(type(value[k]) is type(v) and value[k]==v for k,v in expected.items()),"REPORT_SCHEMA")
    return value

def exploratory_plan(train_meta,dev_meta,*,calibration_groups,additional_model_exposure):
    """Explicitly exploratory only. Does NOT qualify independent calibration.
    Caller must not pass this plan off as an untouched calibration cohort.
    """
    p=f.plan(train_meta,dev_meta,calibration_groups=calibration_groups,
             additional_exposed_fit_groups=additional_model_exposure)
    return p,"EXPLORATORY_PRIOR_LABEL_EXPOSURE"
