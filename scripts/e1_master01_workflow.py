"""Private MASTER01 GPU controller. GitHub receives sanitized aggregates only."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request

from kaggle_status import check
import e1_master01_validation_kaggle as master

ROOT = Path(__file__).resolve().parents[1]
KERNEL = "yeatasimbillah/medical-llm-e1-master01-gpu-v1"
PIN = "40f96e132d81a86e61978a29c8a7a3d5aa444fc7"
SHA = "11d803b06167689df6a3caa290be69a8b9b519ee9e83ec0d5de619e95a43761e"
REPORT = "master01_e1_validation_v1.json"
SIZES = {"train":7579,"dev":947,"test":948,"official_holdout":500}

def require(ok, code):
    if not ok:
        raise ValueError(code)

def build(out):
    body = (ROOT/"scripts/e1_master01_validation_kaggle.py").read_bytes()
    require(hashlib.sha256(body).hexdigest()==SHA,"MASTER01_SOURCE_HASH_CHANGED")
    out.mkdir(parents=True,exist_ok=False)
    # Script mode avoids IPython's chained SystemExit formatting failure.
    # No Telegram credentials enter the Kaggle package.
    wrapper = """import os, hashlib, sys
from pathlib import Path
os.environ["MASTER01_PRIVATE_RIGHTS_CONFIRMED"] = "YES"
os.environ["MASTER01_RUNNER_COMMIT"] = %r
body = %r
if hashlib.sha256(body).hexdigest() != %r:
    raise SystemExit("MASTER01_PIN_MISMATCH")
scope = {"__name__":"master01_pinned", "__file__":"runner.py"}
exec(compile(body,"runner.py","exec"),scope)
def workflow_notify(event):
    print("MASTER01_RUNTIME_EVENT="+event,flush=True)
    scope["NOTIFY"].append({"event":event,"delivered":False})
scope["notify"] = workflow_notify
failed = False
try:
    scope["notify"]("START")
    scope["run"]()
except Exception as exc:
    code = str(exc) if type(exc) is RuntimeError else ""
    allowed = {"KAGGLE_GPU_REQUIRED","PRIVATE_DATA_RIGHTS_CONFIRMATION_REQUIRED",
               "PRIVATE_KAGGLE_STORAGE_REQUIRED","REPORT_EXISTS_DO_NOT_OVERWRITE",
               "PINNED_MODULE_HASH_MISMATCH","REMEDIATED_MAPPING_CHANGED",
               "AMBIGUOUS_OFFICIAL_HOLDOUT_TEXT","OFFICIAL_HOLDOUT_TEXT_NOT_COMPLETE",
               "HOLDOUT_HASH_MISMATCH","OLD_MAPPING_CHANGED","NONFINITE_SIMILARITY"}
    print("MASTER01_FAILED_STAGE="+scope["STAGE"],flush=True)
    print("MASTER01_FAILED_CODE="+(code if code in allowed else "SAFE_FAILURE_REQUIRES_REVIEW"),flush=True)
    failed = True
if failed:
    sys.exit(1)
""" % (PIN,body,SHA)
    compile(wrapper,"runner.py","exec")
    (out/"runner.py").write_text(wrapper,encoding="utf-8")
    meta={"id":KERNEL,"title":"medical-llm-e1-master01-gpu-v1","code_file":"runner.py",
          "language":"python","kernel_type":"script","is_private":True,
          "enable_gpu":True,"enable_tpu":False,"enable_internet":True,
          "dataset_sources":[],"competition_sources":[],"kernel_sources":[],"model_sources":[]}
    (out/"kernel-metadata.json").write_text(json.dumps(meta,sort_keys=True,indent=2)+"\n")
    print("MASTER01_PRIVATE_GPU_PACKAGE_VERIFIED")

def telegram(message):
    try:
        token=os.environ["TELEGRAM_BOT_TOKEN"]
        chat=os.environ["TELEGRAM_CHAT_ID"]
        data=urllib.parse.urlencode({"chat_id":chat,"text":"E1 MASTER 01 | "+message}).encode()
        with urllib.request.urlopen(urllib.request.Request(
                "https://api.telegram.org/bot"+token+"/sendMessage",data=data),timeout=15) as r:
            ok=json.loads(r.read(10000)).get("ok") is True
    except Exception:
        ok=False
    print("MASTER01_TELEGRAM_"+("DELIVERED" if ok else "UNAVAILABLE"),flush=True)
    return ok

def preflight():
    result=subprocess.run(["kaggle","quota"],capture_output=True,timeout=90)
    require(result.returncode==0,"QUOTA_ENDPOINT_UNAVAILABLE")
    require(check(KERNEL,"any")=="missing","DUPLICATE_MASTER01_KERNEL_BLOCKED")
    print("MASTER01_PREFLIGHT_OK_QUOTA_AMOUNT_NOT_CERTIFIED")

def integer(v,limit):
    return type(v) is int and 0<=v<=limit

def validate_report(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<60000,"REPORT_FILE_POLICY")
    data=json.loads(path.read_text(encoding="utf-8"))
    require(type(data) is dict,"REPORT_SCHEMA")
    digest=data.pop("integrity_sha256",None)
    require(hashlib.sha256(master.canonical(data)).hexdigest()==digest,"REPORT_DIGEST_MISMATCH")
    expected={"schema_version","status","source_commit","master_runner_commit","pinned_module_sha256",
       "mapping_sha256","medhallu_revision","pubmedqa_revision","source_sha256","official_500_sha256",
       "remediation_summary","model","model_revision","seed","device","versions","diagnostics",
       "exact_screen","semantic_screen","limitations","scientific_binary_labels_adjudicated",
       "source_lineage_fully_certified","complete_semantic_leakage_excluded","final_split_approved",
       "scientific_e1_experiment_executed","private_row_data_exported","notifications_before_finish"}
    require(set(data)==expected,"REPORT_SCHEMA")
    require(data["schema_version"]==1 and data["status"]=="SCREEN_COMPLETE_PENDING_SCIENTIFIC_REVIEW"
      and data["source_commit"]==master.BASE and data["master_runner_commit"]==PIN
      and data["pinned_module_sha256"]==master.PINS and data["mapping_sha256"]==master.EXPECTED_MAPPING
      and data["model"]==master.MODEL and data["model_revision"]==master.MODEL_REV
      and data["seed"]==master.SEED,"REPORT_PIN_MISMATCH")
    for k in ("scientific_binary_labels_adjudicated","source_lineage_fully_certified",
              "complete_semantic_leakage_excluded","final_split_approved",
              "scientific_e1_experiment_executed","private_row_data_exported"):
        require(data[k] is False,"REPORT_SCIENTIFIC_OR_PRIVACY_FLAG")
    require(data["medhallu_revision"]=="515060458a945c633debc6fd5baac7764416b724"
      and data["pubmedqa_revision"]=="323752ad243ce751ba75c9222d34a3130d080030"
      and data["official_500_sha256"]=="939fe566f09017d13b1ca64d2ddfee0bc2374b366048152997669cccedc44d51","DATA_PIN_MISMATCH")
    require(data["source_sha256"]=={
      "medhallu":{"pqa_artificial":"70de8ce7f89a38314fa231a12e45168f3ee1aa591e8a38f6556b89aa8b1ca513",
                  "pqa_labeled":"ac963f9aaeb163f494edd4dbfec04086c3e35c7cc55e3f9c8b1d7b31188fa34c"},
      "pubmedqa":{"pqa_artificial":"cdee2d3c01369f47d2ea7b89cd32ee64b142cc0debcb863bf39b3a81a3cece05",
                  "pqa_labeled":"3d56bd1abc11884579ecc3aab9dd3cdfce8b7cf54715daeca93b8022b7be231c"}},"DATA_FILE_HASH_MISMATCH")
    sys.path.insert(0,str(ROOT/"src"))
    from research_data.e1_pubmedqa_test_exclusion import verify_summary
    s=data["remediation_summary"]
    verify_summary(s)
    require(s["private_mapping_sha256"]==master.EXPECTED_MAPPING
      and s["provisional_parent_split_counts"]=={k:SIZES[k] for k in master.SPLITS[:3]}
      and s["excluded_parent_records_by_reason"]["original_official_500_test_heldout"]==500
      and s["excluded_parent_records_by_reason"]["multiple_pubmedqa_pubids"]==23,"REMEDIATION_MISMATCH")
    pairs={a+"__"+b:(SIZES[a],SIZES[b]) for i,a in enumerate(master.SPLITS) for b in master.SPLITS[i+1:]}
    require(set(data["exact_screen"])==set(data["semantic_screen"])=={"question","context"},"SCREEN_VIEWS")
    for view in ("question","context"):
        exact=data["exact_screen"][view]
        sem=data["semantic_screen"][view]
        require(set(exact)==set(pairs) and set(sem)=={"truncated_rows","comparisons"}
                and set(sem["comparisons"])==set(pairs) and set(sem["truncated_rows"])==set(SIZES),"SCREEN_COVERAGE")
        for k,n in SIZES.items():
            require(integer(sem["truncated_rows"][k],n),"TRUNCATION_COUNT")
        for k,(a,b) in pairs.items():
            x=exact[k]
            require(set(x)=={"shared_groups","left_rows","right_rows"} and integer(x["shared_groups"],min(a,b))
                    and integer(x["left_rows"],a) and integer(x["right_rows"],b),"EXACT_COUNT")
            y=sem["comparisons"][k]
            require(set(y)=={"pairs_compared","maximum_cosine","thresholds"} and y["pairs_compared"]==a*b
                and type(y["maximum_cosine"]) in (int,float) and -1<=y["maximum_cosine"]<=1
                and set(y["thresholds"])=={str(t) for t in master.THRESHOLDS},"SEMANTIC_COVERAGE")
            previous=(a*b,a,b)
            for t in master.THRESHOLDS:
                z=y["thresholds"][str(t)]
                require(set(z)=={"pairs","left_rows","right_rows"},"THRESHOLD_SCHEMA")
                values=(z["pairs"],z["left_rows"],z["right_rows"])
                require(all(integer(v,n) for v,n in zip(values,previous)),"THRESHOLD_MONOTONICITY")
                previous=values
    require(set(data["diagnostics"])=={"pqa_artificial","pqa_labeled"},"DIAGNOSTICS_CONFIG")
    for c, by_split in data["diagnostics"].items():
        require(set(by_split)==set(master.SPLITS[:3]),"DIAGNOSTICS_SPLIT")
        for split,row in by_split.items():
            n=s["source_configuration_by_split"][c][split]
            require(set(row)=={"parents","benchmark_role_0","benchmark_role_1","category_frequency_sorted",
                    "difficulty_frequency_sorted","min_observed_category_count"}
                    and row["parents"]==row["benchmark_role_0"]==row["benchmark_role_1"]==n,"DIAGNOSTICS_ROLES")
            for key in ("category_frequency_sorted","difficulty_frequency_sorted"):
                f=row[key]
                require(type(f) is list and len(f)<=4 and all(integer(v,n) for v in f)
                        and sum(f)==n and sorted(f,reverse=True)==f,"DIAGNOSTICS_COUNTS")
            require(row["category_frequency_sorted"]==s["category_frequency_sorted_without_names_by_config_split"][c][split]
                    and row["min_observed_category_count"]==min(row["category_frequency_sorted"],default=0),"CATEGORY_MISMATCH")
    # Never upload free-form remote strings. Produce a strictly reduced verified aggregate.
    safe={k:data[k] for k in ("schema_version","status","source_commit","master_runner_commit",
            "pinned_module_sha256","mapping_sha256","medhallu_revision","pubmedqa_revision","source_sha256",
            "official_500_sha256","model","model_revision","seed","diagnostics","exact_screen","semantic_screen",
            "scientific_binary_labels_adjudicated","source_lineage_fully_certified",
            "complete_semantic_leakage_excluded","final_split_approved","scientific_e1_experiment_executed",
            "private_row_data_exported")}
    safe["parent_counts"]={k:SIZES[k] for k in master.SPLITS[:3]}
    safe["remote_report_sha256"]=digest
    safe["limitations"]=["Generic embedding model; heuristic thresholds; context prefix truncated to 256 tokens; no clinical adjudication."]
    safe["integrity_sha256"]=hashlib.sha256(master.canonical(safe)).hexdigest()
    return safe

def verify(out):
    require(check(KERNEL,"any")=="complete","REMOTE_NOT_COMPLETE")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        # Select only the fixed aggregate filename; do not retrieve raw notebook logs.
        p=subprocess.run(["kaggle","kernels","output",KERNEL,"-p",tmp,
                          "--file-pattern","^master01_e1_validation_v1[.]json$"],
                         capture_output=True,timeout=180)
        require(p.returncode==0,"REMOTE_AGGREGATE_DOWNLOAD_FAILED")
        safe=validate_report(Path(tmp)/REPORT)
    out.mkdir(parents=True,exist_ok=True)
    (out/"master01_verified_aggregate.json").write_text(json.dumps(safe,sort_keys=True,indent=2)+"\n")
    print("MASTER01_REMOTE_AGGREGATE_VERIFIED_NO_MODEL_TRAINING")

def monitor(out):
    announced=False
    deadline=time.monotonic()+120*60
    while time.monotonic()<deadline:
        s=check(KERNEL,"any")
        if s=="running" and not announced:
            telegram("Kaggle runtime RUNNING observed")
            announced=True
        if s=="complete":
            verify(out)
            telegram("Kaggle COMPLETE | aggregate independently verified | scientific approval pending")
            return
        if s in {"error","failed","cancelled","cancelacknowledged"}:
            telegram("Kaggle terminal failure | "+s)
            raise ValueError("REMOTE_TERMINAL_FAILURE")
        time.sleep(60)
    telegram("Monitoring window ended; use status/verify; no new submission")
    raise ValueError("MONITOR_TIMEOUT_NO_RESUBMIT")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("op",choices=["build","preflight","status","verify","monitor","notify"])
    p.add_argument("--out",type=Path,default=Path("build/master01"))
    p.add_argument("--event",choices=["controller started","submission accepted","controller failed","controller finished"])
    a=p.parse_args()
    try:
        if a.op=="build": build(a.out)
        elif a.op=="preflight": preflight()
        elif a.op=="status": print("MASTER01_KAGGLE_STATUS="+check(KERNEL,"any"))
        elif a.op=="verify": verify(a.out)
        elif a.op=="monitor": monitor(a.out)
        elif a.op=="notify": telegram(a.event)
    except Exception:
        print("MASTER01_CONTROLLER_FAILED="+a.op,flush=True)
        sys.exit(1)
