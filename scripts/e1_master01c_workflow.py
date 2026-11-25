"""Code-only packaging and private GPU lifecycle controller. No automatic retries."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from kaggle_status import check
import e1_master01c_core as c
import e1_master01c_private as p
import e1_master01c_validate as v

ROOT=Path(__file__).resolve().parents[1]
PREVIOUS="yeatasimbillah/medical-llm-e1-master01b-gpu-v2"
KERNEL="yeatasimbillah/medical-llm-e1-master01c-gpu-v1"
REPORT="master01c_provisional_aggregate_v1.json"
VERIFIED="master01c_verified_aggregate.json"
require=c.require

def sha(b): return hashlib.sha256(b).hexdigest()

def source(commit=None):
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    commit=commit or head
    require(type(commit) is str and re.fullmatch("[0-9a-f]{40}",commit),"SOURCE_COMMIT")
    require(subprocess.run(["git","merge-base","--is-ancestor",commit,head],
                 cwd=ROOT,capture_output=True).returncode==0,"SOURCE_NOT_IN_HISTORY")
    bodies={name:subprocess.check_output(["git","show",commit+":scripts/"+name+".py"],
                                        cwd=ROOT) for name in v.MODULES}
    hashes={name:sha(body) for name,body in bodies.items()}
    require(hashes[v.MODULES[0]]==p.BASE_SHA and hashes[v.MODULES[1]]==p.PRIOR_SHA,
            "PREVIOUS_SOURCE_PIN")
    # Current controller must understand the exact expected report schema.
    for name in v.MODULES:
        require((ROOT/"scripts"/(name+".py")).read_bytes()==bodies[name],
                "SOURCE_WORKTREE_OR_VERSION_MISMATCH")
    return commit,bodies,hashes

def build(out):
    commit,bodies,hashes=source()
    out.mkdir(parents=True,exist_ok=False)
    wrapper="""import hashlib, json, os, sys, tempfile, types
from pathlib import Path
bodies=%r
hashes=%r
commit=%r
try:
    if any(hashlib.sha256(body).hexdigest()!=hashes[name] for name,body in bodies.items()):
        raise ValueError("PACKAGE_PIN")
    for name,body in bodies.items():
        module=types.ModuleType(name)
        sys.modules[name]=module
        exec(compile(body,name+".py","exec"),module.__dict__)
    import e1_master01c_private as runner
    import e1_master01c_core as core
    os.environ["MASTER01C_PRIVATE_RIGHTS_CONFIRMED"]="YES"
    work=Path("/kaggle/working")
    scratch=Path("/kaggle/temp")
    if not work.is_dir() or work.is_symlink() or scratch.is_symlink():
        raise ValueError("PRIVATE_STORAGE")
    scratch.mkdir(mode=0o700,exist_ok=True)
    out=work/%r
    if out.exists() or out.is_symlink():
        raise ValueError("DO_NOT_OVERWRITE")
    with tempfile.TemporaryDirectory(dir=scratch,prefix="master01c-") as tmp:
        report=runner.prepare_private(Path(tmp))
        report["package_source_commit"]=commit
        report["package_hashes"]=hashes
        report["integrity_sha256"]=core.digest(report)
        payload=json.dumps(report,sort_keys=True,indent=2,allow_nan=False)
        if len(payload.encode())>=100000:
            raise ValueError("REPORT_SIZE")
        with out.open("x",encoding="utf-8") as f:
            f.write(payload+"\n")
    print("MASTER01C_PROVISIONAL_AGGREGATE_SAVED",flush=True)
    print("MASTER01C_FINAL_STATUS=SCIENTIFIC_APPROVAL_REQUIRED",flush=True)
except Exception as exc:
    stage=getattr(sys.modules.get("e1_master01c_private"),"STAGE","BOOT")
    if not isinstance(stage,str) or not stage.isascii() or not stage.replace("_","").isalnum():
        stage="UNKNOWN"
    allowed={"PACKAGE_PIN","PRIVATE_STORAGE","DO_NOT_OVERWRITE","RIGHTS_REQUIRED",
             "KAGGLE_GPU_REQUIRED","MODEL_ASSET_PIN","MODEL_SMOKE_COVERAGE","MODEL_SMOKE_PAIR",
             "HOLDOUT_PIN","OLD_MAPPING_PIN","MAPPING_PIN","INDEPENDENT_MAPPING_PIN",
             "SOURCE_PROXY","HOLDOUT_SOURCE_CONTAMINATION","PAIRED_ANSWER_SCHEMA",
             "AMBIGUOUS_HOLDOUT","HOLDOUT_COVERAGE","PARENT_COUNTS","SOURCE_REPRODUCTION",
             "NUMERICAL_REPRODUCTION","ENCODER_AGGREGATE_REPRODUCTION","AGGREGATE_REPRODUCTION",
             "RESIDUAL_SOURCE_PROXY_CONFLICT","RESIDUAL_FLAGGED_PAIR","EMPTY_RETAINED_SPLIT",
             "REPORT_SIZE"}
    code=str(exc) if type(exc) is ValueError else ""
    print("MASTER01C_FAILED_STAGE="+stage,flush=True)
    print("MASTER01C_FAILED_CODE="+(code if code in allowed else "SAFE_FAILURE_REQUIRES_REVIEW"),flush=True)
    sys.exit(1)
"""%(bodies,hashes,commit,REPORT)
    compile(wrapper,"runner.py","exec")
    (out/"runner.py").write_text(wrapper,encoding="utf-8")
    meta={"id":KERNEL,"title":KERNEL.split("/")[1],"code_file":"runner.py","language":"python",
          "kernel_type":"script","is_private":True,"enable_gpu":True,"enable_tpu":False,
          "enable_internet":True,"dataset_sources":[],"competition_sources":[],
          "kernel_sources":[],"model_sources":[]}
    (out/"kernel-metadata.json").write_text(json.dumps(meta,sort_keys=True,indent=2)+"\n")
    print("MASTER01C_CODE_ONLY_PACKAGE_VERIFIED",flush=True)

def telegram(message):
    try:
        token=os.environ["TELEGRAM_BOT_TOKEN"];chat=os.environ["TELEGRAM_CHAT_ID"]
        payload=urllib.parse.urlencode({"chat_id":chat,"text":"E1 MASTER 01C | "+message}).encode()
        with urllib.request.urlopen(urllib.request.Request(
            "https://api.telegram.org/bot"+token+"/sendMessage",data=payload),timeout=15) as resp:
            ok=json.loads(resp.read(10000)).get("ok") is True
    except Exception:
        ok=False
    print("MASTER01C_TELEGRAM_"+("DELIVERED" if ok else "UNAVAILABLE"),flush=True)
    return ok

def preflight():
    q=subprocess.run(["kaggle","quota"],capture_output=True,timeout=90)
    require(q.returncode==0,"QUOTA_ENDPOINT_UNAVAILABLE")
    require(check(PREVIOUS,"any")=="complete","PREVIOUS_RUN_NOT_COMPLETE")
    require(check(KERNEL,"any")=="missing","DUPLICATE_MASTER01C_KERNEL_BLOCKED")
    print("MASTER01C_PREFLIGHT_OK_QUOTA_AMOUNT_NOT_CERTIFIED")

def verify(out,expected_commit=None):
    commit,_,hashes=source(expected_commit)
    require(check(KERNEL,"any")=="complete","REMOTE_NOT_COMPLETE")
    with tempfile.TemporaryDirectory() as tmp:
        proc=subprocess.run(["kaggle","kernels","output",KERNEL,"-p",tmp,
                            "--file-pattern","^master01c_provisional_aggregate_v1[.]json$"],
                            capture_output=True,timeout=180)
        require(proc.returncode==0,"REMOTE_AGGREGATE_DOWNLOAD_FAILED")
        path=Path(tmp)/REPORT
        require(path.is_file() and not path.is_symlink() and path.stat().st_size<100000,
                "REPORT_FILE_POLICY")
        d=json.loads(path.read_text(encoding="utf-8"))
        safe=v.validate(d,commit,hashes)
    out.mkdir(parents=True,exist_ok=False)
    (out/VERIFIED).write_text(json.dumps(safe,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("MASTER01C_AGGREGATE_VERIFIED_SCIENTIFIC_APPROVAL_REQUIRED")

def monitor(out,expected_commit=None):
    announced=False
    deadline=time.monotonic()+240*60
    while time.monotonic()<deadline:
        state=check(KERNEL,"any")
        if state=="running" and not announced:
            telegram("Kaggle runtime RUNNING observed");announced=True
        if state=="complete":
            verify(out,expected_commit)
            return
        if state in {"error","failed","cancelled","cancelacknowledged"}:
            telegram("Kaggle terminal failure | "+state)
            raise ValueError("REMOTE_TERMINAL_FAILURE")
        time.sleep(60)
    telegram("Monitoring window ended; use status/verify; no new submission")
    raise ValueError("MONITOR_TIMEOUT_NO_RESUBMIT")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("op",choices=["build","preflight","status","verify","monitor","notify"])
    parser.add_argument("--out",type=Path,default=Path("build/master01c"))
    parser.add_argument("--expected-commit",default=None)
    parser.add_argument("--event",choices=["controller started","submission accepted","controller failed",
        "Kaggle COMPLETE | aggregate independently verified | scientific approval pending"])
    a=parser.parse_args()
    try:
        if a.op=="build": build(a.out)
        elif a.op=="preflight": preflight()
        elif a.op=="status": print("MASTER01C_KAGGLE_STATUS="+check(KERNEL,"any"))
        elif a.op=="verify": verify(a.out,a.expected_commit)
        elif a.op=="monitor": monitor(a.out,a.expected_commit)
        elif a.op=="notify": telegram(a.event)
    except Exception as exc:
        code=str(exc) if type(exc) is ValueError and re.fullmatch("[A-Z_0-9]{1,80}",str(exc)) else "SAFE_FAILURE_REQUIRES_REVIEW"
        print("MASTER01C_CONTROLLER_FAILED="+a.op+" | "+code,flush=True)
        sys.exit(1)
