"""Manual-only private exploratory neural pilot controller. No retry or E1 test."""
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
import e1_master01d_v2_integrity as d
import e1_master01e_v3_validate as v
import e1_master01e_v3_assets as assets

ROOT=Path(__file__).resolve().parents[1]
PREVIOUS="yeatasimbillah/medical-llm-e1-master01d-v3-gpu-v1"
KERNEL="yeatasimbillah/medical-llm-e1-master01e-v3-pilot-gpu-v1"
REPORT="master01ev3_private_aggregate.json"
VERIFIED="master01ev3_verified_aggregate.json"
CHECKPOINT="configs/master01c-verified-aggregate-2026-11-26.json"
MANIFEST="configs/master01e-v3-model-manifest.json"
ARCHIVE="configs/master01dv3-verified-aggregate-2026-11-26.json"
ARCHIVE_DIGEST="a8e159311f8829a5b5ece7e12e0ffd3a76812f20dd3f066c9d508bf23304ace8"
EVENTS=("controller started","submission accepted","controller failed",
        "Kaggle COMPLETE | aggregate independently verified | exploratory pilot only")
require=c.require

def source(commit=None,*,require_manifest=True):
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    commit=commit or head
    require(type(commit) is str and re.fullmatch("[0-9a-f]{40}",commit),"SOURCE_COMMIT")
    require(subprocess.run(["git","merge-base","--is-ancestor",commit,head],
                cwd=ROOT,capture_output=True).returncode==0,"SOURCE_NOT_IN_HISTORY")
    bodies={k:subprocess.check_output(["git","show",commit+":scripts/"+k+".py"],
               cwd=ROOT) for k in v.MODULES}
    hashes={k:hashlib.sha256(x).hexdigest() for k,x in bodies.items()}
    require(all(hashes[k]==h for k,h in d.PACKAGE_HASHES.items()),"PRIOR_SOURCE_PIN")
    for k,x in bodies.items():
        require((ROOT/"scripts"/(k+".py")).read_bytes()==x,"SOURCE_VERSION_MISMATCH")
    report=json.loads(subprocess.check_output(["git","show",commit+":"+CHECKPOINT],cwd=ROOT))
    d.verify_checkpoint(report)
    archived=json.loads(subprocess.check_output(["git","show",commit+":"+ARCHIVE],cwd=ROOT))
    saved=archived.pop("integrity_sha256")
    require(saved==ARCHIVE_DIGEST and c.digest(archived)==saved,"ARCHIVE_PIN")
    require(all(hashes[k]==h for k,h in archived["package_hashes"].items()),"ARCHIVED_SOURCE_PIN")
    proc=subprocess.run(["git","show",commit+":"+MANIFEST],cwd=ROOT,capture_output=True)
    manifest=None;manifest_digest=None
    if proc.returncode==0:
        pin=json.loads(proc.stdout)
        require(type(pin) is dict and set(pin)=={"manifest","manifest_sha256"},"MANIFEST_SCHEMA")
        manifest=assets.validate_manifest(pin["manifest"])
        manifest_digest=pin["manifest_sha256"]
        require(assets.digest(manifest)==manifest_digest,"MANIFEST_PIN")
        require(json.loads((ROOT/MANIFEST).read_text())==pin,"MANIFEST_SOURCE_DRIFT")
    else:
        require(not require_manifest,"REVIEWED_MODEL_MANIFEST_REQUIRED")
    return commit,bodies,hashes,report,manifest,manifest_digest

def build(out,authorized=False):
    require(type(authorized) is bool,"AUTHORIZATION_TYPE")
    commit,bodies,hashes,report,manifest,manifest_digest=source(require_manifest=authorized)
    out.mkdir(parents=True,exist_ok=False)
    wrapper=r"""import hashlib,json,os,sys,tempfile
from pathlib import Path
bodies=%r
hashes=%r
commit=%r
checkpoint=%r
manifest=%r
manifest_digest=%r
authorized=%r
try:
    if authorized is not True:
        raise ValueError("AUTHORIZATION_REQUIRED")
    if any(hashlib.sha256(body).hexdigest()!=hashes[k] for k,body in bodies.items()):
        raise ValueError("PACKAGE_PIN")
    work=Path("/kaggle/working");scratch=Path("/kaggle/temp")
    if not work.is_dir() or work.is_symlink() or scratch.is_symlink():
        raise ValueError("PRIVATE_STORAGE")
    scratch.mkdir(mode=0o700,exist_ok=True)
    out=work/%r
    if out.exists() or out.is_symlink():
        raise ValueError("DO_NOT_OVERWRITE")
    print("MASTER01EV3_PRIVATE_PILOT_STARTED",flush=True)
    # Silence raw library diagnostics at the OS descriptor level during private processing.
    sys.stdout.flush();sys.stderr.flush()
    saved=[os.dup(i) for i in (1,2)]
    try:
        with open(os.devnull,"w") as sink:
            for i in (1,2): os.dup2(sink.fileno(),i)
            with tempfile.TemporaryDirectory(dir=scratch,prefix="master01ev3-") as tmp:
                pkg=Path(tmp)/"code";pkg.mkdir()
                root=Path(tmp)/"private";root.mkdir(mode=0o700)
                for k,body in bodies.items():
                    (pkg/(k+".py")).write_bytes(body)
                sys.path.insert(0,str(pkg))
                import e1_master01e_v3_private as runner
                import e1_master01e_v3_validate as validator
                import e1_master01c_core as core
                os.environ["MASTER01EV3_AUTHORIZED_ONE_PILOT"]="YES"
                result=runner.run(root,checkpoint,manifest,manifest_digest)
                result["package_source_commit"]=commit
                result["package_hashes"]=hashes
                result["integrity_sha256"]=core.digest(result)
                validator.validate(result,commit,hashes,manifest_digest)
                payload=json.dumps(result,sort_keys=True,indent=2,allow_nan=False)
                if len(payload.encode())>=100000:
                    raise ValueError("REPORT_SIZE")
            sys.stdout.flush();sys.stderr.flush()
    finally:
        for i,fd in zip((1,2),saved):
            os.dup2(fd,i);os.close(fd)
    with out.open("x",encoding="utf-8") as f:
        f.write(payload+"\n")
    print("MASTER01EV3_AGGREGATE_SAVED_EXPLORATORY_ONLY",flush=True)
except Exception as exc:
    allowed={"AUTHORIZATION_REQUIRED","PACKAGE_PIN","PRIVATE_STORAGE","DO_NOT_OVERWRITE",
             "GPU_REQUIRED","RUNTIME_VERSION_DRIFT","NLI_SMOKE","CONTEXT_BUDGET_EXCEEDED","HYPOTHESIS_BUDGET_EXCEEDED","ASSET_HASH","MODEL_LOAD_MISMATCH","MANIFEST_PIN","MODEL_LABEL_MAP","MODEL_ASSET_PIN","GPU_SMOKE_COVERAGE",
             "GPU_SMOKE_PAIR","CHECKPOINT_DRIFT","FLAG_COUNT","HOLDOUT_PIN","HOLDOUT_IDENTITY",
             "SUPPLEMENTAL_PROXY_CONFLICT","REPORT_SIZE","LOCAL_PARQUET_SCHEMA",
             "LOCAL_PARQUET_PIN","LABELED_COUNT","NONHOLDOUT_COUNT","LABELED_PMID",
             "REPORT_INTEGRITY","POWER_RECOMPUTATION","INVENTORY_RECONCILIATION",
             "NUMERICAL_REPRODUCTION","ENCODER_AGGREGATE_REPRODUCTION","AGGREGATE_REPRODUCTION"}
    code=str(exc) if type(exc) is ValueError else ""
    print("MASTER01EV3_FAILED_CODE="+(code if code in allowed else "SAFE_FAILURE_REQUIRES_REVIEW"),flush=True)
    sys.exit(1)
"""%(bodies,hashes,commit,report,manifest,manifest_digest,authorized,REPORT)
    compile(wrapper,"runner.py","exec")
    (out/"runner.py").write_text(wrapper,encoding="utf-8")
    meta={"id":KERNEL,"title":KERNEL.split("/")[1],"code_file":"runner.py","language":"python",
          "kernel_type":"script","is_private":True,"enable_gpu":True,"enable_tpu":False,
          "enable_internet":True,"dataset_sources":[],"competition_sources":[],
          "kernel_sources":[],"model_sources":[]}
    (out/"kernel-metadata.json").write_text(json.dumps(meta,sort_keys=True,indent=2)+"\n")
    print("MASTER01EV3_CODE_ONLY_PACKAGE_VERIFIED")

def telegram(message):
    try:
        token=os.environ["TELEGRAM_BOT_TOKEN"];chat=os.environ["TELEGRAM_CHAT_ID"]
        payload=urllib.parse.urlencode({"chat_id":chat,"text":"E1 MASTER 01E V3 | "+message}).encode()
        with urllib.request.urlopen(urllib.request.Request(
            "https://api.telegram.org/bot"+token+"/sendMessage",data=payload),timeout=15) as resp:
            ok=json.loads(resp.read(10000)).get("ok") is True
    except Exception:
        ok=False
    print("MASTER01EV3_TELEGRAM_"+("DELIVERED" if ok else "UNAVAILABLE"),flush=True)
    return ok

def preflight():
    source()
    q=subprocess.run(["kaggle","quota"],capture_output=True,timeout=90)
    require(q.returncode==0,"QUOTA_ENDPOINT_UNAVAILABLE")
    require(check(PREVIOUS,"any")=="complete","PREVIOUS_RUN_NOT_COMPLETE")
    require(check(KERNEL,"any")=="missing","DUPLICATE_KERNEL_BLOCKED")
    print("MASTER01EV3_PREFLIGHT_OK_QUOTA_AMOUNT_NOT_CERTIFIED")

def verify(out,expected_commit):
    require(bool(expected_commit),"EXPECTED_SUBMISSION_COMMIT_REQUIRED")
    commit,_,hashes,_,_,manifest_digest=source(expected_commit)
    require(check(KERNEL,"any")=="complete","REMOTE_NOT_COMPLETE")
    with tempfile.TemporaryDirectory() as tmp:
        proc=subprocess.run(["kaggle","kernels","output",KERNEL,"-p",tmp,
                        "--file-pattern","^master01ev3_private_aggregate[.]json$"],
                        capture_output=True,timeout=180)
        require(proc.returncode==0,"REMOTE_AGGREGATE_DOWNLOAD_FAILED")
        path=Path(tmp)/REPORT
        require(path.is_file() and not path.is_symlink() and 0<path.stat().st_size<100000,
                "REPORT_FILE_POLICY")
        a=json.loads(path.read_text(encoding="utf-8"))
        safe=v.validate(a,commit,hashes,manifest_digest)
    out.mkdir(parents=True,exist_ok=False)
    (out/VERIFIED).write_text(json.dumps(safe,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("MASTER01EV3_AGGREGATE_VERIFIED_EXPLORATORY_ONLY")

def monitor(out,expected_commit):
    announced=False;deadline=time.monotonic()+240*60
    while time.monotonic()<deadline:
        state=check(KERNEL,"any")
        if state=="running" and not announced:
            telegram("Kaggle runtime RUNNING observed");announced=True
        if state=="complete":
            verify(out,expected_commit);return
        if state in {"error","failed","cancelled","cancelacknowledged"}:
            telegram("Kaggle terminal failure | "+state)
            raise ValueError("REMOTE_TERMINAL_FAILURE")
        time.sleep(60)
    telegram("Monitoring window ended; use status/verify; no new submission")
    raise ValueError("MONITOR_TIMEOUT_NO_RESUBMIT")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("op",choices=["build","preflight","status","verify","monitor","notify"])
    parser.add_argument("--out",type=Path,default=Path("build/master01ev3"))
    parser.add_argument("--expected-commit",default="")
    parser.add_argument("--authorized-submit",action="store_true")
    parser.add_argument("--event",choices=EVENTS)
    a=parser.parse_args()
    try:
        if a.op=="build":
            if a.authorized_submit:
                require(os.environ.get("MASTER01EV3_RIGHTS")=="true" and
                        os.environ.get("MASTER01EV3_GPU")=="true" and
                        os.environ.get("MASTER01EV3_MODEL_RIGHTS")=="true" and
                        os.environ.get("GITHUB_RUN_ATTEMPT")=="1","AUTHORIZATION_REQUIRED")
            build(a.out,a.authorized_submit)
        elif a.op=="preflight": preflight()
        elif a.op=="status": print("MASTER01EV3_KAGGLE_STATUS="+check(KERNEL,"any"))
        elif a.op=="verify": verify(a.out,a.expected_commit)
        elif a.op=="monitor": monitor(a.out,a.expected_commit)
        elif a.op=="notify": require(a.event in EVENTS,"EVENT");telegram(a.event)
    except Exception as exc:
        code=str(exc) if type(exc) is ValueError and re.fullmatch("[A-Z_0-9]{1,80}",str(exc)) else "SAFE_FAILURE_REQUIRES_REVIEW"
        print("MASTER01EV3_CONTROLLER_FAILED="+a.op+" | "+code,flush=True)
        sys.exit(1)

if __name__=="__main__": main()
