"""Versioned recovery controller. Synthetic/default build cannot execute a pilot.
No automatic retries, no raw Kaggle logs, no new scientific approval.
"""
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
import e1_master01e_v3_workflow as old
import e1_master01e_v3_validate as validator
import e1_master01c_core as core
from kaggle_status import check

ROOT=old.ROOT
ORIGINAL="2f62bbbb7f4b08350317a9c6d9ec810493ca23a8"
KERNEL="yeatasimbillah/medical-llm-e1-master01e-v3-recovery-gpu-v1"
FAILED=old.KERNEL
REPORT="master01ev3r_private_aggregate.json"
VERIFIED="master01ev3r_verified_aggregate.json"
MANIFEST_DIGEST="a1d1677a97ca4a7b6a57c0f0e369c54a53eca014bf5daf1791122d9d7642b1e8"
EXTRA=("e1_master01e_v3_recovery_diagnostics","e1_master01e_v3_recovery_private")
RECOVERY_FILES=tuple("scripts/"+x+".py" for x in EXTRA)+(
    "scripts/e1_master01e_v3_recovery_workflow.py",
    ".github/workflows/e1-master01e-v3-recovery-private-pilot.yml")
EVENTS=("controller started","submission accepted","controller failed",
        "Kaggle runtime RUNNING observed","Kaggle terminal failure",
        "Monitoring timeout | no resubmission",
        "Kaggle COMPLETE | aggregate independently verified | exploratory pilot only")
require=core.require

def git_bytes(commit,path):
    return subprocess.check_output(["git","show",commit+":"+path],cwd=ROOT,stderr=subprocess.DEVNULL)

def sha(b):return hashlib.sha256(b).hexdigest()

def source(commit=None):
    commit,bodies,hashes,checkpoint,manifest,md=old.source(commit)
    require(md==MANIFEST_DIGEST,"MODEL_MANIFEST_DRIFT")
    for k,body in bodies.items():
        require(body==git_bytes(ORIGINAL,"scripts/"+k+".py"),"ORIGINAL_MODULE_DRIFT")
    for path in (old.CHECKPOINT,old.ARCHIVE,old.MANIFEST,"scripts/kaggle_status.py",
                 "scripts/e1_master01e_v3_workflow.py"):
        require(git_bytes(commit,path)==git_bytes(ORIGINAL,path)
                and (ROOT/path).read_bytes()==git_bytes(commit,path),"ORIGINAL_EVIDENCE_DRIFT")
    recovery={}
    for path in RECOVERY_FILES:
        data=git_bytes(commit,path)
        require((ROOT/path).read_bytes()==data,"RECOVERY_SOURCE_DRIFT")
        recovery[path]=sha(data)
        if path.startswith("scripts/") and Path(path).stem in EXTRA:
            bodies[Path(path).stem]=data
    return commit,bodies,hashes,checkpoint,manifest,md,recovery

def envelope(aggregate,commit,hashes,recovery,md):
    validator.validate(aggregate,commit,hashes,md)
    result={"schema_version":1,"kind":"MASTER01EV3_RECOVERY_V1",
            "recovery_source_commit":commit,"recovery_file_sha256":recovery,
            "aggregate":aggregate}
    result["integrity_sha256"]=core.digest(result)
    return result

def validate_envelope(value,commit,hashes,recovery,md):
    require(type(value) is dict and set(value)=={"schema_version","kind",
        "recovery_source_commit","recovery_file_sha256","aggregate","integrity_sha256"},
        "RECOVERY_SCHEMA")
    require(value==envelope(value["aggregate"],commit,hashes,recovery,md)
            and type(value["schema_version"]) is int,"RECOVERY_INTEGRITY")
    return value

def render(commit,bodies,hashes,checkpoint,manifest,md,recovery,authorized,
           *,work="/kaggle/working",scratch="/kaggle/temp"):
    # work/scratch overrides are used only by isolated invented-fixture tests.
    return r"""import hashlib,json,os,sys,tempfile
from pathlib import Path
bodies=%r
hashes=%r
commit=%r
checkpoint=%r
manifest=%r
md=%r
recovery=%r
authorized=%r
work=Path(%r)
scratch=Path(%r)
report_name=%r
if authorized is not True:
    print("MASTER01EV3R_FAILED_CODE=AUTHORIZATION_REQUIRED",flush=True)
    sys.exit(1)
# Keep raw library and interpreter shutdown diagnostics suppressed until exit.
sys.stdout.flush();sys.stderr.flush()
fd=os.dup(1)
recorder=None
try:
    with open(os.devnull,"w") as sink:
        os.dup2(sink.fileno(),1);os.dup2(sink.fileno(),2)
    expected=dict(hashes)
    for name in ("e1_master01e_v3_recovery_diagnostics","e1_master01e_v3_recovery_private"):
        expected[name]=recovery["scripts/"+name+".py"]
    if set(bodies)!=set(expected) or any(hashlib.sha256(bodies[k]).hexdigest()!=v for k,v in expected.items()):
        raise ValueError("PACKAGE_PIN")
    if not work.is_dir() or work.is_symlink() or scratch.is_symlink():
        raise ValueError("PRIVATE_STORAGE")
    scratch.mkdir(mode=0o700,exist_ok=True)
    out=work/report_name
    if out.exists() or out.is_symlink():
        raise ValueError("DO_NOT_OVERWRITE")
    with tempfile.TemporaryDirectory(dir=scratch,prefix="master01ev3r-") as tmp:
        pkg=Path(tmp)/"code";pkg.mkdir(mode=0o700)
        root=Path(tmp)/"private";root.mkdir(mode=0o700)
        for k,body in bodies.items():
            (pkg/(k+".py")).write_bytes(body)
        sys.path.insert(0,str(pkg))
        from e1_master01e_v3_recovery_diagnostics import recorder_for_fd
        recorder=recorder_for_fd(fd)
        import e1_master01e_v3_recovery_private as runner
        import e1_master01e_v3_validate as validator
        import e1_master01c_core as core
        os.environ["MASTER01EV3_AUTHORIZED_ONE_PILOT"]="YES"
        os.environ["MASTER01EV3_RECOVERY_AUTHORIZED_ONE_PILOT"]="YES"
        aggregate=runner.run(root,checkpoint,manifest,md,recorder)
        recorder.mark("AGGREGATE_VALIDATE")
        aggregate["package_source_commit"]=commit
        aggregate["package_hashes"]=hashes
        aggregate["integrity_sha256"]=core.digest(aggregate)
        validator.validate(aggregate,commit,hashes,md)
        value={"schema_version":1,"kind":"MASTER01EV3_RECOVERY_V1",
               "recovery_source_commit":commit,"recovery_file_sha256":recovery,
               "aggregate":aggregate}
        value["integrity_sha256"]=core.digest(value)
        payload=json.dumps(value,sort_keys=True,indent=2,allow_nan=False)
        if len(payload.encode())>=100000:
            raise ValueError("REPORT_SIZE")
    recorder.mark("AGGREGATE_WRITE")
    with out.open("x",encoding="utf-8") as f:
        f.write(payload+"\n")
    recorder.mark("COMPLETE")
    os._exit(0)
except BaseException as exc:
    try:
        if recorder is None:
            os.write(fd,b"MASTER01EV3R_FAILED_STAGE=BOOTSTRAP\nMASTER01EV3R_FAILURE_CODE=UNCLASSIFIED\n")
        else:
            recorder.failure(exc)
    except BaseException:
        pass
    os._exit(1)
"""%(bodies,hashes,commit,checkpoint,manifest,md,recovery,authorized,work,scratch,REPORT)

def authorization():
    return (all(os.environ.get(k)=="true" for k in
            ("MASTER01EV3R_RIGHTS","MASTER01EV3R_MODEL_RIGHTS","MASTER01EV3R_GPU"))
            and os.environ.get("GITHUB_RUN_ATTEMPT")=="1"
            and os.environ.get("GITHUB_EVENT_NAME")=="workflow_dispatch"
            and os.environ.get("GITHUB_REF")=="refs/heads/main"
            and os.environ.get("GITHUB_REPOSITORY")=="yeatasim-cse9/risk-aware-medical-llm-verifier")

def build(out,authorized=False):
    require(type(authorized) is bool and (not authorized or authorization()),"AUTHORIZATION_REQUIRED")
    commit,bodies,hashes,checkpoint,manifest,md,recovery=source()
    wrapper=render(commit,bodies,hashes,checkpoint,manifest,md,recovery,authorized)
    compile(wrapper,"runner.py","exec")
    out.mkdir(parents=True,exist_ok=False)
    (out/"runner.py").write_text(wrapper,encoding="utf-8")
    meta={"id":KERNEL,"title":KERNEL.split("/")[1],"code_file":"runner.py","language":"python",
          "kernel_type":"script","is_private":True,"enable_gpu":True,"enable_tpu":False,
          "enable_internet":True,"dataset_sources":[],"competition_sources":[],
          "kernel_sources":[],"model_sources":[]}
    (out/"kernel-metadata.json").write_text(json.dumps(meta,sort_keys=True,indent=2)+"\n")
    print("MASTER01EV3R_PACKAGE_VERIFIED",flush=True)

def preflight():
    source()
    q=subprocess.run(["kaggle","quota"],capture_output=True,timeout=90)
    require(q.returncode==0,"QUOTA_ENDPOINT_UNAVAILABLE")
    require(check(old.PREVIOUS,"any")=="complete","PREVIOUS_RUN_NOT_COMPLETE")
    require(check(FAILED,"any")=="error","FAILED_RUN_NOT_PRESERVED")
    require(check(KERNEL,"any")=="missing","RECOVERY_DUPLICATE_BLOCKED")
    print("MASTER01EV3R_PREFLIGHT_OK_QUOTA_AMOUNT_NOT_CERTIFIED",flush=True)

def telegram(event):
    require(event in EVENTS,"EVENT")
    try:
        token=os.environ["TELEGRAM_BOT_TOKEN"];chat=os.environ["TELEGRAM_CHAT_ID"]
        body=urllib.parse.urlencode({"chat_id":chat,"text":"E1 MASTER 01E V3 RECOVERY | "+event}).encode()
        with urllib.request.urlopen(urllib.request.Request(
                "https://api.telegram.org/bot"+token+"/sendMessage",data=body),timeout=15) as resp:
            ok=json.loads(resp.read(10000)).get("ok") is True
    except Exception:
        ok=False
    print("MASTER01EV3R_TELEGRAM_"+("DELIVERED" if ok else "UNAVAILABLE"),flush=True)

def verify(out,commit):
    require(bool(commit),"EXPECTED_SUBMISSION_COMMIT_REQUIRED")
    commit,_,hashes,_,_,md,recovery=source(commit)
    require(check(KERNEL,"any")=="complete","REMOTE_NOT_COMPLETE")
    with tempfile.TemporaryDirectory() as tmp:
        p=subprocess.run(["kaggle","kernels","output",KERNEL,"-p",tmp,
                         "--file-pattern","^master01ev3r_private_aggregate[.]json$"],
                         capture_output=True,timeout=180)
        require(p.returncode==0,"REMOTE_AGGREGATE_DOWNLOAD_FAILED")
        path=Path(tmp)/REPORT
        require(path.is_file() and not path.is_symlink() and 0<path.stat().st_size<100000,
                "REPORT_FILE_POLICY")
        value=validate_envelope(json.loads(path.read_text()),commit,hashes,recovery,md)
    out.mkdir(parents=True,exist_ok=False)
    (out/VERIFIED).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("MASTER01EV3R_AGGREGATE_VERIFIED_EXPLORATORY_ONLY",flush=True)

def monitor(out,commit):
    seen=False;deadline=time.monotonic()+240*60
    while time.monotonic()<deadline:
        state=check(KERNEL,"any")
        if state=="running" and not seen:
            telegram("Kaggle runtime RUNNING observed");seen=True
        if state=="complete":
            verify(out,commit);return
        if state in {"error","failed","cancelled","cancelacknowledged"}:
            telegram("Kaggle terminal failure")
            raise ValueError("REMOTE_TERMINAL_FAILURE")
        time.sleep(60)
    telegram("Monitoring timeout | no resubmission")
    raise ValueError("MONITOR_TIMEOUT_NO_RESUBMIT")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("op",choices=["build","preflight","status","verify","monitor","notify"])
    p.add_argument("--out",type=Path,default=Path("build/master01ev3r"))
    p.add_argument("--expected-commit",default="")
    p.add_argument("--authorized-submit",action="store_true")
    p.add_argument("--event",choices=EVENTS)
    a=p.parse_args()
    try:
        if a.op=="build":build(a.out,a.authorized_submit)
        elif a.op=="preflight":preflight()
        elif a.op=="status":print("MASTER01EV3R_KAGGLE_STATUS="+check(KERNEL,"any"),flush=True)
        elif a.op=="verify":verify(a.out,a.expected_commit)
        elif a.op=="monitor":monitor(a.out,a.expected_commit)
        elif a.op=="notify":telegram(a.event)
    except Exception as exc:
        codes={"AUTHORIZATION_REQUIRED","SOURCE_COMMIT","SOURCE_NOT_IN_HISTORY",
               "ORIGINAL_MODULE_DRIFT","ORIGINAL_EVIDENCE_DRIFT","RECOVERY_SOURCE_DRIFT",
               "MODEL_MANIFEST_DRIFT","QUOTA_ENDPOINT_UNAVAILABLE","PREVIOUS_RUN_NOT_COMPLETE",
               "FAILED_RUN_NOT_PRESERVED","RECOVERY_DUPLICATE_BLOCKED","REMOTE_NOT_COMPLETE",
               "EXPECTED_SUBMISSION_COMMIT_REQUIRED","REMOTE_AGGREGATE_DOWNLOAD_FAILED",
               "REPORT_FILE_POLICY","RECOVERY_SCHEMA","RECOVERY_INTEGRITY",
               "REMOTE_TERMINAL_FAILURE","MONITOR_TIMEOUT_NO_RESUBMIT"}
        code=exc.args[0] if type(exc) is ValueError and len(exc.args)==1 and type(exc.args[0]) is str and exc.args[0] in codes else "SAFE_FAILURE_REQUIRES_REVIEW"
        print("MASTER01EV3R_CONTROLLER_FAILED="+a.op+" | "+code,flush=True)
        sys.exit(1)

if __name__=="__main__":main()
