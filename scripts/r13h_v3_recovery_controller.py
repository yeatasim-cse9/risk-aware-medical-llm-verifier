"""R13-H V3 transport recovery, using unchanged V2 scientific runner.
Read-only preflight and separately authorized one-shot submission.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time
import urllib.request
import r13h_v2_private_gpu_controller as v2

STAGE="INIT"
V2_RUN="46158800788"
V2_HEAD="cf5c6607a6e0c7ad9a88be028667999dae219307"
V1_RUN="46142643713"
V1_HEAD="4cceea4795889171703e4f718dfc4d97b096561c"

def mark(name):
    global STAGE
    STAGE=name
    print("R13HV3_STAGE="+name,flush=True)

def gate(ok):
    if not ok:
        raise ValueError("R13HV3_GATE")

def gh(path):
    req=urllib.request.Request(
        "https://api.github.com/repos/"+os.environ["GITHUB_REPOSITORY"]+path,
        headers={"Authorization":"Bearer "+os.environ["GH_TOKEN"],"Accept":"application/vnd.github+json"})
    with urllib.request.urlopen(req,timeout=30) as r:
        return json.load(r)

def history():
    for run,head in ((V1_RUN,V1_HEAD),(V2_RUN,V2_HEAD)):
        r=gh("/actions/runs/"+run)
        gate(str(r["id"])==run and r["head_sha"]==head and r["event"]=="workflow_dispatch"
             and r["status"]=="completed" and r["conclusion"]=="failure" and r["run_attempt"]==1)

def lock():
    gate(os.environ["GITHUB_RUN_ATTEMPT"]=="1")
    r=gh("/actions/workflows/r13h-v3-private-gpu.yml/runs?event=workflow_dispatch&per_page=2")
    gate(r["total_count"]==1 and len(r["workflow_runs"])==1 and
         str(r["workflow_runs"][0]["id"])==os.environ["GITHUB_RUN_ID"])

def preflight():
    mark("HISTORICAL_RUNS"); history()
    mark("PRIVATE_CONFIG"); c=v2.config()
    c.update(run_id=os.environ["GITHUB_RUN_ID"],source_commit=os.environ["GITHUB_SHA"])
    mark("MODEL_METADATA"); c["model_manifest"]=v2.get_manifest()
    mark("CLI_INTERFACE"); h=v2.cli("kernels","push","--help").decode()
    gate("--timeout" in h and "--accelerator" in h)
    owner=c["dataset_source"].split("/")[0]
    prefix=owner+"/master-01f-r13h-private-gpu-"
    mark("V1_CANCEL_STATUS"); gate(v2.state(prefix+"v1") in v2.CANCELLED)
    mark("V2_ABSENCE"); gate(v2.state(prefix+"v2")=="missing")
    mark("V3_ABSENCE"); gate(v2.state(prefix+"v3")=="missing")
    # Build in temporary directory without submission or external assets.
    mark("PRIVATE_PACKAGE")
    with tempfile.TemporaryDirectory() as tmp:
        v2.package(Path(tmp),c,prefix+"v3")
        meta=json.loads((Path(tmp)/"kernel-metadata.json").read_text())
        gate(meta["is_private"] is True and meta["enable_gpu"] is True and meta["enable_tpu"] is False
             and meta["dataset_sources"]==[c["dataset_source"]] and meta["model_sources"]==[]
             and meta["competition_sources"]==[] and meta["kernel_sources"]==[])
    mark("PREFLIGHT_PASS")
    return c,prefix+"v3"

def notify(event):
    # Use existing status-only Telegram implementation; identify recovery explicitly.
    v2.notify("V3 recovery | "+event)

def submit():
    mark("ONE_DISPATCH_LOCK"); lock()
    c,kernel=preflight()
    notify("controller started")
    mark("SUBMISSION_ONCE")
    with tempfile.TemporaryDirectory() as tmp:
        v2.package(Path(tmp),c,kernel)
        v2.cli("kernels","push","-p",tmp,"--accelerator","NvidiaTeslaT4","--timeout",str(v2.CAP_SECONDS))
    notify("submission accepted")
    mark("MONITOR")
    start=time.monotonic(); next_notice=1800; seen=False; runtime_start=None
    while time.monotonic()-start<v2.CONTROLLER_SECONDS:
        s=v2.state(kernel); phase=v2.classify(s); now=time.monotonic(); elapsed=now-start
        if phase=="runtime" and not seen:
            seen=True; runtime_start=now
            notify("Kaggle runtime RUNNING observed")
        if phase=="queue" and not seen and elapsed>=next_notice:
            notify("still QUEUED | no runtime observed | no retry")
            next_notice+=1800
        if phase=="complete":
            mark("PRIVATE_RECEIPT_REPLAY")
            with tempfile.TemporaryDirectory() as tmp:
                v2.cli("kernels","output",kernel,"-p",tmp,"--file-pattern","^R13HV2_(RECEIPT|GROUP_STATS)[.]json$")
                v2.verify(Path(tmp),c)
            mark("VERIFIED_COMPLETE")
            notify("Kaggle COMPLETE | paired group receipt verified | exploratory scientific audit pending")
            return
        if phase=="cancelled":
            mark("REMOTE_CANCELLED"); raise ValueError("CANCELLED")
        gate(phase in ("queue","runtime") or (s=="missing" and elapsed<120))
        if runtime_start is not None and now-runtime_start>v2.CAP_SECONDS+600:
            break
        time.sleep(30)
    mark("MONITOR_DEADLINE")
    raise ValueError("DEADLINE")

def main():
    a=argparse.ArgumentParser()
    a.add_argument("mode",choices=("preflight","submit"))
    mode=a.parse_args().mode
    try:
        if mode=="preflight":
            preflight()
            print("R13HV3_READONLY_PASS_NO_SUBMISSION",flush=True)
        else:
            submit()
    except Exception:
        # Fixed stage labels only; no exception content or private input printed.
        print("R13HV3_BLOCKED_STAGE="+STAGE,flush=True)
        if mode=="submit":
            notify("blocked at "+STAGE+" | no retry | review Kaggle status")
        raise SystemExit("R13HV3_BLOCKED_NO_RETRY") from None

if __name__=="__main__": main()
