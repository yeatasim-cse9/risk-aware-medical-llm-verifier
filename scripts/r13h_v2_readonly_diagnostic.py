"""Read-only diagnosis of the V2 pre-submission failure; never submits a kernel."""
import contextlib
import io
import json
import os
import re
import subprocess
import urllib.request
from r13h_v2_private_gpu_controller import config, state, CANCELLED
from r13h_model_gate import get_manifest
from kaggle_status import parse_kernel_list, parse_status

def probe(name, fn):
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result=fn()
        print("R13HV2_DIAG_"+name+"="+result,flush=True)
        return result
    except Exception:
        print("R13HV2_DIAG_"+name+"=BLOCKED",flush=True)
        return "BLOCKED"

def cli(*args):
    env={k:v for k,v in os.environ.items() if k not in ("R13G_PRIVATE_CONFIG","R13H_PRIVATE_EXPECTED","GH_TOKEN")}
    return subprocess.run(["kaggle",*args],capture_output=True,text=True,timeout=90,env=env)

def historical_lock():
    u="https://api.github.com/repos/"+os.environ["GITHUB_REPOSITORY"]+"/actions/workflows/r13h-v2-private-gpu.yml/runs?event=workflow_dispatch&per_page=2"
    q=urllib.request.Request(u,headers={"Authorization":"Bearer "+os.environ["GH_TOKEN"],"Accept":"application/vnd.github+json"})
    with urllib.request.urlopen(q,timeout=30) as r:
        j=json.load(r)
    return "PASS" if j["total_count"]==1 and len(j["workflow_runs"])==1 and str(j["workflow_runs"][0]["id"])=="46158800788" else "FAIL"

def config_gate():
    config()
    return "PASS"

def metadata_gate():
    get_manifest()
    return "PASS"

def help_gate():
    p=cli("kernels","push","--help")
    return "PASS" if p.returncode==0 and "--timeout" in p.stdout and "--accelerator" in p.stdout else "FAIL"

def list_gate(kernel):
    p=cli("kernels","list","--mine","--search",kernel.split("/")[1],"--format","json","--page-size","100")
    if p.returncode!=0: return "CLI_ERROR"
    if p.stderr.strip(): return "STDERR_BLOCKED"
    return "PRESENT" if parse_kernel_list(p.stdout,kernel) else "ABSENT"

SAFE={"running","queued","pending","starting","initializing","complete","error","failed","cancelacknowledged","cancelled","canceled","cancel_acknowledged","cancelrequested","cancel_requested","canceling","cancelling","aborted","unknown"}
def status_probe(kernel):
    p=cli("kernels","status",kernel)
    if p.returncode!=0: return "CLI_ERROR"
    raw=p.stdout+"\n"+p.stderr
    try:
        return parse_status(raw).upper()
    except ValueError:
        m=re.search(r'has status\s+["\']?(?:KernelWorkerStatus\.)?([A-Za-z_]+)',raw,re.I)
        if m and m.group(1).lower() in SAFE:
            return "UNSUPPORTED_"+m.group(1).upper()
        return "UNRECOGNIZED_FORMAT"

def main():
    r={}
    r["HISTORICAL_LOCK"]=probe("HISTORICAL_LOCK",historical_lock)
    r["CONFIG"]=probe("CONFIG",config_gate)
    r["MODEL_METADATA"]=probe("MODEL_METADATA",metadata_gate)
    r["CLI_HELP"]=probe("CLI_HELP",help_gate)
    # Fixed previously authorized notebook references. No data downloads.
    owner="yeatasimbillah"
    for version in ("v1","v2"):
        k=owner+"/master-01f-r13h-private-gpu-"+version
        tag=version.upper()
        r[tag+"_LIST"]=probe(tag+"_LIST",lambda:list_gate(k))
        if r[tag+"_LIST"]=="PRESENT":
            r[tag+"_STATUS"]=probe(tag+"_STATUS",lambda:status_probe(k))
        else:
            print("R13HV2_DIAG_"+tag+"_STATUS=NOT_CHECKED",flush=True)
        r[tag+"_CONTROLLER_STATE"]=probe(tag+"_CONTROLLER_STATE",lambda:state(k).upper())
    print("R13HV2_DIAG_READONLY_COMPLETE_NO_GPU_SUBMISSION",flush=True)

if __name__=="__main__": main()
