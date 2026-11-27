"""R9C: reuse the verified private notebook; no private results in Actions.
Only a manually authorized workflow may push. Default CLI is read-only.
The scientific controller bytes and plan are fixed, not reconstructed.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import urllib.request
from kaggle_status import check, ACTIVE, TERMINAL
import e1_master01fr6r_checkpoint_v1 as cp

REPO="yeatasim-cse9/risk-aware-medical-llm-verifier"
KERNEL="yeatasimbillah/medical-llm-master01fr9-long-context-gpu-v1"
SOURCE_SHA="a9c6e44e4676e59d1472e98d374ed3c7527d4fa72fefa601f5d7ae534c7cb8db"
PLAN_PIN="4370ada0b5c9731f756c127b503e697c1df04e77dca0920daa2931ed8c02d547"
SCIENCE="91d2165461f4e679d1c86e537e79b0efd7d68d09"
PREFIX=cp.PREFIX+"r6r-r9-long-context-001/"
QUAL_FILE="r9c-secret-qualification.json"
QUAL_CODE='''from kaggle_secrets import UserSecretsClient
import json,re
from pathlib import Path
c=UserSecretsClient()
for name,pattern in (("THESIS_PRIVATE_ARCHIVE_TOKEN",r"[^\\s]{10,512}"),("TELEGRAM_BOT_TOKEN",r"[0-9]+:[A-Za-z0-9_-]+"),("TELEGRAM_CHAT_ID",r"-?[0-9]+")):
    v=c.get_secret(name)
    if type(v) is not str or re.fullmatch(pattern,v) is None: raise ValueError("R9C_SECRET_QUALIFICATION_FAILED")
    del v
# The preceding exact R9B CPU preflight also checks archive rights.
p=Path("/kaggle/working/r9c-secret-qualification.json")
with p.open("x") as f: json.dump({"schema":1,"plan_sha256":"'''+PLAN_PIN+'''","source_sha256":"'''+SOURCE_SHA+'''","secret_access":True},f,sort_keys=True)
print("R9C_API_CPU_SECRET_QUALIFICATION_PASSED",flush=True)
'''
def require(ok,code):
    if not ok: raise ValueError(code)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def cli(*args,timeout=120):
    p=subprocess.run(["kaggle",*args],capture_output=True,timeout=timeout)
    require(p.returncode==0,"R9C_KAGGLE_COMMAND_FAILED_OR_UNCERTAIN_NO_RETRY")
    return p.stdout
def cell(text):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":text.splitlines(keepends=True)}
def source(cell):
    require(type(cell) is dict and cell.get("cell_type")=="code","R9C_CELL_TYPE")
    v=cell["source"]
    require(type(v) is str or type(v) is list and all(type(x) is str for x in v),"R9C_CELL_SOURCE")
    return "".join(v)
def read_notebook(folder):
    meta=json.loads((folder/"kernel-metadata.json").read_text())
    require(meta.get("id")==KERNEL and meta.get("is_private") is True
            and meta.get("language")=="python" and meta.get("kernel_type")=="notebook"
            and meta.get("enable_internet") is True
            and meta.get("enable_tpu",False) is False,"R9C_PRIVATE_NOTEBOOK_REQUIRED")
    require(all(meta.get(k,[])==[] for k in ("dataset_sources","competition_sources","kernel_sources","model_sources")),"R9C_ATTACHMENTS_BLOCKED")
    name=meta.get("code_file")
    require(type(name) is str and Path(name).name==name and name.endswith(".ipynb"),"R9C_NOTEBOOK_PATH")
    p=folder/name
    require(p.is_file() and not p.is_symlink() and p.stat().st_size<2_000_000,"R9C_NOTEBOOK_FILE")
    n=json.loads(p.read_text());cells=n.get("cells")
    require(type(cells) is list and len(cells) in (1,2),"R9C_CELL_COUNT")
    raw=source(cells[0])
    # Kaggle may remove only the final newline; accept no other variation.
    if not raw.endswith("\n"):raw+="\n"
    require(sha(raw.encode())==SOURCE_SHA,"R9C_SOURCE_DRIFT")
    qualified=len(cells)==2
    if qualified:
        q=source(cells[1])
        if not q.endswith("\n"):q+="\n"
        require(q==QUAL_CODE,"R9C_QUALIFIER_DRIFT")
    require(meta.get("enable_gpu") is False,"R9C_CPU_SOURCE_REQUIRED")
    return meta,raw,qualified
def pull(folder):
    require(check(KERNEL,"any")=="complete","R9C_NOTEBOOK_NOT_COMPLETE")
    cli("kernels","pull",KERNEL,"-p",str(folder),"-m")
    return read_notebook(folder)
def authorized(kind):
    return (os.environ.get("GITHUB_EVENT_NAME")=="workflow_dispatch"
            and os.environ.get("GITHUB_REPOSITORY")==REPO
            and os.environ.get("GITHUB_REF")=="refs/heads/main"
            and os.environ.get("GITHUB_RUN_ATTEMPT")=="1"
            and os.environ.get("R9C_"+kind)=="true")
def render(raw,mode):
    require(sha(raw.encode())==SOURCE_SHA,"R9C_SOURCE_DRIFT")
    if mode=="gpu":
        for a,b in (('MODE = "PREFLIGHT"','MODE = "EXECUTE"'),
                    ("FRESH_GPU_AUTHORIZATION = False","FRESH_GPU_AUTHORIZATION = True"),
                    ("SUPERVISOR_PROTOCOL_REVIEW_CONFIRMED = False","SUPERVISOR_PROTOCOL_REVIEW_CONFIRMED = True"),
                    ("RIGHTS_AND_QUOTA_RECONFIRMED = False","RIGHTS_AND_QUOTA_RECONFIRMED = True")):
            require(raw.count(a)==1,"R9C_FLAG_BINDING");raw=raw.replace(a,b,1)
    elif mode!="cpu":raise ValueError("R9C_MODE")
    compile(raw,"runner","exec")
    return raw
def package(folder,raw,mode):
    code=render(raw,mode)
    folder.mkdir(exist_ok=False)
    cells=[cell(code)]+([cell(QUAL_CODE)] if mode=="cpu" else [])
    n={"nbformat":4,"nbformat_minor":4,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"cells":cells}
    (folder/"runner.ipynb").write_text(json.dumps(n))
    m={"id":KERNEL,"title":KERNEL.split("/")[1],"code_file":"runner.ipynb",
       "language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":mode=="gpu",
       "enable_tpu":False,"enable_internet":True,"dataset_sources":[],"competition_sources":[],
       "kernel_sources":[],"model_sources":[]}
    if mode=="gpu":m["machine_shape"]="NvidiaTeslaT4"
    (folder/"kernel-metadata.json").write_text(json.dumps(m))
def read_store():
    # Separate least-privilege Contents:read token. Never passed to Kaggle CLI.
    store=cp.PrivateStore(os.environ["THESIS_PRIVATE_ARCHIVE_READ_TOKEN"]);store.verify()
    return store
def fresh(store):
    require(all(store.read(PREFIX+k) is None for k in ("plan.json","complete.json",*[f"stage-{i:03d}.json" for i in range(4)])),"R9C_RUN_EXISTS_NO_RETRY")
def qualify_marker():
    with tempfile.TemporaryDirectory() as tmp:
        cli("kernels","output",KERNEL,"-p",tmp,"--file-pattern","^r9c-secret-qualification[.]json$",timeout=180)
        p=Path(tmp)/QUAL_FILE
        require(p.is_file() and not p.is_symlink() and p.stat().st_size<1000,"R9C_QUAL_MARKER_MISSING")
        require(json.loads(p.read_text())=={"schema":1,"plan_sha256":PLAN_PIN,"source_sha256":SOURCE_SHA,"secret_access":True},"R9C_QUAL_MARKER")
def notify(event,count=0,total=4):
    # Reuse private checkpoint Telegram sender for fixed status labels.
    cp.notify(event,count,total)
def verify(store):
    plan_raw=store.read(PREFIX+"plan.json")
    require(type(plan_raw) is bytes and sha(plan_raw)==PLAN_PIN,"R9C_PLAN")
    plan=cp.validate_plan(json.loads(plan_raw),PLAN_PIN)
    require(plan["source_commit"]==SCIENCE and plan["scope"]=="private_development_aggregate","R9C_SCOPE")
    prev=PLAN_PIN;records=[];values={}
    for i,s in enumerate(plan["stages"]):
        raw=store.read(PREFIX+f"stage-{i:03d}.json")
        require(type(raw) is bytes,"R9C_CHECKPOINT_MISSING")
        obj=json.loads(raw)
        m=cp.metrics(obj["metrics"],s["metrics"])
        expected={"schema":1,"plan_sha256":PLAN_PIN,"index":i,"stage":s["name"],"previous_sha256":prev,"metrics":m}
        require(raw==cp.canonical(expected),"R9C_CHAIN")
        prev=sha(raw);records.append(prev);values[s["name"]]=m
    require(len(records)==4,"R9C_STAGE_COUNT")
    expected={"schema":1,"plan_sha256":PLAN_PIN,"stage_sha256":records,"aggregates":values,"public_export_approved":False,"scientific_approval":False}
    require(store.read(PREFIX+"complete.json")==cp.canonical(expected),"R9C_COMPLETE")
    # No aggregate values, hashes of results, or output artifacts in public logs.
    print("R9C_PRIVATE_COMPLETE_CHAIN_VERIFIED_SCIENTIFIC_APPROVAL_PENDING",flush=True)
def one_dispatch():
    # Separate submit-only workflow. Any previous dispatch (even failed or
    # cancelled) locks this one-shot lane. No automatic retries.
    url="https://api.github.com/repos/"+REPO+"/actions/workflows/e1-master01fr9c-submit.yml/runs?event=workflow_dispatch&per_page=2"
    req=urllib.request.Request(url,headers={"Authorization":"Bearer "+os.environ["GH_TOKEN"],"Accept":"application/vnd.github+json"})
    with urllib.request.urlopen(req,timeout=30) as r: data=json.load(r)
    require(data["total_count"]==1 and len(data["workflow_runs"])==1 and
            str(data["workflow_runs"][0]["id"])==os.environ["GITHUB_RUN_ID"],"R9C_PRIOR_DISPATCH_LOCKED")
def monitor(store,mode):
    deadline=time.monotonic()+240*60;seen=False
    while time.monotonic()<deadline:
        state=check(KERNEL,"any")
        if state=="running" and not seen:
            print("R9C_KAGGLE_RUNNING",flush=True);seen=True
        if state=="complete":
            if mode=="cpu":qualify_marker()
            else:verify(store)
            return
        require(state in ACTIVE,"R9C_TERMINAL_OR_UNKNOWN_NO_RETRY")
        time.sleep(60)
    raise ValueError("R9C_MONITOR_TIMEOUT_NO_RETRY")
def main():
    a=argparse.ArgumentParser();a.add_argument("op",choices=("preflight","qualify","submit","status","verify"));op=a.parse_args().op
    if op=="status":check(KERNEL,"any");return
    store=read_store()
    if op=="verify":
        require(check(KERNEL,"any")=="complete","R9C_NOTEBOOK_NOT_COMPLETE");verify(store);return
    fresh(store)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);src=root/"source";src.mkdir()
        _,raw,qualified=pull(src)
        if op=="preflight":
            if qualified:qualify_marker()
            print("R9C_READ_ONLY_SOURCE_AND_ARCHIVE_PASS",flush=True)
            print("R9C_API_CPU_SECRET_QUALIFIED="+str(qualified).upper(),flush=True);return
        if op=="qualify":
            require(authorized("CPU") and not qualified,"R9C_CPU_QUALIFICATION_AUTH_REQUIRED")
            mode="cpu"
        else:
            require(authorized("GPU") and os.environ.get("R9C_SUPERVISOR")=="true"
                    and os.environ.get("R9C_RIGHTS_QUOTA")=="true","R9C_GPU_AUTH_REQUIRED")
            require(qualified,"R9C_CPU_QUALIFICATION_REQUIRED");qualify_marker();one_dispatch()
            cli("quota");mode="gpu"
        out=root/"package";package(out,raw,mode)
        fresh(store)
        # Re-read the exact current source immediately before the sole push.
        again=root/"again";again.mkdir()
        _,r,q=pull(again);require(r==raw and q==qualified,"R9C_NOTEBOOK_CHANGED")
        print("R9C_CONTROLLER_STARTED",flush=True);notify("started")
        cmd=["kernels","push","-p",str(out)]
        if mode=="gpu":cmd+=["--accelerator","NvidiaTeslaT4"]
        cli(*cmd,timeout=180)
        print("R9C_SUBMISSION_ACCEPTED_NOT_COMPLETE",flush=True)
        monitor(store,mode)
        print("R9C_"+mode.upper()+"_VERIFIED",flush=True);notify("complete",4,4)
if __name__=="__main__":
    try:main()
    except Exception:
        print("R9C_FAILED_RECONCILE_NO_RETRY",flush=True)
        raise SystemExit(1) from None
