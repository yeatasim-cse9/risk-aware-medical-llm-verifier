"""R9C: reuse the verified private notebook; no private results in Actions.
Only a manually authorized workflow may push. Default CLI is read-only.
The scientific controller bytes and plan are fixed, not reconstructed.
"""
import argparse
import ast
import sys
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import urllib.request
import urllib.parse
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
STAGE="BOOT"
class GateError(ValueError):
    pass
def mark(stage):
    global STAGE
    STAGE=stage
    print("R9C_STAGE="+stage,flush=True)
def require(ok,code):
    if not ok: raise GateError(code)
def report_failure(exc):
    # Never print remote error text, private data, source bytes, paths or secrets.
    print("R9C_FAILED_STAGE="+STAGE,flush=True)
    print("R9C_FAILURE_TYPE="+type(exc).__name__,flush=True)
    if type(exc) is GateError and len(exc.args)==1 and type(exc.args[0]) is str and re.fullmatch(r"R9C_[A-Z0-9_]{1,90}",exc.args[0]):
        print("R9C_FAILURE_CODE="+exc.args[0],flush=True)
    elif isinstance(exc,subprocess.TimeoutExpired):
        print("R9C_FAILURE_CODE=R9C_CLI_TIMEOUT",flush=True)
    elif isinstance(exc,json.JSONDecodeError):
        print("R9C_FAILURE_CODE=R9C_INVALID_JSON",flush=True)
    elif isinstance(exc,FileNotFoundError):
        print("R9C_FAILURE_CODE=R9C_FILE_NOT_FOUND",flush=True)
    else:
        print("R9C_FAILURE_CODE=R9C_UNCLASSIFIED_REQUIRES_EVIDENCE",flush=True)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def cli(*args,timeout=120):
    env={k:v for k,v in os.environ.items() if k not in ("THESIS_PRIVATE_ARCHIVE_READ_TOKEN","TELEGRAM_BOT_TOKEN","TELEGRAM_CHAT_ID","GH_TOKEN")}
    p=subprocess.run(["kaggle",*args],capture_output=True,timeout=timeout,env=env)
    if p.returncode!=0:
        print("R9C_CLI_EXIT_CODE="+str(p.returncode),flush=True)
        # Classify only fixed signatures from argparse/HTTP errors; do not echo CLI text.
        diagnostic=p.stdout+p.stderr
        for code in (401,403,404,429,500,502,503,504):
            if re.search(rb"\b"+str(code).encode()+rb"\b",diagnostic):
                print("R9C_CLI_POSSIBLE_HTTP_STATUS="+str(code),flush=True)
        if b"unrecognized arguments" in diagnostic.lower():
            print("R9C_CLI_ARGUMENT_REJECTION=TRUE",flush=True)
        raise GateError("R9C_KAGGLE_COMMAND_FAILED_OR_UNCERTAIN_NO_RETRY")
    return p.stdout
def cell(text):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":text.splitlines(keepends=True)}
def source(cell):
    require(type(cell) is dict and cell.get("cell_type")=="code","R9C_CELL_TYPE")
    v=cell.get("source")
    require(type(v) is str or type(v) is list and all(type(x) is str for x in v),"R9C_CELL_SOURCE")
    return "".join(v)
def canonical_source(raw,expected):
    # Only accept newline representations that restore the exact pinned bytes.
    # No AST equivalence, whitespace stripping or source changes.
    candidates=[raw,raw.replace("\r\n","\n")]
    # The operator's downloaded notebook showed exactly one extra leading LF.
    # Recover only this observed serialization difference, and ONLY when the
    # resulting bytes reproduce the pre-existing SHA256. Execute those bytes,
    # never the unverified candidate. No arbitrary whitespace stripping.
    candidates += [text[1:] for text in candidates if text.startswith("\n")]
    for text in candidates:
        for candidate in (text,text+"\n" if not text.endswith("\n") else text):
            if sha(candidate.encode())==expected:
                if candidate!=raw:print("R9C_SOURCE_CANONICALIZED_TO_EXACT_PIN",flush=True)
                return candidate
    raise GateError("R9C_SOURCE_DRIFT")
def read_notebook(folder):
    mark("METADATA_READ")
    f=folder/"kernel-metadata.json"
    require(f.is_file() and not f.is_symlink() and 0<f.stat().st_size<100_000,"R9C_METADATA_FILE")
    meta=json.loads(f.read_text(encoding="utf-8"))
    require(type(meta) is dict,"R9C_METADATA_SCHEMA")
    mark("METADATA_POLICY")
    require(meta.get("id")==KERNEL,"R9C_KERNEL_ID_MISMATCH")
    require(meta.get("is_private") is True,"R9C_PRIVATE_NOTEBOOK_REQUIRED")
    require(meta.get("language")=="python","R9C_LANGUAGE_MISMATCH")
    require(meta.get("kernel_type")=="notebook","R9C_KERNEL_TYPE_MISMATCH")
    require(meta.get("enable_internet") is True,"R9C_INTERNET_REQUIRED")
    require(meta.get("enable_tpu",False) is False,"R9C_TPU_BLOCKED")
    require(meta.get("enable_gpu") is False,"R9C_CPU_SOURCE_REQUIRED")
    require(all(meta.get(k,[])==[] for k in ("dataset_sources","competition_sources","kernel_sources","model_sources")),"R9C_ATTACHMENTS_BLOCKED")
    print("R9C_METADATA_POLICY_PASS",flush=True)
    name=meta.get("code_file")
    require(type(name) is str and Path(name).name==name and name.endswith(".ipynb"),"R9C_NOTEBOOK_PATH")
    mark("NOTEBOOK_READ")
    p=folder/name
    require(p.is_file() and not p.is_symlink() and 0<p.stat().st_size<2_000_000,"R9C_NOTEBOOK_FILE")
    n=json.loads(p.read_text(encoding="utf-8"))
    require(type(n) is dict,"R9C_NOTEBOOK_SCHEMA")
    cells=n.get("cells")
    require(type(cells) is list,"R9C_CELL_SCHEMA")
    print("R9C_NOTEBOOK_CELL_COUNT="+str(len(cells)),flush=True)
    require(len(cells) in (1,2),"R9C_CELL_COUNT")
    mark("SOURCE_INTEGRITY")
    raw=canonical_source(source(cells[0]),SOURCE_SHA)
    print("R9C_PINNED_SOURCE_PASS",flush=True)
    qualified=len(cells)==2
    if qualified:
        mark("QUALIFIER_INTEGRITY")
        try:canonical_source(source(cells[1]),sha(QUAL_CODE.encode()))
        except GateError as exc:raise GateError("R9C_QUALIFIER_DRIFT") from exc
    return meta,raw,qualified
def pull(folder):
    mark("KAGGLE_STATUS")
    require(check(KERNEL,"any")=="complete","R9C_NOTEBOOK_NOT_COMPLETE")
    mark("KAGGLE_PULL")
    cli("kernels","pull",KERNEL,"-p",str(folder),"-m")
    print("R9C_KAGGLE_PULL_PASS",flush=True)
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
def notify(event):
    require(event in ("controller started","submission accepted","Kaggle running",
                      "CPU qualification verified","GPU private chain verified",
                      "controller failed; reconcile; no retry"),"R9C_NOTIFY_EVENT")
    try:
        token=os.environ.get("TELEGRAM_BOT_TOKEN","");chat=os.environ.get("TELEGRAM_CHAT_ID","")
        require(re.fullmatch(r"[0-9]+:[A-Za-z0-9_-]+",token) is not None and
                re.fullmatch(r"-?[0-9]+",chat) is not None,"R9C_TELEGRAM_FORMAT")
        data=urllib.parse.urlencode({"chat_id":chat,"text":"MASTER 01F-R9C | "+event}).encode()
        req=urllib.request.Request("https://api.telegram.org/bot"+token+"/sendMessage",data=data)
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),cp.NoRedirect())
        with opener.open(req,timeout=10) as res:
            raw=res.read(32769)
            require(res.status==200 and len(raw)<=32768 and json.loads(raw).get("ok") is True,"R9C_TELEGRAM_RESPONSE")
        print("R9C_TELEGRAM_SENT",flush=True)
    except Exception:
        print("R9C_TELEGRAM_DELIVERY_UNCONFIRMED",flush=True)
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
    deadline=time.monotonic()+240*60;seen=False;complete_since=None
    while time.monotonic()<deadline:
        state=check(KERNEL,"any")
        if state=="running" and not seen:
            print("R9C_KAGGLE_RUNNING",flush=True);notify("Kaggle running");seen=True
        if state=="complete":
            # A previous version's COMPLETE can briefly persist after a push.
            # Require new evidence; poll reads only, never submit again.
            if complete_since is None:complete_since=time.monotonic()
            if mode=="cpu":
                with tempfile.TemporaryDirectory() as tmp:
                    _,_,qualified=pull(Path(tmp))
                if qualified:
                    try:qualify_marker()
                    except ValueError:
                        if time.monotonic()-complete_since>=300:raise
                    else:return
            elif store.read(PREFIX+"complete.json") is not None:
                verify(store);return
            require(time.monotonic()-complete_since<300,"R9C_COMPLETE_WITHOUT_NEW_EVIDENCE")
        else:
            require(state in ACTIVE,"R9C_TERMINAL_OR_UNKNOWN_NO_RETRY")
            complete_since=None
        time.sleep(30)
    raise ValueError("R9C_MONITOR_TIMEOUT_NO_RETRY")
def main():
    a=argparse.ArgumentParser();a.add_argument("op",choices=("preflight","qualify","submit","status","verify"));op=a.parse_args().op
    if op=="status":check(KERNEL,"any");return
    mark("ARCHIVE_AUTHENTICATION")
    store=read_store()
    print("R9C_ARCHIVE_AUTHENTICATION_PASS",flush=True)
    if op=="verify":
        require(check(KERNEL,"any")=="complete","R9C_NOTEBOOK_NOT_COMPLETE");verify(store);return
    mark("ARCHIVE_FRESHNESS")
    fresh(store)
    print("R9C_ARCHIVE_FRESHNESS_PASS",flush=True)
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
        mark("PACKAGE")
        out=root/"package";package(out,raw,mode)
        fresh(store)
        # Re-read the exact current source immediately before the sole push.
        again=root/"again";again.mkdir()
        _,r,q=pull(again);require(r==raw and q==qualified,"R9C_NOTEBOOK_CHANGED")
        print("R9C_CONTROLLER_STARTED",flush=True);notify("controller started")
        cmd=["kernels","push","-p",str(out)]
        if mode=="gpu":cmd+=["--accelerator","NvidiaTeslaT4"]
        mark("PUSH")
        cli(*cmd,timeout=180)
        print("R9C_SUBMISSION_ACCEPTED_NOT_COMPLETE",flush=True);notify("submission accepted")
        mark("MONITOR")
        monitor(store,mode)
        print("R9C_"+mode.upper()+"_VERIFIED",flush=True);notify("CPU qualification verified" if mode=="cpu" else "GPU private chain verified")
if __name__=="__main__":
    try:main()
    except Exception as exc:
        report_failure(exc)
        print("R9C_FAILED_RECONCILE_NO_RETRY",flush=True)
        if len(sys.argv)>1 and sys.argv[1] in ("qualify","submit"):
            notify("controller failed; reconcile; no retry")
        raise SystemExit(1) from None
