"""One-shot private CPU controller; no private results in public logs/artifacts."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from kaggle_status import check, ACTIVE

def require(ok):
    if not ok:
        raise ValueError("R13G_GATE_FAILED")

def notify(event):
    try:
        token=os.environ["TELEGRAM_BOT_TOKEN"]
        chat=os.environ["TELEGRAM_CHAT_ID"]
        require(re.fullmatch(r"[0-9]+:[A-Za-z0-9_-]+",token) is not None)
        require(re.fullmatch(r"-?[0-9]+",chat) is not None)
        body=urllib.parse.urlencode({"chat_id":chat,"text":"MASTER 01F-R13-G | "+event}).encode()
        req=urllib.request.Request("https://api.telegram.org/bot"+token+"/sendMessage",data=body)
        with urllib.request.urlopen(req,timeout=15) as r:
            require(json.load(r).get("ok") is True)
        print("R13G_TELEGRAM_API_ACCEPTED",flush=True)
    except Exception:
        print("R13G_TELEGRAM_DELIVERY_UNCONFIRMED",flush=True)

def config():
    c=json.loads(os.environ["R13G_PRIVATE_CONFIG"])
    require(set(c)=={"dataset_source","dataset_sha256","dataset_bytes","inventory_sha256","rows","candidate_articles","candidate_rows"})
    require(re.fullmatch(r"[a-z0-9_-]+/[a-z0-9_-]+",c["dataset_source"]) is not None)
    for k in ("dataset_sha256","inventory_sha256"):
        require(re.fullmatch(r"[a-f0-9]{64}",c[k]) is not None)
    for k in ("dataset_bytes","rows","candidate_articles","candidate_rows"):
        require(type(c[k]) is int and c[k]>0)
    require(c["candidate_articles"]<=c["candidate_rows"]<=c["rows"])
    return c

def cli(*args,timeout=180):
    env={k:v for k,v in os.environ.items() if k not in ("R13G_PRIVATE_CONFIG","TELEGRAM_BOT_TOKEN","TELEGRAM_CHAT_ID","GH_TOKEN")}
    p=subprocess.run(["kaggle",*args],capture_output=True,env=env,timeout=timeout)
    require(p.returncode==0)

def state(kernel):
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        return check(kernel,"any")

def lock():
    require(os.environ.get("GITHUB_RUN_ATTEMPT")=="1")
    url="https://api.github.com/repos/"+os.environ["GITHUB_REPOSITORY"]+"/actions/workflows/r13g-private-cpu.yml/runs?event=workflow_dispatch&per_page=2"
    req=urllib.request.Request(url,headers={"Authorization":"Bearer "+os.environ["GH_TOKEN"],"Accept":"application/vnd.github+json"})
    with urllib.request.urlopen(req,timeout=30) as r:
        data=json.load(r)
    require(data["total_count"]==1 and str(data["workflow_runs"][0]["id"])==os.environ["GITHUB_RUN_ID"])

def package(path,c,kernel):
    code=Path(__file__).with_name("r13g_correction_baseline.py").read_text()
    code+="\ntry:\n    run(json.loads("+repr(json.dumps(c))+"))\nexcept Exception:\n    raise RuntimeError('R13G_PRIVATE_EXECUTION_BLOCKED') from None\n"
    compile(code,"runner.py","exec")
    (path/"runner.py").write_text(code)
    meta={"id":kernel,"title":kernel.split("/")[1],"code_file":"runner.py","language":"python",
          "kernel_type":"script","is_private":True,"enable_gpu":False,"enable_tpu":False,
          "enable_internet":False,"dataset_sources":[c["dataset_source"]],
          "competition_sources":[],"kernel_sources":[],"model_sources":[]}
    (path/"kernel-metadata.json").write_text(json.dumps(meta))

def validate_receipt(p,c):
    require(p.is_file() and not p.is_symlink() and p.stat().st_size<100000)
    r=json.loads(p.read_text())
    require(r["status"]=="EXPLORATORY_BASELINE_COMPLETE")
    for k in ("run_id","source_commit","dataset_sha256","inventory_sha256","candidate_rows"):
        require(r[k]==c[k])
    for k in ("public_export_approved","protected_split_read","gpu_used","model_weights_used","reference_provenance_established","clinical_correctness_claim"):
        require(r[k] is False)
    n=r["evaluation_prompts"]
    require(type(n) is int and n>0 and 30<=r["evaluation_groups"]<=n)
    require(n+r["development_prompts"]+sum(r["excluded_rows"].values())==c["candidate_rows"])
    for arm in ("unchanged","majority","lexical"):
        m=r["metrics"][arm]
        require(m["prompts"]==n and 0<=m["corrupt_corrected"]<=2*n)
        require(m["clean_preserved"]+m["introduced_clean_errors"]==n)

def main():
    c=config()
    c.update(run_id=os.environ["GITHUB_RUN_ID"],source_commit=os.environ["GITHUB_SHA"])
    kernel=c["dataset_source"].split("/")[0]+"/master-01f-r13g-private-cpu-v1"
    lock()
    require(state(kernel)=="missing")
    notify("controller started")
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)
        package(path,c,kernel)
        cli("kernels","push","-p",tmp)
    notify("submission accepted")
    deadline=time.monotonic()+90*60
    seen=False
    while time.monotonic()<deadline:
        s=state(kernel)
        if s=="running" and not seen:
            notify("Kaggle runtime RUNNING observed")
            seen=True
        if s=="complete":
            with tempfile.TemporaryDirectory() as tmp:
                cli("kernels","output",kernel,"-p",tmp,"--file-pattern","^R13G_RECEIPT[.]json$")
                validate_receipt(Path(tmp)/"R13G_RECEIPT.json",c)
            notify("Kaggle COMPLETE | receipt checks passed | exploratory scientific audit pending")
            print("R13G_PRIVATE_RECEIPT_CHECKS_PASS",flush=True)
            return
        require(s in ACTIVE or (s=="missing" and time.monotonic()>deadline-90*60 and time.monotonic()<deadline-88*60))
        time.sleep(30)
    raise ValueError("R13G_TIMEOUT")

if __name__=="__main__":
    try:
        main()
    except Exception:
        notify("controller failed or status uncertain | no automatic retry")
        raise SystemExit("R13G_BLOCKED_NO_RETRY") from None
