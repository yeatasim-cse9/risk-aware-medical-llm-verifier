"""One dispatch, bounded GPU runtime, private receipt checks, Telegram status only."""
import contextlib, io, json, os, re, subprocess, tempfile, time, urllib.request, urllib.parse
from pathlib import Path
from kaggle_status import check, ACTIVE
from r13h_model_gate import get_manifest, REV
from r13h_v2_gpu_runner import aggregate, CAP_SECONDS, SOFT_SECONDS
from r13g_correction_baseline import sha

def require(ok):
    if not ok: raise ValueError("R13H_GATE_FAILED")

def notify(event):
    try:
        token=os.environ["TELEGRAM_BOT_TOKEN"]; chat=os.environ["TELEGRAM_CHAT_ID"]
        require(re.fullmatch(r"[0-9]+:[A-Za-z0-9_-]+",token) is not None and re.fullmatch(r"-?[0-9]+",chat) is not None)
        data=urllib.parse.urlencode({"chat_id":chat,"text":"MASTER 01F-R13-H V2 | "+event}).encode()
        req=urllib.request.Request("https://api.telegram.org/bot"+token+"/sendMessage",data=data)
        with urllib.request.urlopen(req,timeout=15) as r: require(json.load(r).get("ok") is True)
        print("R13H_TELEGRAM_API_ACCEPTED",flush=True)
    except Exception: print("R13H_TELEGRAM_DELIVERY_UNCONFIRMED",flush=True)

def config():
    c=json.loads(os.environ["R13G_PRIVATE_CONFIG"])
    require(set(c)=={"dataset_source","dataset_sha256","dataset_bytes","inventory_sha256","rows","candidate_articles","candidate_rows"})
    require(re.fullmatch(r"[a-z0-9_-]+/[a-z0-9_-]+",c["dataset_source"]) is not None)
    for k in ("dataset_sha256","inventory_sha256"): require(re.fullmatch(r"[a-f0-9]{64}",c[k]) is not None)
    for k in ("dataset_bytes","rows","candidate_articles","candidate_rows"): require(type(c[k]) is int and c[k]>0)
    c["expected"]=json.loads(os.environ["R13H_PRIVATE_EXPECTED"])
    e=c["expected"]
    require(e["dataset_sha256"]==c["dataset_sha256"] and e["inventory_sha256"]==c["inventory_sha256"] and e["candidate_rows"]==c["candidate_rows"])
    require(e["status"]=="EXPLORATORY_BASELINE_COMPLETE" and e["protected_split_read"] is False and e["public_export_approved"] is False)
    require(e["development_prompts"]+e["evaluation_prompts"]+sum(e["excluded_rows"].values())==c["candidate_rows"])
    require(e["run_id"]=="46141360290" and e["source_commit"]=="7a6043ea40125e4882e98041d043266fd6fab5f2")
    return c

def cli(*args,timeout=180):
    env={k:v for k,v in os.environ.items() if k not in ("R13G_PRIVATE_CONFIG","R13H_PRIVATE_EXPECTED","TELEGRAM_BOT_TOKEN","TELEGRAM_CHAT_ID","GH_TOKEN")}
    p=subprocess.run(["kaggle",*args],capture_output=True,env=env,timeout=timeout)
    require(p.returncode==0)
    return p.stdout

def state(kernel):
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        return check(kernel,"any")

def lock():
    require(os.environ["GITHUB_RUN_ATTEMPT"]=="1")
    url="https://api.github.com/repos/"+os.environ["GITHUB_REPOSITORY"]+"/actions/workflows/r13h-v2-private-gpu.yml/runs?event=workflow_dispatch&per_page=2"
    req=urllib.request.Request(url,headers={"Authorization":"Bearer "+os.environ["GH_TOKEN"],"Accept":"application/vnd.github+json"})
    with urllib.request.urlopen(req,timeout=30) as r: data=json.load(r)
    require(data["total_count"]==1 and len(data["workflow_runs"])==1 and str(data["workflow_runs"][0]["id"])==os.environ["GITHUB_RUN_ID"])

def package(path,c,kernel):
    root=Path(__file__).parent
    parts=[(root/name).read_text() for name in ("r13g_correction_baseline.py","r13h_model_gate.py","r13h_v2_gpu_runner.py")]
    parts[2]="\n".join(x for x in parts[2].splitlines() if not x.startswith(("from r13g_correction_baseline import ","from r13h_model_gate import ")))
    # Exact reviewed source; only private config embedded in private Kaggle code.
    code="\n".join(parts)+"\ntry:\n    run_h(json.loads("+repr(json.dumps(c))+"))\nexcept Exception:\n    raise RuntimeError('R13H_PRIVATE_EXECUTION_BLOCKED') from None\n"
    compile(code,"runner.py","exec")
    (path/"runner.py").write_text(code)
    meta={"id":kernel,"title":kernel.split("/")[1],"code_file":"runner.py","language":"python","kernel_type":"script",
          "is_private":True,"enable_gpu":True,"enable_tpu":False,"enable_internet":True,
          "dataset_sources":[c["dataset_source"]],"competition_sources":[],"kernel_sources":[],"model_sources":[]}
    (path/"kernel-metadata.json").write_text(json.dumps(meta))

def verify(root,c):
    rp=root/"R13HV2_RECEIPT.json"; gp=root/"R13HV2_GROUP_STATS.json"
    require(all(p.is_file() and not p.is_symlink() and p.stat().st_size<1_000_000 for p in (rp,gp)))
    r=json.loads(rp.read_text()); g=json.loads(gp.read_text()); e=c["expected"]
    require(r["status"]=="EXPLORATORY_LLM_COMPLETE" and r["model_revision"]==REV and r["model_manifest"]==c["model_manifest"])
    require(r["model_weights_verified"] is True and r["baseline_replay"] is True)
    for k in ("run_id","source_commit","dataset_sha256","inventory_sha256"): require(r[k]==c[k])
    for k in ("partition_sha256","evaluation_prompts","evaluation_groups"): require(r[k]==e[k])
    for k in ("protected_split_read","public_export_approved","clinical_correctness_claim","reference_provenance_established","historical_overlap_checked","study_independence_established"): require(r[k] is False)
    require(r["baseline_metrics"]==e["metrics"] and r["group_stats_sha256"]==sha(gp))
    require(r["llm_evaluations"]==6*e["evaluation_prompts"] and len(g)==e["evaluation_groups"])
    m,ci=aggregate(g)
    require(m==r["metrics"] and ci==r["primary_contrast"])
    require(all(x["clean_n"]==e["evaluation_prompts"] and x["corrupt_n"]==2*e["evaluation_prompts"] for x in m.values()))
    require(0<r["runtime_seconds"]<SOFT_SECONDS and r["gpu_count"]==2 and r["gpu_count"]*CAP_SECONDS<=28800 and r["max_input_tokens"]<=1536)
    require(r["execution_version"]=="V2" and len(r["device_evaluations"])==2)
    require(all(type(n) is int and n>0 and n%2==0 for n in r["device_evaluations"]))
    require(sum(r["device_evaluations"])==r["llm_evaluations"] and abs(r["device_evaluations"][0]-r["device_evaluations"][1])<=4)
    require(len(r["device_peak_reserved_bytes"])==2 and min(r["device_peak_reserved_bytes"])>0)

QUEUE_NOTICE_SECONDS=1800
QUEUE_REVIEW_SECONDS=10800
CONTROLLER_SECONDS=19800  # 5h30; hosted job has 5h55 total cap.
CANCELLED={"cancelacknowledged","cancelled","canceled"}

def classify(s):
    if s in ("queued","pending","starting","initializing"): return "queue"
    if s=="running": return "runtime"
    if s=="complete": return "complete"
    if s in CANCELLED: return "cancelled"
    return "other"

def main():
    lock(); c=config()
    c.update(run_id=os.environ["GITHUB_RUN_ID"],source_commit=os.environ["GITHUB_SHA"])
    c["model_manifest"]=get_manifest()
    helptext=cli("kernels","push","--help").decode()
    require("--timeout" in helptext and "--accelerator" in helptext)
    owner=c["dataset_source"].split("/")[0]
    v1=owner+"/master-01f-r13h-private-gpu-v1"
    # Cancellation is independently checked against Kaggle before new push.
    # A completed/active/missing/unknown V1 cannot authorize recovery.
    require(state(v1) in CANCELLED)
    kernel=owner+"/master-01f-r13h-private-gpu-v2"
    require(state(kernel)=="missing")
    notify("V1 cancellation verified | controller started")
    with tempfile.TemporaryDirectory() as tmp:
        package(Path(tmp),c,kernel)
        cli("kernels","push","-p",tmp,"--accelerator","NvidiaTeslaT4","--timeout",str(CAP_SECONDS))
    notify("submission accepted")
    start=time.monotonic(); seen=False; review=False; next_notice=QUEUE_NOTICE_SECONDS
    runtime_start=None
    while time.monotonic()-start<CONTROLLER_SECONDS:
        s=state(kernel); phase=classify(s); now=time.monotonic(); elapsed=now-start
        if phase=="runtime" and not seen:
            notify("Kaggle runtime RUNNING observed"); seen=True; runtime_start=now
        if phase=="queue" and not seen:
            if elapsed>=QUEUE_REVIEW_SECONDS and not review:
                notify("still QUEUED after 3 hours | review only | no resubmission"); review=True
            elif elapsed>=next_notice:
                notify("still QUEUED | no runtime observed | no retry")
                next_notice+=QUEUE_NOTICE_SECONDS
        if phase=="complete":
            with tempfile.TemporaryDirectory() as tmp:
                cli("kernels","output",kernel,"-p",tmp,"--file-pattern","^R13HV2_(RECEIPT|GROUP_STATS)[.]json$")
                verify(Path(tmp),c)
            print("R13HV2_PRIVATE_AGGREGATE_REPLAY_PASS",flush=True)
            notify("Kaggle COMPLETE | paired group receipt verified | exploratory scientific audit pending")
            return
        if phase=="cancelled":
            notify("Kaggle cancellation observed | no retry")
            raise ValueError("R13HV2_CANCELLED")
        require(phase in ("queue","runtime") or (s=="missing" and elapsed<120))
        # The remote session cap is independent of controller wall time.
        if runtime_start is not None and now-runtime_start>CAP_SECONDS+600:
            break
        time.sleep(30)
    notify("monitoring deadline | Kaggle may still be active | check status | no retry")
    raise ValueError("R13HV2_MONITOR_TIMEOUT_NO_RETRY")

if __name__=="__main__":
    try: main()
    except Exception:
        notify("controller blocked or failed | no retry | review Kaggle status")
        raise SystemExit("R13HV2_BLOCKED_NO_RETRY") from None
