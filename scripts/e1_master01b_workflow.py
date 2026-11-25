"""MASTER01B workflow packaging, fail-closed verification and monitoring."""
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
import e1_master01_validation_kaggle as base
import e1_master01b_kaggle as m

ROOT=Path(__file__).resolve().parents[1]
PREVIOUS_KERNEL="yeatasimbillah/medical-llm-e1-master01b-gpu-v1"
KERNEL="yeatasimbillah/medical-llm-e1-master01b-gpu-v2"
BASE_SHA="11d803b06167689df6a3caa290be69a8b9b519ee9e83ec0d5de619e95a43761e"
RUNNER_SHA="89bdecce4109aa0cc4dc90f1f8a537ce085f13a707c9c0df9258d36625f62e2f"
RUNNER_COMMIT="bdf92f6c940c8f40064d1f44f6bff5dda04dfd5e"
SIZES=m.SIZES
FLAGS=("clinical_adjudication_completed","article_lineage_fully_certified",
       "complete_semantic_leakage_excluded","split_changed","final_split_approved",
       "scientific_e1_experiment_executed","private_rows_exported")
VERSIONS={"torch","transformers","huggingface_hub","numpy","pyarrow"}
MODEL_FILES={"config.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json",
             "added_tokens.json","vocab.txt","model.safetensors"}
EXPECTED_CONFIGS={"pqa_artificial":{"train":7185,"dev":890,"test":899},
                  "pqa_labeled":{"train":394,"dev":57,"test":49}}
EXPECTED_CATEGORIES={"pqa_artificial":{"train":[5472,879,793,41],"dev":[675,115,98,2],"test":[682,117,97,3]},
                     "pqa_labeled":{"train":[300,84,10],"dev":[47,8,2],"test":[33,14,2]}}

def require(ok,code):
    if not ok:
        raise ValueError(code)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def build(out):
    body=(ROOT/"scripts/e1_master01b_kaggle.py").read_bytes()
    previous=(ROOT/"scripts/e1_master01_validation_kaggle.py").read_bytes()
    require(sha(body)==RUNNER_SHA and sha(previous)==BASE_SHA,"SOURCE_HASH_MISMATCH")
    out.mkdir(parents=True,exist_ok=False)
    wrapper="""import hashlib, os, sys, types
old=%r
new=%r
if hashlib.sha256(old).hexdigest()!=%r or hashlib.sha256(new).hexdigest()!=%r:
    raise SystemExit("MASTER01B_SOURCE_HASH_MISMATCH")
os.environ["MASTER01B_PRIVATE_RIGHTS_CONFIRMED"]="YES"
os.environ["MASTER01B_RUNNER_SHA256"]=%r
base=types.ModuleType("e1_master01_validation_kaggle")
sys.modules[base.__name__]=base
exec(compile(old,"base.py","exec"),base.__dict__)
runner=types.ModuleType("master01b_pinned")
sys.modules[runner.__name__]=runner
exec(compile(new,"master01b.py","exec"),runner.__dict__)
try:
    runner.run()
except Exception as exc:
    allowed={"RIGHTS_REQUIRED","KAGGLE_GPU_REQUIRED","PRIVATE_STORAGE","DO_NOT_OVERWRITE",
             "HOLDOUT_PIN","OLD_MAPPING_PIN","MAPPING_PIN","AMBIGUOUS_HOLDOUT","HOLDOUT_COVERAGE",
             "PARENT_COUNTS","EXACT_CONFLICT_REVIEW","MODEL_WEIGHT_PIN_MISMATCH","MODEL_CONTEXT_LIMIT",
             "INVALID_WINDOW_INPUT","WINDOW_LIMIT_EXCEEDED_NO_TRUNCATION_ALLOWED","WINDOW_COVERAGE_FAILURE",
             "TOKENIZER_SPECIAL_TOKEN_POLICY","WINDOW_OVERFLOW","INCOMPLETE_WINDOW_COVERAGE",
             "INVALID_EMBEDDING","INVALID_WINDOW_COUNT","PAIR_SHAPE","PAIR_COVERAGE_FAILURE",
             "NONFINITE_SIMILARITY","REPORT_SIZE","BERT_SPECIAL_ID_POLICY","BERT_FRAMING_MISMATCH",
             "BATCH_WINDOW_POLICY","MODEL_SMOKE_COVERAGE","MODEL_SMOKE_PAIR"}
    code=str(exc) if type(exc) is ValueError else ""
    print("MASTER01B_FAILED_STAGE="+runner.STAGE,flush=True)
    kind=type(exc).__name__
    kinds={"AttributeError","TypeError","ValueError","RuntimeError","NotImplementedError",
           "KeyError","IndexError","ImportError","OSError","OutOfMemoryError","MemoryError",
           "ModuleNotFoundError","HTTPError","TimeoutError"}
    print("MASTER01B_FAILED_KIND="+(kind if kind in kinds else "OTHER"),flush=True)
    print("MASTER01B_FAILED_CODE="+(code if code in allowed else "SAFE_FAILURE_REQUIRES_REVIEW"),flush=True)
    sys.exit(1)
"""%(previous,body,BASE_SHA,RUNNER_SHA,RUNNER_SHA)
    compile(wrapper,"runner.py","exec")
    (out/"runner.py").write_text(wrapper,encoding="utf-8")
    meta={"id":KERNEL,"title":"medical-llm-e1-master01b-gpu-v2","code_file":"runner.py",
          "language":"python","kernel_type":"script","is_private":True,
          "enable_gpu":True,"enable_tpu":False,"enable_internet":True,
          "dataset_sources":[],"competition_sources":[],"kernel_sources":[],"model_sources":[]}
    (out/"kernel-metadata.json").write_text(json.dumps(meta,sort_keys=True,indent=2)+"\n")
    print("MASTER01B_PRIVATE_GPU_PACKAGE_VERIFIED",flush=True)

def telegram(message):
    try:
        token=os.environ["TELEGRAM_BOT_TOKEN"]
        chat=os.environ["TELEGRAM_CHAT_ID"]
        body=urllib.parse.urlencode({"chat_id":chat,"text":"E1 MASTER 01B | "+message}).encode()
        with urllib.request.urlopen(urllib.request.Request(
                "https://api.telegram.org/bot"+token+"/sendMessage",data=body),timeout=15) as r:
            ok=json.loads(r.read(10000)).get("ok") is True
    except Exception:
        ok=False
    print("MASTER01B_TELEGRAM_"+("DELIVERED" if ok else "UNAVAILABLE"),flush=True)
    return ok

def preflight():
    q=subprocess.run(["kaggle","quota"],capture_output=True,timeout=90)
    require(q.returncode==0,"QUOTA_ENDPOINT_UNAVAILABLE")
    require(check(PREVIOUS_KERNEL,"any")=="error","PREVIOUS_V1_NOT_CONFIRMED_ERROR")
    require(check(KERNEL,"any")=="missing","DUPLICATE_MASTER01B_KERNEL_BLOCKED")
    print("MASTER01B_PREFLIGHT_OK_QUOTA_AMOUNT_NOT_CERTIFIED")

def keys(d,names,code="REPORT_SCHEMA"):
    require(type(d) is dict and set(d)==set(names),code)

def integer(v,lo,hi):
    return type(v) is int and lo<=v<=hi

def hexhash(s):
    return isinstance(s,str) and re.fullmatch("[0-9a-f]{64}",s) is not None

def count_triplet(x,a,b,previous=None):
    keys(x,{"pairs","left_rows","right_rows"})
    vals=(x["pairs"],x["left_rows"],x["right_rows"])
    limit=(a*b,a,b) if previous is None else previous
    require(all(integer(v,0,n) for v,n in zip(vals,limit)),"PAIR_COUNT")
    require(max(vals[1:])<=vals[0]<=vals[1]*vals[2] and
            (vals[0]==0)==(vals[1]==0)==(vals[2]==0),"PAIR_COUNT_RELATION")
    return vals

def validate_report(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<100000,"REPORT_FILE_POLICY")
    d=json.loads(path.read_text(encoding="utf-8"))
    keys(d,{"schema_version","status","source_commit","runner_sha256","mapping_sha256",
            "pinned_module_sha256","seed","parents","window_length","window_overlap","device","versions",
            "encoders","exact_screen","label_diagnostics","review_candidates","affected_parents",
            "integrity_sha256",*FLAGS})
    remote=d.pop("integrity_sha256")
    require(hexhash(remote) and sha(base.canonical(d))==remote,"REPORT_DIGEST_MISMATCH")
    require(d["schema_version"]==1 and d["status"]=="SCREEN_COMPLETE_REVIEW_REQUIRED"
      and d["source_commit"]==base.BASE and d["runner_sha256"]==RUNNER_SHA
      and d["mapping_sha256"]==base.EXPECTED_MAPPING and d["pinned_module_sha256"]==base.PINS
      and d["parents"]==SIZES and d["seed"]==base.SEED and d["window_length"]==m.MAX_LENGTH
      and d["window_overlap"]==m.OVERLAP,"REPORT_PIN_MISMATCH")
    require(all(d[k] is False for k in FLAGS),"SCIENTIFIC_OR_PRIVACY_FLAG")
    # Keep actual version strings only if they match a narrow package-version syntax.
    keys(d["versions"],VERSIONS)
    require(all(type(v) is str and re.fullmatch(r"[0-9][0-9A-Za-z.+_-]{0,39}",v)
                for v in d["versions"].values()),"VERSION_FORMAT")
    require(type(d["device"]) is str and len(d["device"])<=80,"DEVICE_FORMAT")
    # Preserve only explicitly recognized GPU model names; unknown names are omitted.
    device=d["device"]
    known={"Tesla T4","Tesla P100-PCIE-16GB","Tesla P100-SXM2-16GB",
           "NVIDIA T4","NVIDIA L4","NVIDIA A100-SXM4-40GB","NVIDIA A100 80GB PCIe",
           "NVIDIA H100 80GB HBM3","NVIDIA GeForce RTX 4090"}
    d["device"]=device if device in known else "UNRECOGNIZED_GPU_NAME_OMITTED"
    pairs={a+"__"+b:(a,b) for a,b in m.PAIRS}
    keys(d["encoders"],m.MODELS)
    keys(d["exact_screen"],{"question","context"})
    for exact in d["exact_screen"].values():
        keys(exact,pairs)
        for x in exact.values():
            require(x=={"shared_groups":0,"left_rows":0,"right_rows":0},"EXACT_CONFLICT")
    flagged={name:[] for name in pairs}
    for name,(model,revision,weight) in m.MODELS.items():
        enc=d["encoders"][name]
        keys(enc,{"model","revision","weights_sha256","file_sha256","views"})
        require((enc["model"],enc["revision"],enc["weights_sha256"])==(model,revision,weight),"MODEL_PIN")
        files=enc["file_sha256"]
        require(type(files) is dict and {"model.safetensors","config.json","tokenizer.json"}<=set(files)<=MODEL_FILES
                and all(hexhash(v) for v in files.values()) and files["model.safetensors"]==weight,"MODEL_FILE_HASHES")
        keys(enc["views"],{"question","context"})
        for view in enc["views"].values():
            keys(view,{"coverage","comparisons"})
            keys(view["coverage"],SIZES)
            keys(view["comparisons"],pairs)
            for s,n in SIZES.items():
                c=view["coverage"][s]
                keys(c,{"parents","windows","total_tokens","covered_tokens","multiwindow_parents",
                        "max_windows_per_parent","payload_tokens"})
                w=c["windows"]
                p=c["payload_tokens"]
                mx=c["max_windows_per_parent"]
                multi=c["multiwindow_parents"]
                require(c["parents"]==n and integer(w,n,n*m.MAX_WINDOWS_PER_PARENT)
                  and integer(mx,1,m.MAX_WINDOWS_PER_PARENT) and integer(p,m.OVERLAP+1,m.MAX_LENGTH)
                  and integer(multi,0,n) and n+multi<=w<=n+multi*(mx-1)
                  and (multi==0)==(mx==1)
                  and integer(c["total_tokens"],n,w*p) and c["total_tokens"]==c["covered_tokens"],
                  "TOKEN_COVERAGE")
            for pair,(a,b) in pairs.items():
                z=view["comparisons"][pair]
                keys(z,{"parent_pairs","window_pairs","maximum_cosine","thresholds"})
                n,p=SIZES[a],SIZES[b]
                require(z["parent_pairs"]==n*p and z["window_pairs"]==view["coverage"][a]["windows"]*view["coverage"][b]["windows"]
                  and type(z["maximum_cosine"]) in (float,int) and -1<=z["maximum_cosine"]<=1,"PAIR_COVERAGE")
                keys(z["thresholds"],{str(t) for t in base.THRESHOLDS})
                previous=None
                for t in base.THRESHOLDS:
                    previous=count_triplet(z["thresholds"][str(t)],n,p,previous)
                    if t==base.THRESHOLDS[0]:
                        flagged[pair].append(previous)
                    # maximum is rounded to seven decimals, allow rounding tolerance.
                    require((previous[0]>0 and z["maximum_cosine"]>=t-1e-6)
                       or (previous[0]==0 and z["maximum_cosine"]<=t+1e-6),"MAXIMUM_THRESHOLD_CONFLICT")
    keys(d["review_candidates"],pairs)
    keys(d["affected_parents"],SIZES)
    per_split={s:[] for s in SIZES}
    for pair,(a,b) in pairs.items():
        x=d["review_candidates"][pair]
        keys(x,{"union_pairs","left_rows","right_rows","lexical_sample"})
        union=count_triplet({"pairs":x["union_pairs"],"left_rows":x["left_rows"],"right_rows":x["right_rows"]},SIZES[a],SIZES[b])
        for j,v in enumerate(union):
            require(max(f[j] for f in flagged[pair])<=v<=sum(f[j] for f in flagged[pair]),"UNION_COUNTS")
        per_split[a].append(union[1])
        per_split[b].append(union[2])
        keys(x["lexical_sample"],{"flagged","unflagged"})
        for label,pop in (("flagged",union[0]),("unflagged",SIZES[a]*SIZES[b]-union[0])):
            y=x["lexical_sample"][label]
            keys(y,{"population_pairs","sample_pairs","question_jaccard_ge_0_5","context_jaccard_ge_0_5"})
            sample=min(pop,m.REVIEW_PER_STRATUM)
            require(y["population_pairs"]==pop and y["sample_pairs"]==sample and
                    all(integer(y[k],0,sample) for k in ("question_jaccard_ge_0_5","context_jaccard_ge_0_5")),
                    "LEXICAL_SAMPLE_COUNTS")
    for s,n in SIZES.items():
        require(integer(d["affected_parents"][s],max(per_split[s]),min(n,sum(per_split[s]))),"AFFECTED_COUNT")
    keys(d["label_diagnostics"],EXPECTED_CONFIGS)
    for c,by_split in d["label_diagnostics"].items():
        keys(by_split,EXPECTED_CONFIGS[c])
        for s,x in by_split.items():
            n=EXPECTED_CONFIGS[c][s]
            keys(x,{"parents","benchmark_role_0","benchmark_role_1","category_frequency_sorted",
                    "difficulty_frequency_sorted","min_observed_category_count"})
            require(x["parents"]==x["benchmark_role_0"]==x["benchmark_role_1"]==n
              and x["category_frequency_sorted"]==EXPECTED_CATEGORIES[c][s]
              and x["min_observed_category_count"]==min(EXPECTED_CATEGORIES[c][s]),"ROLE_COUNTS")
            f=x["difficulty_frequency_sorted"]
            require(type(f) is list and 1<=len(f)<=3 and all(integer(v,1,n) for v in f)
                    and sum(f)==n and sorted(f,reverse=True)==f,"DIFFICULTY_COUNTS")
    d["remote_report_sha256"]=remote
    d["runner_commit"]=RUNNER_COMMIT
    d["limitations"]=["All tokenizer tokens windowed; maximum window similarity has length/window-count bias.",
      "Two fixed encoders and heuristic thresholds are not clinical equivalence or exhaustive semantic recall.",
      "Lexical samples are automated diagnostics only; no human review or split changes.",
      "External article identity and clinical label adjudication still pending."]
    d["integrity_sha256"]=sha(base.canonical(d))
    return d

def verify(out):
    require(check(KERNEL,"any")=="complete","REMOTE_NOT_COMPLETE")
    with tempfile.TemporaryDirectory() as tmp:
        p=subprocess.run(["kaggle","kernels","output",KERNEL,"-p",tmp,
                          "--file-pattern","^master01b_full_context_v2[.]json$"],
                         capture_output=True,timeout=180)
        require(p.returncode==0,"REMOTE_AGGREGATE_DOWNLOAD_FAILED")
        safe=validate_report(Path(tmp)/m.REPORT)
    out.mkdir(parents=True,exist_ok=True)
    (out/"master01b_verified_aggregate.json").write_text(json.dumps(safe,sort_keys=True,indent=2)+"\n")
    print("MASTER01B_REMOTE_AGGREGATE_VERIFIED_SCIENTIFIC_APPROVAL_PENDING")

def monitor(out):
    announced=False
    deadline=time.monotonic()+240*60
    while time.monotonic()<deadline:
        s=check(KERNEL,"any")
        if s=="running" and not announced:
            telegram("Kaggle runtime RUNNING observed")
            announced=True
        if s=="complete":
            verify(out)
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
    p.add_argument("--out",type=Path,default=Path("build/master01b"))
    p.add_argument("--event",choices=["controller started","submission accepted","controller failed",
                                     "Kaggle COMPLETE | aggregate independently verified | scientific approval pending"])
    a=p.parse_args()
    try:
        if a.op=="build": build(a.out)
        elif a.op=="preflight": preflight()
        elif a.op=="status": print("MASTER01B_KAGGLE_STATUS="+check(KERNEL,"any"))
        elif a.op=="verify": verify(a.out)
        elif a.op=="monitor": monitor(a.out)
        elif a.op=="notify": telegram(a.event)
    except Exception as exc:
        code=str(exc) if type(exc) is ValueError and re.fullmatch("[A-Z_0-9]{1,80}",str(exc)) else "SAFE_FAILURE_REQUIRES_REVIEW"
        print("MASTER01B_CONTROLLER_FAILED="+a.op+" | "+code,flush=True)
        sys.exit(1)
