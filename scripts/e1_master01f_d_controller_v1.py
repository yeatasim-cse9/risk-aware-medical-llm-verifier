"""01F-D private Kaggle entrypoint, offline inputs and immutable checkout.
No Kaggle push, automatic retry, network acquisition or public artifact upload.
Actual private bundle and trust anchor must be provisioned independently.
"""
import argparse,hashlib,importlib.metadata,json,math,os,re,signal,subprocess,sys,time
from pathlib import Path
import e1_master01f_d_inputs_v1 as inp
import e1_master01f_d_pilot_v1 as pilot
import e1_master01e_v4_neural as n
from e1_master01e_v4_public_cpu_diagnostic import PIN,DIGEST,EXPECTED
from e1_master01e_v4_public_gpu_diagnostic import qualify_engine

BASE="f4ab5bed8e7cbff98b1b56aa620371748b131b9b"
ROOT=Path(__file__).resolve().parents[1]
REPORT="master01fd_private_aggregate.json"
MAX_SECONDS=5400
FALSE=("clinical_validation","test_used","calibration_used","public_export_approved",
       "final_scientific_approval","iid_groups_verified","independent_provenance_certified")
require=inp.require

def git(*args):
    return subprocess.check_output(["git",*args],cwd=ROOT,stderr=subprocess.DEVNULL)

def verify_code(commit):
    require(type(commit) is str and re.fullmatch(r"[0-9a-f]{40}",commit)
            and git("rev-parse","HEAD").decode().strip()==commit,"SOURCE_COMMIT")
    git("merge-base","--is-ancestor",BASE,commit)
    # Pin all checked-in Python dependencies (including unchanged historical modules).
    paths=git("ls-tree","-r","--name-only",commit,"scripts").decode().splitlines()
    require(bool(paths),"SOURCE_TREE")
    for path in paths:
        if not path.endswith(".py"):continue
        p=ROOT/path
        require(p.is_file() and not p.is_symlink() and p.read_bytes()==git("show",commit+":"+path),"SOURCE_DRIFT")
        if not p.name.startswith("e1_master01f_d_"):
            require(p.read_bytes()==git("show",BASE+":"+path),"DEPENDENCY_DRIFT")
    present={p.relative_to(ROOT).as_posix() for p in (ROOT/"scripts").glob("*.py")}
    require(present=={p for p in paths if p.endswith(".py") and p.count("/")==1},"EXTRA_MODULE")
    return commit

def digest(value):return inp.sha(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode())

def methods():
    conditioned={"neural","hybrid","hybrid_legacy","evidence_only","legacy_lexical","full_lexical"}
    return {m+"/"+k for m in pilot.MODES for k in (pilot.CONDITIONS if m in conditioned else ("full",))}|{
        m+"/removed_retrained" for m in ("neural","hybrid","legacy_lexical")}|{
        "nli_contradiction/full","nli_non_entailment/full"}

def check_metrics(ms,parents):
    require(type(ms) is dict and set(ms)==methods(),"METRICS_SCHEMA")
    for v in ms.values():
        require(type(v) is dict and set(v)=={"parents","average_precision","paired_ranking","brier"}
                and type(v["parents"]) is int and v["parents"]==parents
                and all(pilot.prob(v[k]) for k in ("average_precision","paired_ranking","brier")),"METRIC")

def validate_result(v,commit,manifest_pin):
    require(type(v) is dict and set(v)=={"schema","kind","source_commit","manifest_sha256",
            "model_manifest_sha256","exposure","fit_parents","selection_parents","metrics",
            "comparisons","equal_length_metrics","budget","integrity_sha256",*FALSE},"REPORT_SCHEMA")
    require(type(v["schema"]) is int and v["schema"]==1 and v["kind"]=="MASTER01FD_EXPLORATORY_V1"
            and v["source_commit"]==commit and v["manifest_sha256"]==manifest_pin
            and v["model_manifest_sha256"]==DIGEST and v["exposure"]==inp.EXPOSURE
            and type(v["fit_parents"]) is int and v["fit_parents"]==inp.LIMITS["fit"]
            and type(v["selection_parents"]) is int and v["selection_parents"]==inp.LIMITS["selection"]
            and all(v[k] is False for k in FALSE),"REPORT_SCOPE")
    require(v["integrity_sha256"]==digest({k:x for k,x in v.items() if k!="integrity_sha256"}),"REPORT_HASH")
    ms=v["metrics"];check_metrics(ms,inp.LIMITS["selection"])
    sub=v["equal_length_metrics"]
    if sub is not None:
        require(type(sub) is dict and set(sub)==methods(),"SUBSET_SCHEMA")
        size=next(iter(sub.values()))["parents"]
        require(type(size) is int and 10<=size<=inp.LIMITS["selection"],"SUBSET_SIZE")
        check_metrics(sub,size)
    c=v["comparisons"]
    require(type(c) is dict and set(c)=={"groups","parents","replicates","contrasts","exploratory",
            "multiplicity_adjusted","public_export_approved"} and
            type(c["groups"]) is int and type(c["parents"]) is int and
            c["groups"]==c["parents"]==inp.LIMITS["selection"] and
            type(c["replicates"]) is int and c["replicates"]==1000 and c["exploratory"] is True
            and c["multiplicity_adjusted"] is False and c["public_export_approved"] is False
            and type(c["contrasts"]) is list and len(c["contrasts"])==len(pilot.SUITE_CONTRASTS),"COMPARISON_SCHEMA")
    for row,(a,b) in zip(c["contrasts"],pilot.SUITE_CONTRASTS):
        require(type(row) is dict and set(row)=={"first","second","metrics"} and
                row["first"]==a and row["second"]==b and
                type(row["metrics"]) is dict and set(row["metrics"])=={"average_precision","paired_ranking","brier"},"CONTRAST")
        for k,x in row["metrics"].items():
            require(type(x) is dict and set(x)=={"difference","percentile_95"} and
                    type(x["difference"]) in (float,int) and
                    math.isclose(x["difference"],ms[a][k]-ms[b][k],abs_tol=1e-12)
                    and type(x["percentile_95"]) is list and len(x["percentile_95"])==2
                    and all(type(z) in (int,float) and math.isfinite(z) and -1<=z<=1 for z in x["percentile_95"])
                    and x["percentile_95"][0]<=x["percentile_95"][1],"CONTRAST_VALUE")
    require(type(v["budget"]) is dict and set(v["budget"])==set(inp.LIMITS),"BUDGET_SCHEMA")
    for part,counts in v["budget"].items():
        require(type(counts) is dict and set(counts)==set(pilot.CONDITIONS),"BUDGET_CONDITIONS")
        total=inp.LIMITS[part]*2*n.SLOTS
        for cond,x in counts.items():
            require(type(x) is dict and set(x)=={"forward_examples","padded_positions","active_tokens","real_windows"}
                    and all(type(z) is int for z in x.values()) and x["forward_examples"]==total
                    and x["padded_positions"]==total*n.MAX_LEN and 0<x["active_tokens"]<=x["padded_positions"]
                    and (x["real_windows"]==0 if cond=="removed" else inp.LIMITS[part]*2<=x["real_windows"]<=total),"BUDGET_VALUE")
    return v

def envelope(result,commit,pin):
    require(type(result) is dict and set(result)=={"metrics","comparisons","equal_length_metrics","budget"},"RESULT_SCHEMA")
    v={"schema":1,"kind":"MASTER01FD_EXPLORATORY_V1","source_commit":commit,"manifest_sha256":pin,
       "model_manifest_sha256":DIGEST,"exposure":inp.EXPOSURE,
       "fit_parents":inp.LIMITS["fit"],"selection_parents":inp.LIMITS["selection"],
       **result,**dict.fromkeys(FALSE,False)}
    v["integrity_sha256"]=digest(v)
    return validate_result(v,commit,pin)

def write_result(out,result,commit,pin):
    # Validate the exact serialized writer output, not merely an inner fixture.
    v=envelope(result,commit,pin)
    raw=json.dumps(v,sort_keys=True,allow_nan=False).encode()+b"\n"
    require(len(raw)<100000,"REPORT_SIZE")
    validate_result(inp.parse(raw),commit,pin)
    fd=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as f:f.write(raw);f.flush();os.fsync(f.fileno())

def private_preflight(bundle,assets,commit,pin):
    verify_code(commit)
    chosen,_=inp.preflight(bundle,pin)
    n.check_assets(assets,PIN,DIGEST)
    pilot.validate_plan(pilot.Plan(chosen))
    return chosen

def main():
    p=argparse.ArgumentParser()
    p.add_argument("op",choices=("preflight","run"))
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--assets",type=Path,required=True)
    a=p.parse_args()
    # Suppress raw diagnostics including library logs. Only fixed markers exit.
    sys.stdout.flush();sys.stderr.flush();fd=os.dup(1)
    with open(os.devnull,"w") as sink:os.dup2(sink.fileno(),1);os.dup2(sink.fileno(),2)
    stages=("PREFLIGHT","GPU_SMOKE","LOAD_SELECTED","AUDIT_FIT","AUDIT_SELECTION",
            "INFER_FIT","INFER_SELECTION","METRICS","WRITE","COMPLETE")
    state="PREFLIGHT"
    def mark(s):
        nonlocal state
        require(s in stages,"STAGE");state=s
        os.write(fd,("MASTER01FD_STAGE="+s+"\n").encode("ascii"))
    try:
        commit=os.environ.get("MASTER01FD_SOURCE_COMMIT")
        pin=os.environ.get("MASTER01FD_CUSTODIAN_MANIFEST_SHA256")
        mark("PREFLIGHT")
        require(os.environ.get("MASTER01FD_PRIVATE_ENVIRONMENT")=="YES","PRIVATE_ENVIRONMENT_REQUIRED")
        chosen=private_preflight(a.bundle,a.assets,commit,pin)
        if a.op=="preflight":
            os.write(fd,b"MASTER01FD_READONLY_PREFLIGHT_PASSED_METADATA_ONLY\n");os._exit(0)
        require(os.environ.get("MASTER01FD_AUTHORIZED_ONE_GPU_PILOT")=="YES","AUTHORIZATION_REQUIRED")
        require(Path("/kaggle/working").is_dir() and not Path("/kaggle/working").is_symlink(),"PRIVATE_OUTPUT")
        out=Path("/kaggle/working")/REPORT
        require(not out.exists() and not out.is_symlink(),"DO_NOT_OVERWRITE")
        # Local exclusive receipt prevents accidental repeat in same writable session.
        lock=Path("/kaggle/working")/".master01fd-attempted"
        lockfd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);os.close(lockfd)
        # External private session timeout must also be configured; Python signals
        # cannot preempt every native GPU call.
        def timeout(*_):raise TimeoutError("TIME_LIMIT")
        signal.signal(signal.SIGALRM,timeout);signal.alarm(MAX_SECONDS)
        deadline=time.monotonic()+MAX_SECONDS
        os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
        os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
        mark("GPU_SMOKE")
        import torch
        require({k:importlib.metadata.version(k) for k in EXPECTED}==EXPECTED,"RUNTIME_VERSION_DRIFT")
        require(torch.cuda.is_available() and torch.cuda.device_count()==2 and
                all(torch.cuda.get_device_name(i)=="Tesla T4" for i in range(2)),"T4_X2_REQUIRED")
        torch.manual_seed(20261125);torch.cuda.manual_seed_all(20261125)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        engine=n.load_offline(a.assets,PIN,DIGEST,authorized=True,device="cuda:0")
        qualify_engine(engine)
        mark("LOAD_SELECTED")
        blocks=inp.load_selected(a.bundle,pin,authorized=True)
        result=pilot.run(engine,pilot.Plan(chosen),blocks,authorized=True,deadline=deadline,mark=mark)
        mark("WRITE");write_result(out,result,commit,pin);mark("COMPLETE")
        signal.alarm(0);os._exit(0)
    except BaseException:
        os.write(fd,("MASTER01FD_FAILED_STAGE="+state+"\n").encode("ascii"));os._exit(1)
if __name__=="__main__":main()
