"""Public synthetic GPU qualification; no medical dataset, no pilot submission.
Run only after separate explicit authorization of one GPU qualification.
"""
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile
import traceback
import e1_master01e_v4_neural as n
from e1_master01e_v4_public_cpu_diagnostic import PIN,DIGEST,EXPECTED

STAGE="BOOT"
def stage(name):
    global STAGE
    STAGE=name
    print("MASTER01EV4_GPU_STAGE="+name,flush=True)

def qualify_engine(engine):
    """Same 16 x 512 inference contract for all conditions, fixed inputs."""
    stage("WINDOW_BOUNDARY")
    windows,total=engine.windows("The invented object is blue. "*80)
    n.require(total>n.WINDOW and len(windows)>1,"MULTIWINDOW_SMOKE")
    perm,_=engine.windows("An invented triangle is green. "*80)
    try:
        engine.encode([], [engine.sep]*(n.MAX_LEN-n.WINDOW-engine.special+1))
    except ValueError as exc:
        n.require(exc.args==("HYPOTHESIS_BUDGET_EXCEEDED",),"BUDGET_SMOKE")
    else:raise ValueError("BUDGET_SMOKE")
    results=[]
    for label,parts,removed in (("FULL_FORWARD",windows,False),
                                ("REMOVED_FORWARD",windows,True),
                                ("PERMUTED_FORWARD",perm,False),
                                ("REPEAT_FULL",windows,False)):
        stage(label)
        f,c=engine.infer(parts,"The object is blue.",removed=removed,batch_size=4)
        n.require(len(f)==6 and c["forward_examples"]==16 and c["padded_positions"]==8192,
                  "SMOKE_BUDGET")
        n.require(c["real_windows"]==(0 if removed else len(parts)),"SMOKE_WINDOWS")
        results.append((f,c))
    n.require(results[1][1]["active_tokens"]<results[0][1]["active_tokens"],"REMOVAL_SMOKE")
    n.require(results[0]==results[3],"DETERMINISTIC_REPEAT_MISMATCH")

def main():
    n.require(os.environ.get("MASTER01EV4_PUBLIC_GPU_AUTHORIZED")=="YES","GPU_AUTHORIZATION_REQUIRED")
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"]="1"
    stage("RUNTIME")
    import torch
    from huggingface_hub import hf_hub_download
    versions={k:importlib.metadata.version(k) for k in EXPECTED}
    print("MASTER01EV4_GPU_VERSIONS="+json.dumps(versions,sort_keys=True),flush=True)
    n.require(versions==EXPECTED,"RUNTIME_VERSION_DRIFT")
    n.require(torch.cuda.is_available(),"GPU_REQUIRED")
    print("MASTER01EV4_GPU_DEVICE="+torch.cuda.get_device_name(0),flush=True)
    torch.manual_seed(20261125);torch.cuda.manual_seed_all(20261125)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    with tempfile.TemporaryDirectory(prefix="master01ev4-gpu-public-") as tmp:
        root=Path(tmp)/"assets";root.mkdir(mode=0o700)
        stage("PINNED_ASSETS")
        for k in sorted(PIN["files"]):
            p=hf_hub_download(PIN["model"],k,revision=PIN["revision"],token=False,
                              cache_dir=str(Path(tmp)/"cache"))
            shutil.copyfile(p,root/k)
        n.check_assets(root,PIN,DIGEST)
        stage("MODEL_LOAD_AND_PAIR_CONTRACT")
        e=n.load_offline(root,PIN,DIGEST,authorized=True,device="cuda:0")
        torch.cuda.reset_peak_memory_stats()
        qualify_engine(e)
        torch.cuda.synchronize()
        print("MASTER01EV4_GPU_PEAK_ALLOCATED_BYTES="+str(torch.cuda.max_memory_allocated()),flush=True)
        print("MASTER01EV4_GPU_PEAK_RESERVED_BYTES="+str(torch.cuda.max_memory_reserved()),flush=True)
        del e
        torch.cuda.empty_cache()
    print("MASTER01EV4_GPU_SYNTHETIC_MODEL_SMOKE_PASSED",flush=True)
    print("MASTER01EV4_GPU_DIAGNOSTIC=EXACT_RUNTIME_PASS",flush=True)
    print("MASTER01EV4_GPU_MEDICAL_DATA_USED=FALSE",flush=True)

if __name__=="__main__":
    try:main()
    except Exception as exc:
        print("MASTER01EV4_GPU_FAILED_STAGE="+STAGE,flush=True)
        print("MASTER01EV4_GPU_FAILURE_TYPE="+type(exc).__name__,flush=True)
        if type(exc) is ValueError and len(exc.args)==1 and type(exc.args[0]) is str and exc.args[0].isupper() and exc.args[0].replace("_","").isalnum() and len(exc.args[0])<=64:
            print("MASTER01EV4_GPU_FAILURE_CODE="+exc.args[0],flush=True)
        for f in traceback.extract_tb(exc.__traceback__)[-5:]:
            print("MASTER01EV4_GPU_FRAME="+Path(f.filename).name+":"+str(f.lineno)+":"+f.name,flush=True)
        raise SystemExit(1) from None
