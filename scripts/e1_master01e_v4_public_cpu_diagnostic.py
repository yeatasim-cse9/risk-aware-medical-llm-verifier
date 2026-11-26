"""PUBLIC synthetic model compatibility diagnostic only; no medical rows or GPU.
Run in a NEW private Kaggle CPU notebook with Internet enabled and no inputs.
Downloads only the six approved immutable public model files and verifies bytes.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile
import traceback
import e1_master01e_v4_neural as n

PIN={"model":"cross-encoder/nli-deberta-v3-base",
     "revision":"6c749ce3425cd33b46d187e45b92bbf96ee12ec7",
     "files":{
       "added_tokens.json":"a4b6bfe668f2b3cf6f0cd535e98a0663d2d0d4a4a15f13075ad3597d33985a23",
       "config.json":"897e756eb59d3183adb505952e7910e7cbc7750a43f3b3747a96b688d2b02a47",
       "model.safetensors":"d8148c6d49e0a7925134294c56326c71fe0ab1dc390e37355e00c7efbb488afa",
       "special_tokens_map.json":"ed7c099c988dbb414b18a6980d20cb57b91b7cd119f6f6941eb364b0e892e712",
       "spm.model":"c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd",
       "tokenizer_config.json":"f3eecd07c370ef0bf7dd3780d3cd68cf9c8b00c267e21a208ddcd8f82bfec1a6"}}
DIGEST="a1d1677a97ca4a7b6a57c0f0e369c54a53eca014bf5daf1791122d9d7642b1e8"
EXPECTED={"torch":"2.11.0+cu130","transformers":"5.20.2","huggingface_hub":"1.33.0",
          "numpy":"2.1.3","pyarrow":"23.0.1"}
LEGACY=("encode","build_inputs_with_special_tokens","create_token_type_ids_from_sequences")
STAGE="BOOTSTRAP"

def stage(s):
    global STAGE
    STAGE=s
    print("MASTER01EV4_CPU_STAGE="+s,flush=True)

def main():
    os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"]="1"
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    stage("RUNTIME")
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    versions={k:importlib.metadata.version(k) for k in EXPECTED}
    print("MASTER01EV4_CPU_VERSIONS="+json.dumps(versions,sort_keys=True),flush=True)
    exact=versions==EXPECTED
    print("MASTER01EV4_CPU_RUNTIME_EXACT="+str(exact).upper(),flush=True)
    torch.set_num_threads(max(1,min(4,os.cpu_count() or 1)))
    torch.manual_seed(20261125)
    torch.use_deterministic_algorithms(True)
    with tempfile.TemporaryDirectory(prefix="master01ev4-public-") as tmp:
        root=Path(tmp)/"assets";root.mkdir(mode=0o700)
        stage("PINNED_ASSETS")
        for k in sorted(PIN["files"]):
            p=hf_hub_download(PIN["model"],k,revision=PIN["revision"],token=False,
                              cache_dir=str(Path(tmp)/"cache"))
            shutil.copyfile(p,root/k)
        n.check_assets(root,PIN,DIGEST)
        stage("LEGACY_API_CHECK")
        tok=AutoTokenizer.from_pretrained(str(root),local_files_only=True,
                                           trust_remote_code=False,use_fast=False)
        print("MASTER01EV4_CPU_TOKENIZER="+type(tok).__name__,flush=True)
        for key in LEGACY:
            print("MASTER01EV4_CPU_LEGACY_"+key.upper()+"="+
                  str(callable(getattr(tok,key,None))).upper(),flush=True)
        # Reproduce the old tokenizer-only path using invented text. The old
        # exception is diagnostic evidence, not a reason to bypass new checks.
        probe="encode"
        try:
            x=tok.encode("An invented object is blue.",add_special_tokens=False,
                         truncation=False,verbose=False)
            y=tok.encode("The object is blue.",add_special_tokens=False,
                         truncation=False,verbose=False)
            probe="build_inputs_with_special_tokens"
            tok.build_inputs_with_special_tokens(x,y)
            if "token_type_ids" in tok.model_input_names:
                probe="create_token_type_ids_from_sequences"
                tok.create_token_type_ids_from_sequences(x,y)
        except Exception as exc:
            print("MASTER01EV4_CPU_OLD_PATH_FAILED_AT="+probe,flush=True)
            print("MASTER01EV4_CPU_OLD_PATH_FAILURE_TYPE="+type(exc).__name__,flush=True)
            name=getattr(exc,"name",None)
            if type(exc) is AttributeError and name in LEGACY:
                print("MASTER01EV4_CPU_OLD_PATH_MISSING_ATTRIBUTE="+name,flush=True)
        else:
            print("MASTER01EV4_CPU_OLD_PATH_TOKENIZER_CALLS_PASSED",flush=True)
        del tok
        stage("MODEL_LOAD_AND_PAIR_CONTRACT")
        e=n.load_offline(root,PIN,DIGEST,authorized=True,device="cpu")
        stage("WINDOW_AND_BOUNDARY_CONTRACT")
        windows,total=e.windows("The invented object is blue. "*80)
        n.require(total>n.WINDOW and len(windows)>1,"MULTIWINDOW_SMOKE")
        try:e.encode([], [e.sep]*(n.MAX_LEN-n.WINDOW-e.special+1))
        except ValueError as exc:n.require(exc.args==("HYPOTHESIS_BUDGET_EXCEEDED",),"BUDGET_SMOKE")
        else:raise ValueError("BUDGET_SMOKE")
        stage("FULL_FORWARD")
        full,c1=e.infer(windows,"The object is blue.",batch_size=4)
        stage("REMOVED_FORWARD")
        removed,c2=e.infer(windows,"The object is blue.",removed=True,batch_size=4)
        stage("PERMUTED_FORWARD")
        perm,_=e.windows("An invented triangle is green. "*80)
        changed,c3=e.infer(perm,"The object is blue.",batch_size=4)
        for f,c in ((full,c1),(removed,c2),(changed,c3)):
            n.require(len(f)==6 and c["forward_examples"]==16 and c["padded_positions"]==8192,
                      "SMOKE_BUDGET")
        n.require(c2["real_windows"]==0 and c1["real_windows"]==len(windows)
                  and c2["active_tokens"]<c1["active_tokens"],"REMOVAL_SMOKE")
        print("MASTER01EV4_CPU_SYNTHETIC_MODEL_SMOKE_PASSED",flush=True)
        print("MASTER01EV4_CPU_DIAGNOSTIC="+("EXACT_RUNTIME_PASS" if exact else "PASS_RUNTIME_DIFFERS"),flush=True)
        print("MASTER01EV4_CPU_MEDICAL_DATA_USED=FALSE",flush=True)

if __name__=="__main__":
    try:main()
    except Exception as exc:
        print("MASTER01EV4_CPU_FAILED_STAGE="+STAGE,flush=True)
        print("MASTER01EV4_CPU_FAILURE_TYPE="+type(exc).__name__,flush=True)
        if type(exc) is AttributeError and getattr(exc,"name",None) in LEGACY:
            print("MASTER01EV4_CPU_MISSING_LEGACY_API="+exc.name,flush=True)
        if type(exc) is ValueError and len(exc.args)==1 and type(exc.args[0]) is str and exc.args[0].isupper() and exc.args[0].replace("_","").isalnum() and len(exc.args[0])<=64:
            print("MASTER01EV4_CPU_FAILURE_CODE="+exc.args[0],flush=True)
        for f in traceback.extract_tb(exc.__traceback__)[-5:]:
            print("MASTER01EV4_CPU_FRAME="+Path(f.filename).name+":"+str(f.lineno)+":"+f.name,flush=True)
        raise SystemExit(1) from None
