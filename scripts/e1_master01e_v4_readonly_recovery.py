"""Recover the completed V4 run, using its immutable producer, without GPU or submission.
The two absent flags are not invented or inserted into the original aggregate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

COMMIT="03b691dde1081d925bf73b705a91b1557a6ebbf7"
RUN_ID=45797892397
KERNEL="yeatasimbillah/medical-llm-e1-master01e-v4-private-pilot-gpu-v1"
REPORT="master01ev4_private_aggregate.json"
KEYS={"schema_version","kind","v4_source_commit","v4_file_sha256","aggregate","integrity_sha256"}

def require(ok,code):
    if not ok:raise ValueError(code)

def sha(raw):return hashlib.sha256(raw).hexdigest()

def strict_object(pairs):
    value={}
    for k,v in pairs:
        require(k not in value,"DUPLICATE_JSON_KEY")
        value[k]=v
    return value

def reject_constant(_):
    raise ValueError("NONFINITE_JSON")

def validate_original(raw,w,commit,hashes,files,md):
    require(type(raw) is bytes and 0<len(raw)<100000,"REPORT_FILE_POLICY")
    value=json.loads(raw,object_pairs_hook=strict_object,parse_constant=reject_constant)
    require(type(value) is dict and set(value)==KEYS,"ORIGINAL_WRITER_SCHEMA")
    require(type(value["schema_version"]) is int and value["schema_version"]==1
            and value["kind"]=="MASTER01EV4_PRIVATE_V1"
            and value["v4_source_commit"]==commit
            and value["v4_file_sha256"]==files,"ORIGINAL_WRITER_PINS")
    payload={k:v for k,v in value.items() if k!="integrity_sha256"}
    require(w.core.digest(payload)==value["integrity_sha256"],"ORIGINAL_WRITER_INTEGRITY")
    w.validator.validate(value["aggregate"],commit,hashes,md)
    return value

def load_producer(root):
    root=root.resolve()
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    require(head==COMMIT,"PRODUCER_COMMIT")
    sys.path.insert(0,str(root/"scripts"))
    import e1_master01e_v4_workflow as w
    require(w.ROOT.resolve()==root and w.KERNEL==KERNEL and w.REPORT==REPORT,"PRODUCER_IDENTITY")
    return w

def recover(w,out):
    commit,_,hashes,_,_,md,files=w.source(COMMIT)
    require(commit==COMMIT,"PRODUCER_COMMIT")
    require(w.check(KERNEL,"any")=="complete","REMOTE_NOT_COMPLETE")
    with tempfile.TemporaryDirectory() as tmp:
        p=subprocess.run(["kaggle","kernels","output",KERNEL,"-p",tmp,
            "--file-pattern","^master01ev4_private_aggregate[.]json$"],capture_output=True,timeout=180)
        require(p.returncode==0,"REMOTE_AGGREGATE_DOWNLOAD_FAILED")
        path=Path(tmp)/REPORT
        require(path.is_file() and not path.is_symlink() and 0<path.stat().st_size<100000,"REPORT_FILE_POLICY")
        raw=path.read_bytes()
        value=validate_original(raw,w,commit,hashes,files,md)
    receipt={"schema_version":1,"kind":"V4_READ_ONLY_ORIGINAL_WRITER_VERIFICATION",
             "original_submission_run_id":RUN_ID,"original_submission_commit":COMMIT,
             "kernel":KERNEL,"original_file_sha256":sha(raw),
             "original_envelope_integrity_sha256":value["integrity_sha256"],
             "model_manifest_sha256":md,
             "original_writer_schema_validated":True,"original_inner_validator_passed":True,
             "missing_outer_flags":["actual_budget_reconciled","input_budget_audit_passed"],
             "original_bytes_preserved":True,"scientific_approval":False,
             "scope":"EXPLORATORY_TRAIN_DEV_PILOT_ONLY"}
    receipt["integrity_sha256"]=w.core.digest(receipt)
    out.mkdir(parents=True,exist_ok=False)
    (out/REPORT).write_bytes(raw)
    (out/"verification_receipt.json").write_text(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("MASTER01EV4_READONLY_ORIGINAL_SHA256="+sha(raw),flush=True)
    print("MASTER01EV4_READONLY_ORIGINAL_AGGREGATE_VERIFIED",flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--producer",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    try:recover(load_producer(a.producer),a.out)
    except Exception as exc:
        codes={"PRODUCER_COMMIT","PRODUCER_IDENTITY","REMOTE_NOT_COMPLETE",
               "REMOTE_AGGREGATE_DOWNLOAD_FAILED","REPORT_FILE_POLICY",
               "DUPLICATE_JSON_KEY","NONFINITE_JSON","ORIGINAL_WRITER_SCHEMA",
               "ORIGINAL_WRITER_PINS","ORIGINAL_WRITER_INTEGRITY"}
        code=exc.args[0] if type(exc) is ValueError and len(exc.args)==1 and type(exc.args[0]) is str and exc.args[0] in codes else "STRICT_VALIDATION_FAILED"
        print("MASTER01EV4_READONLY_FAILED="+code,flush=True)
        sys.exit(1)

if __name__=="__main__":main()
