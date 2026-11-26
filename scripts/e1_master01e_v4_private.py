"""V4 private exploratory pilot: pinned engine, full pre-inference audit."""
import importlib.metadata,os,random
from pathlib import Path
import e1_master01c_core as c
import e1_master01c_private as p
import e1_master01d_v2_integrity as d
import e1_master01d_v3_private as prior
import e1_master01e_v3_assets as assets
import e1_master01e_v4_neural as neural
import e1_master01e_v4_budget as budget
import e1_master01e_v2_pilot as pilot
require=c.require

def run(root,report,manifest,manifest_digest,recorder):
    require(os.environ.get("MASTER01EV4_AUTHORIZED_ONE_PILOT")=="YES","V4_AUTHORIZATION_REQUIRED")
    recorder.mark("AUTHORIZATION")
    require(os.environ.get("MASTER01EV3_AUTHORIZED_ONE_PILOT")=="YES","AUTHORIZATION_REQUIRED")
    require(isinstance(root,Path) and root.is_dir() and not root.is_symlink()
            and not any(root.iterdir()),"PRIVATE_SCRATCH")
    recorder.mark("CHECKPOINT")
    d.verify_checkpoint(report);d.verify_prior_source_files()
    assets.validate_manifest(manifest)
    require(assets.digest(manifest)==manifest_digest,"MANIFEST_PIN")
    os.environ["MASTER01C_PRIVATE_RIGHTS_CONFIRMED"]="YES"
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"]="1"
    os.environ["HF_HOME"]=str(root/"hf")
    recorder.mark("DEPENDENCIES")
    import torch
    import pyarrow.parquet as pq
    recorder.mark("RUNTIME_PIN")
    versions={k:importlib.metadata.version(k) for k in p.OLD_VERSIONS}
    require(versions==p.OLD_VERSIONS,"RUNTIME_VERSION_DRIFT")
    recorder.mark("GPU_SETUP")
    require(torch.cuda.is_available(),"GPU_REQUIRED")
    require(torch.cuda.device_count()==2 and all(torch.cuda.get_device_name(i)=="Tesla T4"
            for i in range(2)),"T4_X2_REQUIRED")
    random.seed(c.SEED);torch.manual_seed(c.SEED);torch.cuda.manual_seed_all(c.SEED)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    path=root/"nli"
    recorder.mark("MODEL_ASSET_ACQUIRE")
    assets.acquire(path,manifest,manifest_digest,authorized=True)
    recorder.mark("MODEL_LOAD_SMOKE")
    engine=neural.load_offline(path,manifest,manifest_digest,authorized=True)
    recorder.mark("GPU_SMOKE_WINDOWS")
    windows,total=engine.windows("The invented object is blue. "*80)
    other,_=engine.windows("An invented triangle is green. "*80)
    require(total>neural.WINDOW and len(windows)>1,"MULTIWINDOW_SMOKE")
    try:engine.encode([], [engine.sep]*(neural.MAX_LEN-neural.WINDOW-engine.special+1))
    except ValueError as exc:require(exc.args==("HYPOTHESIS_BUDGET_EXCEEDED",),"BUDGET_SMOKE")
    else:raise ValueError("BUDGET_SMOKE")
    recorder.mark("GPU_SMOKE_FULL")
    full,c1=engine.infer(windows,"The object is blue.")
    recorder.mark("GPU_SMOKE_REMOVED")
    removed,c2=engine.infer(windows,"The object is blue.",removed=True)
    recorder.mark("GPU_SMOKE_PERMUTED")
    perm,c3=engine.infer(other,"The object is blue.")
    recorder.mark("GPU_SMOKE_REPEAT")
    repeat,c4=engine.infer(windows,"The object is blue.")
    for f,cost in ((full,c1),(removed,c2),(perm,c3),(repeat,c4)):
        require(len(f)==6 and cost["forward_examples"]==16 and cost["padded_positions"]==8192,"NLI_SMOKE")
    require((full,c1)==(repeat,c4),"DETERMINISTIC_REPEAT_MISMATCH")
    require(c2["real_windows"]==0 and c2["active_tokens"]<c1["active_tokens"]
            and c1["real_windows"]==len(windows) and c3["real_windows"]==len(other),"REMOVAL_SMOKE")
    torch.cuda.synchronize()
    del engine;torch.cuda.empty_cache()
    # Historical inference repeated using unchanged, already verified routines.
    recorder.mark("HISTORICAL_RECONSTRUCTION")
    records,norm,diagnostics,exact,unused=p.reconstruct_private(root,return_private=False)
    require(unused=={},"PRIVATE_ANSWER_SCOPE")
    recorder.mark("HISTORICAL_SCREEN")
    flags,maxima=p.screen(records,norm,diagnostics,exact,root,torch.device("cuda:0"))
    recorder.mark("MAPPING_HOLDOUT")
    kept,mapping,agg=d.reconstruct_candidate(records,flags,norm)
    require(agg["quarantine"]==report["quarantine"] and agg["balance"]==report["balance"]
            and c.digest({"seed":c.SEED,"assignment":mapping})==d.MAPPING,"CHECKPOINT_DRIFT")
    helper=prior.helper_at(root);d.verify_holdout_identity(records,prior.holdout_bytes(helper))
    recorder.mark("TRAIN_DEV_ANSWERS")
    answers={}
    for cfg in sorted(d.CONFIGS):
        rows=prior.checked_local(root/"med",helper.SOURCE_SHA256[cfg],pq,
                           ["Ground Truth","Hallucinated Answer"])
        for split in ("train","dev"):
            for row in kept[split]:
                if row["config"]==cfg:
                    a=rows[row["key"][1]]
                    answers[row["key"]]=(a["Ground Truth"],a["Hallucinated Answer"])
        del rows
    recorder.mark("PILOT_INPUTS")
    blocks=pilot.prepare_verified_inputs(report,records,flags,norm,answers,authorized=True)
    del answers,records,flags,kept
    torch.cuda.empty_cache()
    recorder.mark("MODEL_RELOAD")
    engine=neural.load_offline(path,manifest,manifest_digest,authorized=True)
    torch.cuda.reset_peak_memory_stats()
    expected=budget.audit_both(engine,blocks["train"],blocks["dev"],recorder.mark)
    recorder.mark("PILOT_EXECUTION")
    result=pilot.run(engine,blocks["train"],blocks["dev"],authorized=True)
    require(result["inference_budget"]==expected,"AUDIT_ACTUAL_BUDGET_MISMATCH")
    # Do not export small-cell strata or row IDs; even aggregate counts omitted.
    recorder.mark("AGGREGATE_PREPARE")
    result.pop("length_strata")
    result.update(schema_version=1,checkpoint_sha256=d.CHECKPOINT,mapping_sha256=d.MAPPING,
          model_manifest_sha256=manifest_digest,model_revision=assets.REVISION,
          versions=versions,device=torch.cuda.get_device_name(0),
          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
          peak_reserved_bytes=torch.cuda.max_memory_reserved(),
          subgroup_exported=False,private_rows_exported=False,
          scientific_experiment_executed=False)
    return result
