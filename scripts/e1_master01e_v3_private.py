"""Private pilot entrypoint. No imports of torch or acquisition until gated."""
import importlib.metadata,os,random
from pathlib import Path
import e1_master01c_core as c
import e1_master01c_private as p
import e1_master01d_v2_integrity as d
import e1_master01d_v3_private as prior
import e1_master01e_v3_assets as assets
import e1_master01e_v2_neural as neural
import e1_master01e_v2_pilot as pilot
require=c.require

def run(root,report,manifest,manifest_digest):
    require(os.environ.get("MASTER01EV3_AUTHORIZED_ONE_PILOT")=="YES","AUTHORIZATION_REQUIRED")
    require(isinstance(root,Path) and root.is_dir() and not root.is_symlink()
            and not any(root.iterdir()),"PRIVATE_SCRATCH")
    d.verify_checkpoint(report);d.verify_prior_source_files()
    assets.validate_manifest(manifest)
    require(assets.digest(manifest)==manifest_digest,"MANIFEST_PIN")
    os.environ["MASTER01C_PRIVATE_RIGHTS_CONFIRMED"]="YES"
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
    os.environ["HF_HOME"]=str(root/"hf")
    import torch
    import pyarrow.parquet as pq
    versions={k:importlib.metadata.version(k) for k in p.OLD_VERSIONS}
    require(versions==p.OLD_VERSIONS,"RUNTIME_VERSION_DRIFT")
    require(torch.cuda.is_available(),"GPU_REQUIRED")
    random.seed(c.SEED);torch.manual_seed(c.SEED);torch.cuda.manual_seed_all(c.SEED)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    path=root/"nli"
    assets.acquire(path,manifest,manifest_digest,authorized=True)
    engine=neural.load_offline(path,manifest,manifest_digest,authorized=True)
    windows,n=engine.windows("The invented object is blue. "*80)
    for removed in (False,True):
        f,cost=engine.infer(windows,"The object is blue.",removed=removed)
        require(len(f)==6 and cost["forward_examples"]==16,"NLI_SMOKE")
    del engine;torch.cuda.empty_cache()
    # Historical inference repeated using unchanged, already verified routines.
    records,norm,diagnostics,exact,unused=p.reconstruct_private(root,return_private=False)
    require(unused=={},"PRIVATE_ANSWER_SCOPE")
    flags,maxima=p.screen(records,norm,diagnostics,exact,root,torch.device("cuda:0"))
    kept,mapping,agg=d.reconstruct_candidate(records,flags,norm)
    require(agg["quarantine"]==report["quarantine"] and agg["balance"]==report["balance"]
            and c.digest({"seed":c.SEED,"assignment":mapping})==d.MAPPING,"CHECKPOINT_DRIFT")
    helper=prior.helper_at(root);d.verify_holdout_identity(records,prior.holdout_bytes(helper))
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
    blocks=pilot.prepare_verified_inputs(report,records,flags,norm,answers,authorized=True)
    del answers,records,flags,kept
    torch.cuda.empty_cache()
    engine=neural.load_offline(path,manifest,manifest_digest,authorized=True)
    torch.cuda.reset_peak_memory_stats()
    result=pilot.run(engine,blocks["train"],blocks["dev"],authorized=True)
    # Do not export small-cell strata or row IDs; even aggregate counts omitted.
    result.pop("length_strata")
    result.update(schema_version=1,checkpoint_sha256=d.CHECKPOINT,mapping_sha256=d.MAPPING,
          model_manifest_sha256=manifest_digest,model_revision=assets.REVISION,
          versions=versions,device=torch.cuda.get_device_name(0),
          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
          peak_reserved_bytes=torch.cuda.max_memory_reserved(),
          subgroup_exported=False,private_rows_exported=False,
          scientific_experiment_executed=False)
    return result
