import copy,hashlib,json,os,sys,tempfile,unittest,subprocess
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import e1_master01e_v3_assets as a
import e1_master01e_v3_private as p
import e1_master01e_v3_validate as v
import e1_master01e_v3_workflow as w
import e1_master01d_v2_integrity as d
import e1_master01c_core as c
import e1_master01c_private as prior

def model_fixture():
    raw={k:b"{}" for k in a.SMALL}
    raw["config.json"]=json.dumps({"model_type":"deberta-v2","max_position_embeddings":512,
        "id2label":{"0":"contradiction","1":"entailment","2":"neutral"}}).encode()
    entries=[NS(rfilename=k,size=len(z),lfs=None,blob_id=a.blob(z)) for k,z in raw.items()]
    entries.extend(NS(rfilename=k,size=100,lfs=NS(sha256=h),blob_id=None) for k,h in a.LARGE.items())
    info=NS(sha=a.REVISION,private=False,card_data=NS(license="apache-2.0"),siblings=entries)
    return raw,info

def report_fixture():
    # Invented metrics, never a real pilot result.
    hashes={k:"a"*64 for k in v.MODULES};hashes.update(d.PACKAGE_HASHES)
    commit="b"*40;manifest="c"*64
    row=dict(parents=64,average_precision=.5,paired_ranking=.5,brier=.25)
    metrics={k:row.copy() for k in v.MODES}
    diff={k:dict(groups=64,replicates=1000,ap_difference=0.,percentile_95=[0.,0.])
          for k in v.MODES if k!="neural_full"}
    budget={s:{k:dict(forward_examples=n*32,padded_positions=n*32*512,active_tokens=n*32,
          real_windows=0 if k=="removed" else n*2) for k in ("full","removed","permuted")}
          for s,n in (("train",128),("dev",64))}
    x=dict(status="EXPLORATORY_TRAIN_DEV_PILOT_ONLY",train_parents=128,dev_parents=64,
        metrics=metrics,exploratory_ap_differences=diff,inference_budget=budget,
        elapsed_seconds=1.,schema_version=1,checkpoint_sha256=d.CHECKPOINT,mapping_sha256=d.MAPPING,
        model_manifest_sha256=manifest,model_revision=a.REVISION,versions=prior.OLD_VERSIONS,
        device="Tesla T4",peak_allocated_bytes=100,peak_reserved_bytes=200,
        package_source_commit=commit,package_hashes=hashes,**dict.fromkeys(v.FALSE,False))
    return seal(x),commit,hashes,manifest

def seal(x):
    x.pop("integrity_sha256",None);x["integrity_sha256"]=c.digest(x);return x

class TestMaster01EV3(unittest.TestCase):
    def test_metadata_qualification_no_large_download(self):
        raw,info=model_fixture()
        seen=[]
        def read(k):
            seen.append(k);return raw[k]
        m=a.qualify(NS(model_info=lambda *args,**kwargs:info),read)
        self.assertEqual(set(seen),set(a.SMALL))
        self.assertEqual(m["files"]["model.safetensors"],a.LARGE["model.safetensors"])
        info.siblings[-1].lfs.sha256="0"*64
        with self.assertRaisesRegex(ValueError,"PUBLISHED_LFS_PIN"):
            a.qualify(NS(model_info=lambda *args,**kwargs:info),read)
    def test_blob_and_revision_reject(self):
        raw,info=model_fixture()
        with self.assertRaisesRegex(ValueError,"GIT_BLOB_PIN"):
            a.qualify(NS(model_info=lambda *args,**kwargs:info),lambda k:b"x"*len(raw[k]))
        info.sha="0"*40
        with self.assertRaisesRegex(ValueError,"MODEL_METADATA"):
            a.qualify(NS(model_info=lambda *args,**kwargs:info),lambda k:raw[k])
    def test_no_acquisition_without_auth(self):
        with self.assertRaisesRegex(ValueError,"PRIVATE_AUTHORIZATION_REQUIRED"):
            a.acquire(Path("unused"),{},None,authorized=False)
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,"AUTHORIZATION_REQUIRED"):
                p.run(Path("unused"),{},{},None)
    def test_aggregate_and_tamper(self):
        x,commit,hashes,manifest=report_fixture()
        self.assertEqual(v.validate(x,commit,hashes,manifest),x)
        for key,value,error in [("test_used",True,"FALSE_APPROVAL"),
              ("private_rows",[],"AGGREGATE_SCHEMA"),("model_revision","x","PILOT_SCOPE"),
              ("peak_reserved_bytes",0,"RESOURCE_COUNTS")]:
            z=copy.deepcopy(x);z[key]=value
            with self.assertRaisesRegex(ValueError,error):
                v.validate(seal(z),commit,hashes,manifest)
        z=copy.deepcopy(x);z["metrics"]["length"]["average_precision"]=.9
        with self.assertRaisesRegex(ValueError,"AP_DIFFERENCE"):
            v.validate(seal(z),commit,hashes,manifest)
        z=copy.deepcopy(x);z["inference_budget"]["train"]["full"]["forward_examples"]=1
        with self.assertRaisesRegex(ValueError,"BUDGET_COUNTS"):
            v.validate(seal(z),commit,hashes,manifest)
    def test_code_only_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"code";w.build(out)
            meta=json.loads((out/"kernel-metadata.json").read_text())
            self.assertTrue(meta["is_private"]);self.assertTrue(meta["enable_gpu"])
            self.assertEqual(meta["id"],w.KERNEL);self.assertEqual(meta["dataset_sources"],[])
            proc=subprocess.run([sys.executable,str(out/"runner.py")],capture_output=True,text=True)
            self.assertNotEqual(proc.returncode,0)
            self.assertEqual(proc.stdout.strip(),"MASTER01EV3_FAILED_CODE=AUTHORIZATION_REQUIRED")
            self.assertEqual(proc.stderr,"")
    def test_manual_authorization(self):
        with patch.object(sys,"argv",["controller","build","--authorized-submit"]),patch.dict(
            os.environ,{"MASTER01EV3_RIGHTS":"true","MASTER01EV3_GPU":"true",
            "GITHUB_RUN_ATTEMPT":"1"},clear=True),patch.object(w,"build") as build:
            with self.assertRaises(SystemExit):w.main()
            build.assert_not_called()
        with patch.object(sys,"argv",["controller","build","--authorized-submit"]),patch.dict(
            os.environ,{"MASTER01EV3_RIGHTS":"true","MASTER01EV3_GPU":"true",
            "MASTER01EV3_MODEL_RIGHTS":"true","GITHUB_RUN_ATTEMPT":"2"},clear=True),patch.object(w,"build") as build:
            with self.assertRaises(SystemExit):w.main()
            build.assert_not_called()
    def test_preflight_duplicate(self):
        with patch.object(w,"source"),patch.object(w.subprocess,"run") as run,patch.object(w,"check") as check:
            run.return_value.returncode=0;check.side_effect=["complete","missing"];w.preflight()
            check.side_effect=["complete","running"]
            with self.assertRaisesRegex(ValueError,"DUPLICATE_KERNEL_BLOCKED"):w.preflight()

if __name__=="__main__":unittest.main()
