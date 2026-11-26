import copy,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import e1_master01d_v3_private as p
import e1_master01d_v3_validate as v
import e1_master01d_v3_workflow as w
import e1_master01d_v2_integrity as d
import e1_master01c_core as c
import e1_master01d_v2_cohort as cohort

def seal(a):
    a.pop("integrity_sha256",None);a["integrity_sha256"]=c.digest(a);return a

def fixture():
    # Invented metrics, not observed real data; historical public pins only.
    hist=json.loads((ROOT/w.CHECKPOINT).read_text())
    hashes={k:"a"*64 for k in v.MODULES};hashes.update(d.PACKAGE_HASHES)
    commit="b"*40
    s={k:{f:0 for f in v.SHORTCUTS} for k in ("train","dev")}
    for k,x in s.items():x.update(parents=d.RETAINED[k],length_tie=d.RETAINED[k])
    a={"schema_version":1,"status":"PRIVATE_BENCHMARK_AUDIT_ONLY",
       "checkpoint_sha256":d.CHECKPOINT,"mapping_sha256":d.MAPPING,
       "retained":d.RETAINED.copy(),"source_group_counts":d.RETAINED.copy(),
       "residual_flagged_pairs":0,"supplemental_exact_conflicts":0,"source_proxy_conflicts":0,
       "maxima":hist["maxima"],"versions":hist["versions"],"device":"Tesla T4",
       "shortcut_diagnostics":s,"pubmedqa_inventory":{
       "candidate_rows":{"development":500,"official_test":0},
       "excluded_rows":{"prior_protected_overlap":500,"official_test_overlap":0,
          "missing_eligibility":0,"nonrepresentative_same_group":0},
       "eligible_group_representatives":dict.fromkeys(cohort.PARTITIONS,0)},
       "hypothetical_power":cohort.prospective_power(0),
       "bioasq_status":"NOT_ACQUIRED_RELEASE_AND_RIGHTS_PENDING",
       "inventory_scope":"EXACT_PROXY_ONLY_PRIOR_01C_ORIGINALS_PROTECTED",
       "package_source_commit":commit,"package_hashes":hashes.copy(),
       **dict.fromkeys(v.FALSE_FLAGS,False)}
    return seal(a),commit,hashes

class TestMaster01DV3(unittest.TestCase):
    def test_report_strict(self):
        a,commit,hashes=fixture()
        self.assertEqual(v.validate(a,commit,hashes),a)
        for field,value,reason in [
            ("clinical_labels_approved",True,"FALSE_APPROVAL"),
            ("private_rows_exported",True,"FALSE_APPROVAL"),
            ("checkpoint_sha256","c"*64,"EVIDENCE_SCOPE"),
            ("mapping_sha256","c"*64,"EVIDENCE_SCOPE"),
            ("retained",{**a["retained"],"train":1},"RETENTION"),
            ("device","invented secret text","RUNTIME_PIN"),
            ("hypothetical_power",{},"POWER_RECOMPUTATION"),
            ("private_rows",[],"AGGREGATE_SCHEMA")]:
            b=copy.deepcopy(a);b[field]=value
            with self.assertRaisesRegex(ValueError,reason):v.validate(seal(b),commit,hashes)
        a["source_group_counts"]["train"]=1
        with self.assertRaisesRegex(ValueError,"REPORT_INTEGRITY"):v.validate(a,commit,hashes)
    def test_nested_reconciliation(self):
        a,commit,hashes=fixture()
        a["pubmedqa_inventory"]["excluded_rows"]["prior_protected_overlap"]=499
        with self.assertRaisesRegex(ValueError,"INVENTORY_RECONCILIATION"):
            v.validate(seal(a),commit,hashes)
        a,commit,hashes=fixture()
        a["shortcut_diagnostics"]["train"]["length_tie"]=0
        with self.assertRaisesRegex(ValueError,"SHORTCUT_COUNTS"):v.validate(seal(a),commit,hashes)
    def test_pins(self):
        a,commit,hashes=fixture()
        with self.assertRaisesRegex(ValueError,"SOURCE_COMMIT"):v.validate(a,"d"*40,hashes)
        hashes["e1_master01c_core"]="d"*64
        a["package_hashes"]=hashes
        with self.assertRaisesRegex(ValueError,"PACKAGE_PIN"):v.validate(seal(a),commit,hashes)
    def test_inventory_excludes_holdout_and_connected_source(self):
        raw=json.dumps({str(i):"unused" for i in range(1,501)}).encode()
        rows=[{"pubid":i,"question":"invented "+str(i),"context":{"contexts":["context "+str(i)]},
               "final_decision":"yes"} for i in range(1,1001)]
        protected=[dict(dataset="prior01c",release="master01c",id="x",task="protected_covariates",
             native_split="protected",sources=("pmid:501",),question="other question",
             context="other context",evidence_available=True,reference_available=False,provenance_known=True)]
        with patch.object(cohort,"protected_from_01c",return_value=protected):
            a=p.pubmed_inventory({}, {},raw,rows)
            self.assertEqual(a["candidate_rows"],{"development":500,"official_test":0})
            self.assertEqual(a["excluded_rows"]["prior_protected_overlap"],1)
            self.assertEqual(sum(a["eligible_group_representatives"].values()),499)
            self.assertNotIn("candidate_manifest_sha256",a)
            rows[-1]["pubid"]=999
            with self.assertRaisesRegex(ValueError,"LABELED_PMID"):
                p.pubmed_inventory({}, {},raw,rows)
    def test_run_denied_before_imports_or_download(self):
        with patch.dict(os.environ,{},clear=True),patch.object(p.p,"reconstruct_private") as call:
            with self.assertRaisesRegex(ValueError,"AUTHORIZATION_REQUIRED"):
                p.run(Path("unused"),{})
            call.assert_not_called()
    def test_preflight_never_submits(self):
        with patch.object(w,"source"),patch.object(w.subprocess,"run") as run,patch.object(w,"check") as check:
            run.return_value.returncode=0;check.side_effect=["complete","missing"]
            w.preflight()
            self.assertEqual(run.call_args.args[0],["kaggle","quota"])
            check.side_effect=["complete","running"]
            with self.assertRaisesRegex(ValueError,"DUPLICATE_KERNEL_BLOCKED"):w.preflight()
    def test_readonly_verify_requires_commit(self):
        with patch.object(w,"source") as source:
            with self.assertRaisesRegex(ValueError,"EXPECTED_SUBMISSION_COMMIT_REQUIRED"):
                w.verify(Path("unused"),"")
            source.assert_not_called()
    def test_package_is_private_and_blocked(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"package";w.build(out)
            self.assertEqual({x.name for x in out.iterdir()},{"runner.py","kernel-metadata.json"})
            meta=json.loads((out/"kernel-metadata.json").read_text())
            self.assertTrue(meta["is_private"]);self.assertTrue(meta["enable_gpu"])
            self.assertEqual(meta["id"],w.KERNEL)
            self.assertEqual(meta["dataset_sources"],[])
            p0=subprocess.run([sys.executable,str(out/"runner.py")],capture_output=True,text=True)
            self.assertNotEqual(p0.returncode,0)
            self.assertEqual(p0.stdout.strip(),"MASTER01DV3_FAILED_CODE=AUTHORIZATION_REQUIRED")
            self.assertEqual(p0.stderr,"")
    def test_authorized_build_requires_fresh_first_attempt(self):
        with patch.object(sys,"argv",["controller","build","--authorized-submit"]),patch.dict(
            os.environ,{"MASTER01DV3_RIGHTS":"true","MASTER01DV3_GPU":"true","GITHUB_RUN_ATTEMPT":"2"},
            clear=True),patch.object(w,"build") as build:
            with self.assertRaises(SystemExit):w.main()
            build.assert_not_called()
if __name__=="__main__":unittest.main()
