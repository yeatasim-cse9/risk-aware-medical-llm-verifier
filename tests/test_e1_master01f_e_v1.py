"""Synthetic nonmedical custody tests, no private data or GPU."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import e1_master01f_e_provision_v1 as e
import e1_master01f_e_package_v1 as pkg
import e1_master01f_d_inputs_v1 as d
class Provision(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.cov=self.root/"cov";self.cov.mkdir()
        self.ans=self.root/"ans";self.ans.mkdir();self.out=self.root/"out"
        self.records={};mapping=[]
        sizes={"train":4,"dev":3,"test":1,"official_holdout":1}
        for j,(s,n) in enumerate(sizes.items()):
            self.records[s]=[]
            for k in range(n):
                config="official_holdout" if s=="official_holdout" else "pqa_artificial"
                key=[config,j*100+k]
                row={"key":key,"config":config,"source":str(j*100+k),
                     "question":"invented question "+str(j*100+k),
                     "context":"blue triangle "+str(j*100+k),"category":"invented","difficulty":"easy"}
                self.records[s].append(row)
                if s!="official_holdout":mapping.append([*key,s])
        self.mapping=sorted(mapping)
        for obj,k,v in [(e.hist,"ORIGINAL",sizes),(e.hist,"RETAINED",sizes),
                        (e.hist,"MAPPING",e.c.digest({"seed":e.c.SEED,"assignment":self.mapping})),
                        (d,"COUNTS",{"fit":4,"selection":3}),(d,"LIMITS",{"fit":3,"selection":2})]:
            z=patch.object(obj,k,v);z.start();self.addCleanup(z.stop)
        self.pin=self.write_cov()
        req=e.request(self.cov,self.pin);files={}
        for row in req["selected"]:
            p=row["parent"];name=p+".json"
            a={"parent":p,"key":row["key"],"partition":d.ORIGINAL[row["partition"]],
               "answers":{"ground_truth":"blue triangle","hallucinated":"red square"}}
            raw=e.encode(a);(self.ans/name).write_bytes(raw);files[name]=d.sha(raw)
        self.index={"schema":1,"snapshot_sha256":self.pin,"request_sha256":d.sha(e.encode(req)),"files":files}
        self.ipin=self.write_index()
    def write_cov(self):
        raw=e.encode({"schema":1,"records":self.records,"mapping":self.mapping})
        (self.cov/"covariates.json").write_bytes(raw);return d.sha(raw)
    def write_index(self):
        raw=e.encode(self.index);(self.ans/"index.json").write_bytes(raw);return d.sha(raw)
    def provision(self,authorized=True):
        return e.provision(self.cov,self.pin,self.ans,self.ipin,self.out,authorized=authorized)
    def test_roundtrip_and_scope(self):
        req=e.request(self.cov,self.pin)
        self.assertEqual(len(req["selected"]),5)
        self.assertEqual({x["partition"] for x in req["selected"]},{"fit","selection"})
        pin=self.provision()
        blocks=d.load_selected(self.out,pin,authorized=True)
        self.assertEqual({k:len(v) for k,v in blocks.items()},d.LIMITS)
        with self.assertRaises(ValueError):self.provision()
    def test_unauthorized_no_reads(self):
        with patch.object(d,"read_pinned") as read:
            with self.assertRaises(ValueError):self.provision(False)
            read.assert_not_called()
    def test_extra_file_rejected_before_answer_reads(self):
        (self.ans/"test.json").write_text("not read")
        with patch.object(d,"read_pinned",wraps=d.read_pinned) as read:
            with self.assertRaisesRegex(ValueError,"ANSWER_MOUNT_SCOPE"):self.provision()
            self.assertTrue(all(x.args[1] in ("covariates.json","index.json") for x in read.call_args_list))
        self.assertFalse(self.out.exists())
    def test_mapping_and_extra_field_fail(self):
        self.mapping[0][2]="test";self.pin=self.write_cov()
        with self.assertRaisesRegex(ValueError,"MAPPING_PIN"):e.request(self.cov,self.pin)
        self.records["test"][0]["unexpected"]="invented"
        self.pin=self.write_cov()
        with self.assertRaisesRegex(ValueError,"COVARIATE_FIELDS"):e.request(self.cov,self.pin)
    def test_tampered_answer_and_index(self):
        name=next(iter(self.index["files"]))
        (self.ans/name).write_text("changed")
        with self.assertRaisesRegex(ValueError,"FILE_INTEGRITY"):self.provision()
        self.assertFalse(self.out.exists())
        self.index["request_sha256"]="0"*64;self.ipin=self.write_index()
        with self.assertRaisesRegex(ValueError,"INDEX_BINDING"):self.provision()
    def test_overlap_from_excluded_covariates(self):
        self.records["dev"][0]["source"]=self.records["train"][0]["source"]
        self.pin=self.write_cov()
        with self.assertRaisesRegex(ValueError,"SOURCE_OVERLAP"):e.request(self.cov,self.pin)
class Package(unittest.TestCase):
    def test_external_manifest_required(self):
        with patch.object(pkg,"hashes",return_value={"a":"b"}):
            with self.assertRaises(ValueError):pkg.check_pilot(Path("."),{"a":"c"})
            self.assertEqual(pkg.check_pilot(Path("."),{"a":"b"}),{"a":"b"})
    def test_child_readonly_dispatch(self):
        from types import SimpleNamespace
        with patch.object(pkg,"check_pilot"),patch.object(pkg.subprocess,"run",
                return_value=SimpleNamespace(returncode=0,stdout=b"MASTER01FD_READONLY_PREFLIGHT_PASSED_METADATA_ONLY\n")) as run:
            self.assertEqual(pkg.preflight(Path("."),{"a":"b"},Path("."),Path("."),"a"*64),
                             "MASTER01FE_READONLY_PREFLIGHT_PASSED_METADATA_ONLY")
            args,kw=run.call_args
            self.assertIn("preflight",args[0]);self.assertNotIn("run",args[0])
            self.assertEqual(kw["env"]["CUDA_VISIBLE_DEVICES"],"")
            self.assertNotIn("MASTER01FD_AUTHORIZED_ONE_GPU_PILOT",kw["env"])
if __name__=="__main__":unittest.main()
