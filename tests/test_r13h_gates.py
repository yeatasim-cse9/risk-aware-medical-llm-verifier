import sys, unittest, tempfile, json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import r13h_model_gate as m
import r13h_gpu_runner as h
import r13h_private_gpu_controller as c

class Gates(unittest.TestCase):
    def metadata(self):
        return {"sha":m.REV,"id":m.REPO,"cardData":{"license":"apache-2.0"},
                "siblings":[{"rfilename":n,"size":1,**({"lfs":{"size":1,"sha256":m.SHARDS[n]}} if n in m.SHARDS else {"blobId":"a"*40})} for n in m.FILES]}
    def test_manifest(self):
        self.assertEqual(len(m.manifest(self.metadata())),len(m.FILES))
    def test_revision(self):
        x=self.metadata(); x["sha"]="bad"
        with self.assertRaises(ValueError): m.manifest(x)
    def test_shard(self):
        x=self.metadata(); x["siblings"][1]["lfs"]["sha256"]="f"*64
        with self.assertRaises(ValueError): m.manifest(x)
    def test_license(self):
        x=self.metadata(); x["cardData"]["license"]="unknown"
        with self.assertRaises(ValueError): m.manifest(x)
    def test_window_budget(self):
        g,l=h.select_windows(list(range(3000)),("drug","placebo","pressure"),lambda x:"drug pressure" if x[0]>1500 else "other")
        self.assertEqual(sum(map(len,g)),1024); self.assertEqual(sum(map(len,l)),1024)
        self.assertNotEqual(g,l)
    def test_short(self):
        g,l=h.select_windows(list(range(80)),("a","b","c"),str); self.assertEqual(g,l)
    def test_decision(self):
        a,b,d=h.LABELS
        self.assertEqual(h.decision("B",a),(b,True))
        self.assertEqual(h.decision("K",a),(a,True))
        self.assertEqual(h.decision("B because",a),(a,False))
        self.assertEqual(h.agreement((b,True),(d,True),a),(a,False))
        self.assertEqual(h.agreement((b,True),(b,True),a),(b,True))
    def test_metric(self):
        g=[{"generic":[1,2,1,0,0,0,0],"localized":[1,2,1,0,2,0,2],"agreement":[1,2,1,0,0,2,0]}]
        metrics,ci=h.aggregate(g); self.assertEqual(metrics["localized"]["correction_success"],1)
        self.assertEqual(ci["ci_95"],[1,1])
    def test_privacy(self):
        with self.assertRaises(RuntimeError): h.offline_guard("socket.connect",())
    def test_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            c.package(Path(tmp),{"dataset_source":"synthetic/private-input"},"synthetic/private-kernel")
            p=Path(tmp); meta=json.loads((p/"kernel-metadata.json").read_text())
            self.assertTrue(meta["is_private"]); self.assertTrue(meta["enable_gpu"])
            self.assertFalse(meta["enable_tpu"]); self.assertEqual(meta["dataset_sources"],["synthetic/private-input"])
            self.assertEqual(meta["model_sources"],[]); self.assertEqual(meta["competition_sources"],[])
            self.assertLessEqual(2*h.CAP_SECONDS,28800)
            compile((p/"runner.py").read_text(),"runner.py","exec")
