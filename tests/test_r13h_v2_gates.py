import sys, unittest, tempfile, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import r13h_v2_gpu_runner as h
import r13h_v2_private_gpu_controller as c
import r13h_gpu_runner as v1

class V2Gates(unittest.TestCase):
    def test_pairs_and_waves(self):
        jobs=list(range(3570)); counts=[0,0]; result=[]
        for i in range(0,len(jobs),h.BATCH):
            chunks=h.assignments(jobs[i:i+h.BATCH])
            for j,chunk in enumerate(chunks):
                counts[j]+=len(chunk)*2; result.extend(chunk)
        self.assertEqual(result,jobs)
        self.assertEqual(sum(counts),7140)
        self.assertLessEqual(abs(counts[0]-counts[1]),4)
        self.assertEqual(h.waves_for(len(jobs)),893)
    def test_parallel_dispatch(self):
        class Tok:
            def decode(self,x,skip_special_tokens): return str(x)
        def fake(model,tok,torch,inputs,smoke,device):
            self.assertEqual(model,device)
            return [[device,j] for j in inputs]
        with patch.object(h,"generate",fake), ThreadPoolExecutor(max_workers=2) as p:
            out=h.parallel_wave(p,[0,1],Tok(),None,[[1,2],[3,4]])
            self.assertEqual(out,[["[0, 1]","[0, 2]"],["[1, 3]","[1, 4]"]])
            self.assertEqual(h.parallel_wave(p,[0,1],Tok(),None,[[1,2],[]]),[["[0, 1]","[0, 2]"],[]])
    def test_unchanged_scientific_functions(self):
        import inspect
        for n in ("prepare_cohort","select_windows","messages","decision","agreement","outcome","aggregate","pack_ids","offline_guard"):
            self.assertEqual(inspect.getsource(getattr(h,n)),inspect.getsource(getattr(v1,n)))
    def test_status(self):
        for s in ("cancelled","cancelacknowledged","canceled"): self.assertEqual(c.classify(s),"cancelled")
        for s in ("queued","pending","starting","initializing"): self.assertEqual(c.classify(s),"queue")
        self.assertEqual(c.classify("running"),"runtime")
        self.assertEqual(c.classify("complete"),"complete")
        self.assertEqual(c.classify("missing"),"other")
    def test_cap(self):
        self.assertLessEqual(2*h.CAP_SECONDS,28800)
        self.assertLess(h.SOFT_SECONDS,h.CAP_SECONDS)
        self.assertLess(c.CONTROLLER_SECONDS,355*60)
    def test_private_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); c.package(p,{"dataset_source":"synthetic/private"},"synthetic/v2")
            m=json.loads((p/"kernel-metadata.json").read_text())
            self.assertTrue(m["is_private"] and m["enable_gpu"])
            self.assertFalse(m["enable_tpu"])
            self.assertEqual(m["dataset_sources"],["synthetic/private"])
            self.assertEqual(m["model_sources"],[])
            self.assertEqual(m["competition_sources"],[])
            compile((p/"runner.py").read_text(),"runner.py","exec")
