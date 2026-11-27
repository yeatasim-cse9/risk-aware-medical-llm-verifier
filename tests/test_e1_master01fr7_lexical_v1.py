import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import e1_master01fr7_lexical_v1 as r
import e1_master01f_r3_audit_v1 as r3
try:
    import sklearn
    import numpy
    HAVE=True
except ImportError:
    HAVE=False

def rows():
    out=[]
    for i in range(80):
        y=i%2==0
        # Unique text per group; deterministic simple synthetic data.
        statement=("consistent supported " if y else "fabricated unrelated ")+str(i)
        out.append(dict(id=str(i),inner_id=str(i),source="sumpubmed",
                        context="consistent supported background "+str(i),
                        statement=statement,label=y,synthetic=False))
    return out

class Protocol(unittest.TestCase):
    def test_reconstruction_matches_r3(self):
        data=rows()
        fit,sel,summary=r.reconstruct(lambda:iter(data),32,24)
        old=r3.audit(iter(data),32,24)
        self.assertTrue(all(old[k]==v for k,v in summary.items()))
        self.assertEqual(len(fit),32); self.assertEqual(len(sel),24)
        self.assertFalse(set(x[0] for x in fit)&set(x[0] for x in sel))
    def test_reconstruction_order(self):
        data=rows()
        a=r.reconstruct(lambda:iter(data),32,24)
        b=r.reconstruct(lambda:iter(reversed(data)),32,24)
        self.assertEqual(a,b)
    def test_no_target_explanation(self):
        data=rows(); data[0]["explanation"]="not allowed"
        with self.assertRaises(ValueError):r.reconstruct(lambda:iter(data),32,24)
    def test_cohort_check(self):
        with self.assertRaises(ValueError):r.check_r6({"fit_groups":256,"selection_groups":128},{"fit_groups":255})
    def test_plan_version_and_inputs_bound(self):
        a=r.make_plan("a"*40,"r6r-r7-test","b"*64,"c"*64,{"code":"d"*40},{"python":"3.11"})
        b=r.make_plan("a"*40,"r6r-r7-test","b"*64,"c"*64,{"code":"d"*40},{"python":"3.12"})
        self.assertNotEqual(a["recipe_sha256"],b["recipe_sha256"])
        self.assertEqual(len(a["stages"]),5)
    def test_permutation(self):
        self.assertEqual(r.permute(["a","b","c"]),["b","c","a"])
        with self.assertRaises(ValueError):r.permute(["a"])

@unittest.skipUnless(HAVE,"dedicated lexical CI installs scientific dependencies")
class Science(unittest.TestCase):
    def test_all_real_estimators(self):
        data=rows(); fit,sel,_=r.reconstruct(lambda:iter(data),32,24)
        out={}
        for stage in r.STAGES:
            v=r.experiment(stage["name"],fit,sel,nboot=100)
            self.assertEqual(set(v),set(stage["metrics"]))
            self.assertTrue(all(-1<=x<=1 for x in v.values()))
            out[stage["name"]]=v
        full=out["evidence_ablation"]
        self.assertAlmostEqual(full["statement_ap"],out["statement_combined"]["ap"])
        self.assertAlmostEqual(full["vs_statement_delta"],full["ap"]-full["statement_ap"])
    def test_paired_identical(self):
        y=[0,1]*16; s=[.1,.9]*16
        self.assertEqual(r.paired(y,s,{"same":s},100),{"same_delta":0.,"same_lo":0.,"same_hi":0.})
    def test_vocabulary_fit_only(self):
        a,b,vs=r.block(["fit alpha","fit beta"],["unseenword"],["word"])
        self.assertNotIn("unseenword",vs[0].vocabulary_)
        self.assertEqual(b.nnz,0)
    def test_removed_overlap(self):
        self.assertEqual(r.numeric(["alpha"],[""]).tolist(),[[5.,0.,0.]])
