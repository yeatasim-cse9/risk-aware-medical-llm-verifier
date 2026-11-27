import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import e1_master01fr8_neural_v1 as r
import e1_master01fr7_lexical_v1 as l
try:
    import sklearn
    HAVE=True
except ImportError:
    HAVE=False

class Plan(unittest.TestCase):
    def test_plan_device_bound(self):
        args=("a"*40,"r6r-r8-test","b"*64,"c"*64,"d"*64,{"x":"e"*40},{"python":"3"})
        a=r.make_plan(*args,"cpu");b=r.make_plan(*args,"cuda:0")
        self.assertNotEqual(a["recipe_sha256"],b["recipe_sha256"])
        self.assertTrue(all(len(s["metrics"])<=16 for s in a["stages"]))
    def test_lexical_drift(self):
        v={"ap":.5,"auroc":.5,"brier":.2}
        r.verify_lexical(v,{"aggregates":{"evidence_ablation":v}})
        with self.assertRaises(ValueError):
            r.verify_lexical({**v,"ap":.6},{"aggregates":{"evidence_ablation":v}})
    def test_approval_before_reads(self):
        with self.assertRaisesRegex(ValueError,"R8_APPROVAL_REQUIRED"):
            r.execute(None,None,None,None,None,None,None,None,commit=None,blobs=None,
                      device="cpu",engine_factory=None)
        with self.assertRaisesRegex(ValueError,"R8_GPU_AUTHORIZATION_REQUIRED"):
            r.execute(None,None,None,None,None,None,None,None,commit=None,blobs=None,
                      device="cuda:0",engine_factory=None,authorized=True,model_rights=True,
                      label_reviewed=True,aggregate_reviewed=True)

class FakeEngine:
    special=3
    def tokens(self,s):return list(range(1,len(s.split())+1))
    def windows(self,c):return [[len(c)%13+1]],1
    def infer(self,parts,s,removed=False,batch_size=4):
        x=.5 if removed else parts[0][0]/15
        return [x,1-x,x,1-x,x,0.5],{"forward_examples":16,"padded_positions":8192,
                                                 "real_windows":0 if removed else len(parts)}

@unittest.skipUnless(HAVE,"dedicated synthetic scientific CI")
class Science(unittest.TestCase):
    def cohort(self,n,offset):
        return [("supported "+str(i) if i%2 else "unrelated "+str(i),
                 "background evidence "+"x"*(i%13+1)+str(i),i%2) for i in range(offset,offset+n)]
    def test_full_pipeline_and_ablation(self):
        fit=self.cohort(32,0);sel=self.cohort(24,100)
        v=l.experiment("evidence_ablation",fit,sel,nboot=30)
        study=r.Study(fit,sel,FakeEngine,{"aggregates":{"evidence_ablation":v}})
        prior={}
        for stage in r.STAGES:
            k=stage["name"];out=study.compute(k,prior,nboot=30)
            self.assertEqual(set(out),set(stage["metrics"]));prior[k]=out
        self.assertEqual(prior["token_budget"]["over_budget"],0)
        self.assertAlmostEqual(prior["neural_hybrid_full"]["hybrid_vs_lexical_delta"],
                               prior["neural_hybrid_full"]["hybrid_ap"]-v["ap"])
    def test_over_budget_blocks(self):
        fit=self.cohort(32,0);sel=self.cohort(24,100)
        fit[0]=("x "*400,fit[0][1],fit[0][2])
        v=l.experiment("evidence_ablation",fit,sel,nboot=30)
        study=r.Study(fit,sel,FakeEngine,{"aggregates":{"evidence_ablation":v}})
        self.assertEqual(study.budget()["over_budget"],1)
        with self.assertRaisesRegex(ValueError,"R8_COHORT_TOKEN_BUDGET_BLOCKED"):
            study.full_models()
