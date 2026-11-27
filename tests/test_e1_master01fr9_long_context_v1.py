import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import e1_master01fr9_long_context_v1 as r
import e1_master01fr9_token_audit_v1 as audit
import e1_master01fr7_lexical_v1 as l

class Fake:
    special=3
    def tokens(self,s):return [1]*len(s.split())
    def encode(self,c,h):
        if len(c)+len(h)+3>512:raise AssertionError("too long")
        return {"c":len(c),"h":len(h)}
def forward(engine,batch):
    if len(batch)>4:raise AssertionError("too many")
    return [{"entailment":.6 if x["c"] else .2,
             "contradiction":.1 if x["c"] else .5,"neutral":.3} for x in batch]

class Contract(unittest.TestCase):
    def test_all_token_positions(self):
        for n,w,o in ((5888,192,48),(644,317,64),(317,317,64),(318,317,64)):
            ids=list(range(n));parts=r.tiles(ids,w,o)
            self.assertEqual(set(i for p in parts for i in p),set(ids))
            self.assertTrue(all(len(p)<=w for p in parts))
    def test_full_long_case(self):
        c=r.tiles(list(range(5888)),192,48)
        f,count=r.infer_case(Fake(),"x "*644,c,forward=forward)
        self.assertEqual(count,123)
        self.assertAlmostEqual(f[4],.6)
        _,removed=r.infer_case(Fake(),"x "*644,c,removed=True,forward=forward)
        self.assertEqual(removed,3)
    def test_partial_batch_and_wrong_count(self):
        with self.assertRaisesRegex(ValueError,"R9_FORWARD_COUNT"):
            r.infer_case(Fake(),"x",[[1]],forward=lambda *_:[])
    def test_pool_no_hidden_contradiction(self):
        a={"entailment":.9,"contradiction":.05,"neutral":.05}
        b={"entailment":.1,"contradiction":.8,"neutral":.1}
        x=r.pool([[a,b],[b]])
        self.assertAlmostEqual(x[0],.9)
        self.assertAlmostEqual(x[1],.8)
        self.assertAlmostEqual(x[4],.1)
        self.assertAlmostEqual(x[5],.5)
    def test_reject_bad_probability(self):
        with self.assertRaisesRegex(ValueError,"R9_POSTERIOR"):
            r.pool([[{"entailment":float("nan"),"contradiction":.2,"neutral":.3}]])
    def test_approval_gate(self):
        with self.assertRaisesRegex(ValueError,"R9_PROTOCOL_APPROVAL_REQUIRED"):
            r.Study(None,None,None,None,None)
    def test_plan_bound_to_audit(self):
        args=dict(blobs={"a":"a"*40},runtime={"python":"3"},receipt_pin="a"*64,
                  r6_pin="b"*64,r7_pin="c"*64,audit_pin="d"*64,
                  v5_plan_pin="e"*64,v5_lexical_pin="f"*64,v5_budget_pin="0"*64)
        a=r.make_plan("a"*40,"r6r-r9-test",**args)
        args["audit_pin"]="1"*64
        b=r.make_plan("a"*40,"r6r-r9-test",**args)
        self.assertNotEqual(a["input_sha256"],b["input_sha256"])

try:
    import sklearn
    HAVE=True
except ImportError:
    HAVE=False
@unittest.skipUnless(HAVE,"dedicated scientific synthetic CI")
class Pipeline(unittest.TestCase):
    def test_same_cohort_all_stages(self):
        def rows(n,offset):
            return [("supported "+str(i)+" x "*(320 if i%3==0 else 1),
                     "background "+str(i)+" y "*(2400 if i%5==0 else 5),i%2)
                    for i in range(offset,offset+n)]
        f,s=rows(256,0),rows(128,256)
        expected=audit.audit(f,s,Fake().tokens,authorized=True)
        ref={"aggregates":{"evidence_ablation":{"ap":.5,"auroc":.5,"brier":.25}}}
        study=r.Study(f,s,Fake,ref,expected,authorized=True,forward=forward)
        prior={}
        for stage in r.STAGES:
            name=stage["name"];out=study.compute(name,prior,nboot=30)
            self.assertEqual(set(out),set(stage["metrics"]));prior[name]=out
        self.assertEqual(prior["token_budget"]["over_budget"],0)
        self.assertEqual(sum(study.actual_pairs),expected["candidate_costs"][0]["total_pairs"])
        self.assertAlmostEqual(prior["neural_hybrid_full"]["hybrid_vs_lexical_delta"],
            prior["neural_hybrid_full"]["hybrid_ap"]-prior["lexical_same_run"]["ap"])
