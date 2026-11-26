"""Audit every sampled encoding before private inference; invented fixtures only."""
import ast
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
S=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(S))
import e1_master01e_v4_budget as a
import e1_master01e_v4_neural as n
from test_e1_master01e_v4 import ModernTokenizer,Model

def rows(split,start):
    return [{"parent":str(i),"source_group":str(i),"partition":split,
             "question":"invented question","context":"alpha beta "* (i+1),
             "answers":{"ground_truth":"alpha beta","hallucinated":"wrong"}}
            for i in range(start,start+2)]

class Audit(unittest.TestCase):
    def engine(self):
        m=Model()
        m.config.vocab_size=32
        e=n.Engine(ModernTokenizer(),m,"cpu")
        e.infer=lambda *args,**kwargs:self.fail("audit must not infer")
        return e

    def audit(self,e,tr,dv,mark=lambda x:None):
        with patch.object(a.p,"LIMITS",{"train":2,"dev":2}),patch.object(a.p,"validate_private_inputs",side_effect=a.b.disjoint):
            return a.audit_both(e,tr,dv,mark)

    def test_both_partitions_exact_budget(self):
        e=self.engine();tr=rows("train",0);dv=rows("dev",2);marks=[]
        v=self.audit(e,tr,dv,marks.append)
        self.assertEqual(marks,["AUDIT_TRAIN","AUDIT_DEV"])
        for split in ("train","dev"):
            for k,c in v[split].items():
                self.assertEqual(c["forward_examples"],64)
                self.assertEqual(c["padded_positions"],32768)
                self.assertEqual(c["real_windows"],0 if k=="removed" else 4)
            self.assertLess(v[split]["removed"]["active_tokens"],v[split]["full"]["active_tokens"])
            self.assertEqual(v[split]["full"]["active_tokens"],v[split]["permuted"]["active_tokens"])

    def test_dev_overlong_answer_blocks_before_any_inference(self):
        e=self.engine();dv=rows("dev",2)
        dv[0]["answers"]["hallucinated"]="alpha "*318
        with self.assertRaisesRegex(ValueError,"HYPOTHESIS_BUDGET_EXCEEDED"):
            self.audit(e,rows("train",0),dv)

    def test_context_overflow_and_corrupted_mask(self):
        e=self.engine();tr=rows("train",0)
        tr[0]["context"]="alpha "*2500
        with self.assertRaisesRegex(ValueError,"CONTEXT_BUDGET_EXCEEDED"):
            self.audit(e,tr,rows("dev",2))
        e=self.engine();original=e.encode
        def bad(p,h):
            v=original(p,h);v["attention_mask"][0]=0;return v
        e.encode=bad
        with self.assertRaisesRegex(ValueError,"AUDIT_ATTENTION_MASK"):
            self.audit(e,rows("train",0),rows("dev",2))

    def test_token_range_and_window_coverage(self):
        e=self.engine();e.model.config.vocab_size=3
        with self.assertRaisesRegex(ValueError,"AUDIT_TOKEN_RANGE"):
            self.audit(e,rows("train",0),rows("dev",2))
        e=self.engine();original=e.windows
        def bad(text):
            p,t=original(text);return p,t+1
        e.windows=bad
        with self.assertRaisesRegex(ValueError,"AUDIT_WINDOW_COVERAGE"):
            self.audit(e,rows("train",0),rows("dev",2))

    def test_private_runner_order_and_qualified_engine(self):
        src=(S/"e1_master01e_v4_private.py").read_text()
        tree=ast.parse(src)
        imports={x.name for x in ast.walk(tree) if isinstance(x,ast.alias)}
        self.assertIn("e1_master01e_v4_neural",imports)
        self.assertNotIn("e1_master01e_v2_neural",imports)
        self.assertLess(src.index("budget.audit_both("),src.index("pilot.run("))
        self.assertLess(src.index("GPU_SMOKE_REPEAT"),src.index("HISTORICAL_RECONSTRUCTION"))
        self.assertIn('result["inference_budget"]==expected',src)

if __name__=="__main__":unittest.main()
