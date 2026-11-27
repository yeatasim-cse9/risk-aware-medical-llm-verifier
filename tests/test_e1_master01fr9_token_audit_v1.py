import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import e1_master01fr9_token_audit_v1 as r

def tokens(text):
    return [1]*int(text)

class Audit(unittest.TestCase):
    def run_audit(self,f=None,s=None,tokenize=tokens,authorized=True):
        return r.audit(f or [("317","2352",0),("318","2353",1)],
                       s or [("644","100",1),("1","2353",0)],
                       tokenize,authorized=authorized,expected_sizes=(2,2))
    def test_boundaries(self):
        self.assertEqual([r.windows(n,192,48) for n in (1,192,193,2352,2353)],[1,1,2,16,17])
    def test_full_denominator(self):
        x=self.run_audit()["all"]
        self.assertEqual((x["groups"],x["statement_over"],x["context_over"],
                          x["both_over"],x["either_over"],x["within"]),(4,2,2,1,3,1))
        self.assertEqual(x["statement_tokens"]["max"],644)
        self.assertEqual(x["context_windows_192_48"]["max"],17)
        self.assertEqual(x["legacy_max_context_windows"],16)
    def test_candidate_pair_counts(self):
        x=self.run_audit()["candidate_costs"][0]
        self.assertEqual(x["fit_full_pairs"]["sum"],16+34)
        self.assertEqual(x["selection_full_pairs"]["sum"],3+17)
        self.assertEqual(x["selection_permuted_pairs"]["sum"],51+1)
        self.assertEqual(x["selection_removed_pairs"]["sum"],4)
        self.assertEqual(x["total_pairs"],126)
    def test_gate_before_tokenizer(self):
        def fail(_): raise AssertionError("tokenizer called")
        with self.assertRaisesRegex(ValueError,"R9_AUDIT_AUTHORIZATION"):
            self.run_audit(tokenize=fail,authorized=False)
    def test_bad_tokens(self):
        for bad in ([],[True],[-1],(1,)):
            with self.assertRaisesRegex(ValueError,"R9_TOKEN_CONTRACT"):
                self.run_audit(tokenize=lambda _:bad)
    def test_bad_sizes(self):
        with self.assertRaisesRegex(ValueError,"R9_COHORT_SIZE"):
            r.audit([],[],tokens,authorized=True)
    def test_labels_not_used(self):
        a=self.run_audit()
        b=self.run_audit(f=[("317","2352",object()),("318","2353",object())],
                         s=[("644","100",object()),("1","2353",object())])
        self.assertEqual(a,b)
    def test_no_inference_or_public_approval(self):
        x=self.run_audit()
        self.assertFalse(x["neural_inference"])
        self.assertFalse(x["scientific_approval"])
        self.assertFalse(x["public_export_approved"])
