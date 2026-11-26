import sys
from pathlib import Path
import unittest
from math import comb
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import pirc_calibration_prototype as p

class TestPIRC(unittest.TestCase):
    def test_exact_zero_bound(self):
        self.assertAlmostEqual(p.upper_binomial(0,100,.05),1-.05**.01,places=12)
        self.assertEqual(p.upper_binomial(0,0,.05),1)
        self.assertEqual(p.upper_binomial(4,4,.05),1)
    def test_tail_inversion(self):
        for x,n in [(1,20),(5,20),(19,20),(12,100)]:
            u=p.upper_binomial(x,n,.01)
            tail=sum(comb(n,k)*u**k*(1-u)**(n-k) for k in range(x+1))
            self.assertAlmostEqual(tail,.01,places=9)
    def test_finite_sample_coverage(self):
        for n in (5,12,25):
            for prob in (.01,.05,.2,.5,.8,.95):
                covered=sum(comb(n,x)*prob**x*(1-prob)**(n-x) for x in range(n+1)
                            if p.upper_binomial(x,n,.05)>=prob)
                self.assertGreaterEqual(covered+1e-12,.95)
    def test_transition_counts(self):
        x=p.policy_summary([True,False,True,False,True],
            ["revise","revise","abstain","abstain","accept"],
            [False,True,None,None,True])
        self.assertEqual(x["introduced_errors"],1)
        self.assertEqual(x["repaired_original_errors"],1)
        self.assertEqual(x["original_correct_abstentions"],1)
        self.assertEqual(x["original_wrong_abstentions"],1)
        self.assertEqual(x["correct_released"],2)
    def test_reject_inconsistent_inputs(self):
        with self.assertRaisesRegex(ValueError,"FINAL_LABEL"):
            p.policy_summary([True],["accept"],[False])
        with self.assertRaisesRegex(ValueError,"ABSTENTION_LABEL"):
            p.policy_summary([True],["abstain"],[True])
    def test_certification_and_abstention(self):
        original=[True]*400+[False]*100
        policies={"repair":{"actions":["accept"]*400+["revise"]*100,"final":[True]*500},
                  "drop":{"actions":["abstain"]*500,"final":[None]*500},
                  "harm":{"actions":["revise"]*400+["accept"]*100,"final":[False]*500}}
        result=p.calibrate(original,policies)
        self.assertEqual(result["selected"],"repair")
        self.assertFalse(result["policies"]["drop"]["certified_under_assumptions"])
        self.assertFalse(result["policies"]["harm"]["certified_under_assumptions"])
        self.assertFalse(result["clinical_safety_certified"])
        self.assertFalse(result["real_experiment_executed"])
    def test_small_sample_and_no_original_correct(self):
        for original in ([True]*5,[False]*500):
            q={"rev":{"actions":["revise"]*len(original),"final":[True]*len(original)}}
            self.assertEqual(p.calibrate(original,q)["status"],"NO_CERTIFIED_POLICY")
    def test_multiplicity_and_tie(self):
        row={"actions":["accept"]*500,"final":[True]*500}
        a=p.calibrate([True]*500,{"z":row,"a":row})
        self.assertEqual(a["selected"],"a")
        self.assertAlmostEqual(a["per_bound_delta"],.05/6)

if __name__=="__main__": unittest.main()
