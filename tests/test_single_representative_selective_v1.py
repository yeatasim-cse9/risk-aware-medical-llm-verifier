import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from single_representative_selective_v1 import evaluate_fixed_family

class TestSingleRepresentative(unittest.TestCase):
    def test_pass_and_tie(self):
        rows=[(str(i),False,.05) for i in range(400)]
        r=evaluate_fixed_family(rows,(.05,.1,.2))
        self.assertEqual(r["selected_threshold"],.05)
        self.assertAlmostEqual(r["per_bound_delta"],.05/6)
        self.assertFalse(r["formal_risk_guarantee"])
    def test_negative_and_empty(self):
        rows=[(str(i),True,.2) for i in range(100)]
        r=evaluate_fixed_family(rows,(.05,.2))
        self.assertIsNone(r["selected_threshold"])
        self.assertIsNone(r["policies"][0]["risk"])
        self.assertEqual(r["policies"][0]["risk_upper"],1.)
        self.assertEqual(r["status"],"NO_POLICY_MEETS_BOUNDS")
    def test_monotone_counts_and_inclusive(self):
        r=evaluate_fixed_family([("a",False,.1),("b",True,.2),("c",False,.3)],(.1,.2,.3))
        self.assertEqual([x["accepted"] for x in r["policies"]],[1,2,3])
        self.assertEqual([x["accepted_nonfactual"] for x in r["policies"]],[0,1,1])
    def test_coverage_priority(self):
        rows=[(str(i),False,.1 if i<400 else .2) for i in range(500)]
        self.assertEqual(evaluate_fixed_family(rows,(.1,.2))["selected_threshold"],.2)
    def test_invalid_inputs(self):
        for rows in ([],[("a",False,.1),("a",True,.2)],[("a",0,.1)],
                     [("a",False,float("nan"))],[("a",False,2)],[("a",False,True)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                evaluate_fixed_family(rows,(.1,.2))
        for thresholds in ((),(.2,.1),(.1,.1),(float("inf"),),[.1]):
            with self.subTest(thresholds=thresholds), self.assertRaises(ValueError):
                evaluate_fixed_family([("a",False,.1)],thresholds)
    def test_permutation_invariant(self):
        rows=[("a",True,.2),("b",False,.1),("c",False,.3)]
        self.assertEqual(evaluate_fixed_family(rows,(.1,.2)),evaluate_fixed_family(rows[::-1],(.1,.2)))

if __name__=="__main__": unittest.main()
