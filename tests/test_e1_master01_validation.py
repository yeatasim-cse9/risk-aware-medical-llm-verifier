"""Synthetic-only MASTER 01 tests; no network or real medical data."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import e1_master01_validation_kaggle as m

class TestMaster01(unittest.TestCase):
    def test_exact_pairs_and_no_text_export(self):
        sets = {s:[{"question":"Invented "+s,"context":"context "+s}] for s in m.SPLITS}
        sets["dev"][0]["question"] = sets["train"][0]["question"].upper()
        result = m.exact_screen(sets,lambda s:s.lower())
        self.assertEqual(result["question"]["train__dev"],{"shared_groups":1,"left_rows":1,"right_rows":1})
        self.assertEqual(len(result["question"]),6)
        self.assertNotIn("Invented",str(result))
    def test_diagnostics_counts_and_privacy(self):
        med={"fake":[{"Category of Hallucination":"private_category","Difficulty Level":"private_level"}]}
        out=m.grouped_diagnostics(med,{("fake",0):"train"})
        row=out["fake"]["train"]
        self.assertEqual(row["benchmark_role_0"],1)
        self.assertEqual(row["benchmark_role_1"],1)
        self.assertEqual(row["category_frequency_sorted"],[1])
        self.assertNotIn("private_category",str(out))
    def test_canonical(self):
        self.assertEqual(m.canonical({"b":1,"a":2}),b'{"a":2,"b":1}')
    def test_pairwise_blocked(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed in lightweight repository CI")
        a=torch.tensor([[1.,0.],[0.,1.],[1.,0.]])
        b=torch.tensor([[1.,0.],[0.,1.]])
        out=m.overlap(a,b,block=1,thresholds=(0.9,))
        self.assertEqual(out["pairs_compared"],6)
        self.assertEqual(out["thresholds"]["0.9"],{"pairs":3,"left_rows":3,"right_rows":2})
    def test_require_fail_closed(self):
        with self.assertRaises(RuntimeError):
            m.require(False,"EXPECTED_FAILURE")
