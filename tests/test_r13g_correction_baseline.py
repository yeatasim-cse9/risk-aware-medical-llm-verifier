import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import r13g_correction_baseline as b

class TestR13G(unittest.TestCase):
    def test_reference_agreement(self):
        self.assertEqual(b.reference({"label":b.LABELS[0],"answer":[b.LABELS[0]]}),b.LABELS[0])
        self.assertIsNone(b.reference({"label":b.LABELS[0],"answer":[b.LABELS[1]]}))
        self.assertIsNone(b.reference({"label":b.LABELS[0],"answer":[]}))
    def test_rule(self):
        ico=("drug A","placebo","blood pressure")
        self.assertEqual(b.predict(ico,"Drug A significantly decreased blood pressure."),b.LABELS[0])
        self.assertEqual(b.predict(ico,"There was no significant difference in blood pressure."),b.LABELS[1])
        self.assertIsNone(b.predict(ico,"Blood pressure was measured."))
    def test_groups(self):
        rows=[{"pmcid":1,"provisional_exact_match_component":1,"title_trial_ids":["NCT1"]},
              {"pmcid":2,"provisional_exact_match_component":2,"title_trial_ids":["NCT1","NCT2"]},
              {"pmcid":3,"provisional_exact_match_component":3,"title_trial_ids":["NCT2"]}]
        self.assertEqual(b.group_map(rows),{1:1,2:1,3:1})
    def test_metrics(self):
        groups={1:{"unchanged":[1,1,0,0,0],"majority":[1,0,0,1,1],"lexical":[1,1,2,0,1]}}
        m,c=b.summarize(groups)
        self.assertEqual(m["lexical"]["correction_success"],1)
        self.assertEqual(c["percentile_95_ci"],[1,1])
    def test_rights(self):
        r={"screen":"preliminary_rights_candidate","license_code":"CC BY","is_pmc_openaccess":True,
           "is_retracted":False,"version_count":1,"doi":"x","pmid":"x","title_sha256":"x"}
        self.assertTrue(b.eligible(r))
        r["is_retracted"]=True
        self.assertFalse(b.eligible(r))
