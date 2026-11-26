import sys,json,hashlib
from pathlib import Path
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import e1_master01d_v2_integrity as d
import e1_master01d_v2_cohort as q
import e1_master01c_core as c
import e1_readiness
from test_e1_master01c_core import fixture,norm

def row(i,split="development",sources=None,**kw):
    z=dict(dataset="bioasq14b",release="synthetic-release",id=str(i),task="yes_no",
           native_split=split,sources=("pmid:"+str(i),) if sources is None else sources,
           question="invented question "+str(i),context="invented context "+str(i),
           evidence_available=True,reference_available=True,provenance_known=True)
    z.update(kw);return z
def plan(rows,**kw):
    return q.plan(rows,dataset="bioasq14b",release="synthetic-release",task="yes_no",**kw)

class TestMaster01DV2(unittest.TestCase):
    def test_historical_checkpoint_and_sources(self):
        r=json.loads((ROOT/"configs/master01c-verified-aggregate-2026-11-26.json").read_text())
        self.assertTrue(d.verify_checkpoint(r)["historical_aggregate_verified"])
        d.verify_prior_source_files()
        r["clinical_labels_approved"]=True
        with self.assertRaisesRegex(ValueError,"REPORT_INTEGRITY"):
            d.verify_checkpoint(r)
        r["integrity_sha256"]=c.digest({k:v for k,v in r.items() if k!="integrity_sha256"})
        with self.assertRaisesRegex(ValueError,"HISTORICAL_CHECKPOINT"):
            d.verify_checkpoint(r)
    def test_reconstruct_preserves_holdout_and_quarantine(self):
        rows,flags=fixture()
        kept,mapping,agg=d.reconstruct_candidate(rows,flags,norm)
        self.assertEqual([len(kept[s]) for s in c.SPLITS],[3,3,3,4])
        self.assertEqual(kept["official_holdout"],rows["official_holdout"])
        self.assertEqual(agg["quarantine"]["residual_flagged_pairs"],0)
        self.assertEqual(d.reconstruct_candidate(rows,flags,norm),(kept,mapping,agg))
    def test_supplemental_exact_conflict(self):
        rows,flags=fixture()
        rows["train"][3]["question"]="Ａlpha!"
        rows["dev"][3]["question"]="Alpha"
        with self.assertRaisesRegex(ValueError,"SUPPLEMENTAL_PROXY_CONFLICT"):
            d.reconstruct_candidate(rows,flags,norm)
    def test_holdout_pin(self):
        raw=json.dumps({str(i):"unused" for i in range(500)}).encode()
        records={"official_holdout":[{"source":str(i)} for i in range(500)]}
        with patch.dict(d.prior.DATA_PINS,{"official_500_sha256":hashlib.sha256(raw).hexdigest()}):
            d.verify_holdout_identity(records,raw)
            records["official_holdout"][0]["source"]="different"
            with self.assertRaisesRegex(ValueError,"HOLDOUT_IDENTITY"):
                d.verify_holdout_identity(records,raw)
            with self.assertRaisesRegex(ValueError,"HOLDOUT_FILE_PIN"):
                d.verify_holdout_identity(records,b"bad")
    def test_shortcut_scope(self):
        rows,flags=fixture();kept,_,_=d.reconstruct_candidate(rows,flags,norm)
        answers={r["key"]:("same","same") for s in ("train","dev") for r in kept[s]}
        a=d.shortcut_audit(kept,answers)
        self.assertEqual(a["dev"]["identical_normalized_roles"],3)
        answers[kept["test"][0]["key"]]=("a","b")
        with self.assertRaisesRegex(ValueError,"TRAIN_DEV_ANSWER_SCOPE"):
            d.shortcut_audit(kept,answers)
    def test_protected_transitive_overlap(self):
        rows=[row(1,"protected",dataset="prior01c",sources=("pmid:1",)),
              row(2,sources=("pmid:1","pmid:2")),row(3,sources=("pmid:2",)),
              row(4,"official_test"),row(5,sources=("pmid:4",)),row(6)]
        mapping,a=plan(rows)
        self.assertEqual(a["excluded_rows"]["prior_protected_overlap"],2)
        self.assertEqual(a["excluded_rows"]["official_test_overlap"],1)
        self.assertEqual(a["eligible_group_representatives"]["test"],1)
        self.assertEqual(len(mapping),2)
        self.assertEqual(plan(list(reversed(rows))),(mapping,a))
    def test_semantic_proxy_and_unknown_provenance(self):
        rows=[row(1,"protected",dataset="prior01c"),row(2),row(3,provenance_known=False),
              row(4),row(5,sources=("pmid:4",))]
        links=[(("prior01c","synthetic-release","1"),("bioasq14b","synthetic-release","2"))]
        m,a=plan(rows,near_links=links)
        self.assertEqual(a["excluded_rows"]["prior_protected_overlap"],1)
        self.assertEqual(a["excluded_rows"]["missing_eligibility"],1)
        self.assertEqual(a["excluded_rows"]["nonrepresentative_same_group"],1)
        self.assertEqual(len(m),1)
        self.assertFalse(a["complete_semantic_screen_verified"])
    def test_no_mixed_task_or_restricted_or_gold(self):
        for rows,reason in [([row(1,task="mcq")],"MIXED_TARGET"),
                 ([row(1,dataset="medval")],"RESTRICTED_DATASET"),
                 ([row(1,gold="yes")],"ROW_SCHEMA"),([row(1),row(1)],"DUPLICATE_ROW")]:
            with self.assertRaisesRegex(ValueError,reason):plan(rows)
    def test_power_and_no_approval(self):
        self.assertEqual(q.prospective_power(0)["status"],"NO_ELIGIBLE_CALIBRATION_GROUPS")
        self.assertEqual(q.prospective_power(30)["scenario_count"],6)
        a=d.gate_report()
        for k in ("clinical_labels_approved","human_review_completed","final_split_approved",
                  "final_split_frozen","e1_execution_authorized","scientific_experiment_executed"):
            self.assertFalse(a[k])
        self.assertFalse(e1_readiness.check()["ready"])
if __name__=="__main__":unittest.main()
