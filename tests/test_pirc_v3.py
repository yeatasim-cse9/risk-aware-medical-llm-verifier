import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import pirc_v3_features as f
import pirc_v3_baselines as b
import pirc_v2_method as m

def nli(e,c): return {"entailment":e,"contradiction":c,"neutral":1-e-c}
def candidate(text,scores,ids=("a","b")):
    return {"text":text,"confidence":.7,"units":[{"id":k,"nli":s} for k,s in zip(ids,scores)]}
def row(a,z,changed=1):
    r=dict.fromkeys(m.FEATURES,0.)
    r.update(original_support=a,revised_support=z,answer_changed=changed,evidence_available=1)
    return r

class TestV3(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(f.parse_answer(" YES ",("yes","no","maybe")),"yes")
        for bad in ("yes.","answer: yes","yes or no","maybe"):
            self.assertEqual(f.parse_answer(bad,("yes","no")),"INVALID")
    def test_features(self):
        a=candidate("yes",[nli(.8,.1),nli(.3,.5)])
        z=candidate("no",[nli(.2,.7),nli(.6,.1)])
        x=f.aligned_extract(a,z,unit_ids=("a","b"),allowed=("yes","no"))["features"]
        self.assertAlmostEqual(x["original_support"],.8)
        self.assertAlmostEqual(x["original_contradiction"],.5)
        self.assertAlmostEqual(x["original_evidence_coverage"],.5)
        self.assertAlmostEqual(x["original_support_drop"],.5)
        self.assertEqual(x["revised_support_drop"],0.)
        self.assertEqual(x["answer_changed"],1)
    def test_misalignment_and_gold_rejected(self):
        a=candidate("yes",[nli(.8,.1),nli(.3,.5)])
        z=candidate("no",[nli(.2,.7),nli(.6,.1)],("b","a"))
        with self.assertRaisesRegex(ValueError,"UNIT_ALIGNMENT"):
            f.aligned_extract(a,z,unit_ids=("a","b"),allowed=("yes","no"))
        a["gold"]="yes"
        with self.assertRaisesRegex(ValueError,"ALIGNED_SCHEMA"):
            f.aligned_extract(a,z,unit_ids=("a","b"),allowed=("yes","no"))
    def test_empty_and_invalid(self):
        a=candidate("unknown",[],());z=candidate("yes",[],())
        q=f.aligned_extract(a,z,unit_ids=(),allowed=("yes","no"))
        self.assertFalse(q["original_valid"])
        self.assertTrue(q["revised_valid"])
        self.assertEqual(q["features"]["evidence_available"],0)
        self.assertEqual(q["features"]["original_support"],0)
        self.assertEqual(b.enforce_valid_action("accept",original_valid=False,revised_valid=True),"abstain")
        self.assertEqual(b.enforce_valid_action("revise",original_valid=False,revised_valid=True),"revise")
    def test_bad_nli(self):
        with self.assertRaisesRegex(ValueError,"NLI_PROBABILITIES"):
            f.aggregate([{"entailment":.7,"contradiction":.8,"neutral":0}])
    def test_frechet(self):
        for p0 in (0.,.3,.7,1.):
            for p1 in (0.,.2,.9,1.):
                for h in (max(0.,p0-p1),min(p0,1-p1)):
                    q=m.joint(b.feasible_joint(p0,p1,h))
                    self.assertAlmostEqual(q["10"]+q["11"],p0)
                    self.assertAlmostEqual(q["01"]+q["11"],p1)
        with self.assertRaisesRegex(ValueError,"FRECHET"):
            b.feasible_joint(.7,.9,.2)
        q=b.stress_test(.7,.9,release_penalty=0.,preservation_penalty=3.,coverage_bonus=0.)
        self.assertEqual((q["low_action"],q["high_action"]),("revise","accept"))
    def test_comparators(self):
        rows=[row(.9,.9,0),row(.1,.1,0),row(.1,.9),row(.9,.1),row(.1,.2),row(.9,.8)]*4
        targets=["11","00","01","10","00","11"]*4
        fit=b.fit_comparators(rows,targets)
        self.assertEqual(fit,b.fit_comparators(rows,targets))
        for r in rows[:6]:
            out=b.predict_comparators(fit,r)
            self.assertEqual(len(out),4)
            for q in out.values():
                m.joint(q)
                if not r["answer_changed"]:
                    self.assertEqual(q["10"]+q["01"],0.)
        with self.assertRaisesRegex(ValueError,"MISSING_TRAIN_STRATUM"):
            b.fit_comparators([row(.5,.5)],["11"])
    def test_inventory(self):
        a={"source_group":"synthetic-1","partition":"train","eligible":True}
        z={"source_group":"synthetic-2","partition":"calibration","eligible":False}
        r=b.calibration_inventory([a,z])
        self.assertEqual(r["eligible_group_counts"]["train"],1)
        self.assertEqual(r["eligible_group_counts"]["calibration"],0)
        self.assertFalse(r["execution_authorized"])
        with self.assertRaisesRegex(ValueError,"DUPLICATE_SOURCE_GROUP"):
            b.calibration_inventory([a,{**a,"partition":"test"}])
    def test_baseline_actions(self):
        a=b.baseline_actions({"00":.05,"01":.25,"10":.05,"11":.65},
            changed=True,release_penalty=0.,preservation_penalty=3.)
        self.assertEqual(a["joint_policy"],"revise")
        self.assertEqual(a["independence_policy"],"accept")

if __name__=="__main__":unittest.main()
