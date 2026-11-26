"""Invented nonmedical fixtures only. Never loads private results, models or data."""
import sys,unittest,dataclasses,copy
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import e1_master01f_b_v1 as f
import e1_master01f_b_calibration_v1 as c
import e1_master01e_v2_pilot as prior

def meta(n,prefix,partition):
    return [{"parent":f"{prefix}p{i}","source_group":f"{prefix}g{i}","partition":partition} for i in range(n)]
def row(p,g,k):
    return {"parent":p,"source_group":g,"partition":k,"question":"What color is the shape",
            "context":"The blue circle has three dots "+p,
            "answers":{"ground_truth":"blue circle","hallucinated":"red square"}}
def small():
    members=tuple((k+str(i),k+"g"+str(i),k) for k in f.PARTITIONS for i in range(3))
    p=f.Plan(members,frozenset())
    blocks={k:[row(a,g,s) for a,g,s in members if s==k] for k in f.PARTITIONS}
    return p,blocks
def neural(rows):
    return {k:[(x["source_group"],x["parent"],([.8,.1,.7,.2,.5,.6],[.1,.8,.2,.7,.1,.2]))
            for x in rows] for k in f.CONDITIONS}

class Partition(unittest.TestCase):
    def test_prior_sample_and_exposure(self):
        tr=meta(140,"t","train");dv=meta(4,"d","dev")
        # Extra parent in same group must never split.
        tr.append({"parent":"extra","source_group":"tg0","partition":"train"})
        expected={x["source_group"] for x in prior.sample([row(x["parent"],x["source_group"],"train") for x in tr],"train")}
        self.assertEqual(f.prior_pilot_groups(tr),expected)
        p=f.plan(tr,dv,calibration_groups=6,additional_exposed_fit_groups=frozenset({"tg0"}))
        self.assertEqual(p,f.plan(tr[::-1],dv[::-1],calibration_groups=6,additional_exposed_fit_groups=frozenset({"tg0"})))
        self.assertFalse(p.exposed_fit_groups & {g for a,g,k in p.members if k=="calibration"})
        self.assertEqual({k for a,g,k in p.members if g=="tg0"},{"fit"})
        self.assertEqual({k for a,g,k in p.members if a.startswith("dp")},{"selection"})
    def test_reject_metadata_leak_and_overlap(self):
        tr=meta(140,"t","train");dv=meta(4,"d","dev")
        def make(t,d,ex=frozenset()):return f.plan(t,d,calibration_groups=4,additional_exposed_fit_groups=ex)
        for bad in ([{**dv[0],"answers":{}}]+dv[1:],
                    [{**dv[0],"partition":"test"}]+dv[1:],
                    [{**dv[0],"source_group":"tg0"}]+dv[1:],
                    [{**dv[0],"parent":"tp0"}]+dv[1:]):
            with self.assertRaises(ValueError):make(tr,bad)
        with self.assertRaises(ValueError):make(tr,dv,frozenset({"unknown"}))
        with self.assertRaises(ValueError):make(tr,dv,frozenset(x["source_group"] for x in tr))
    def test_forged_plan_rejects_cross_group_and_exposed_calibration(self):
        p,_=small()
        for q in (f.Plan(p.members+(("other","fitg0","calibration"),),frozenset()),
                  f.Plan(p.members,frozenset({"calibrationg0"}))):
            with self.assertRaises(ValueError):f.validate_plan(q)

class LexicalHybrid(unittest.TestCase):
    def test_evidence_only_and_shortcut_controls(self):
        x={"question":"blue circle","answer":"blue circle","context":"blue circle with dots"}
        y={**x,"question":"unrelated"}
        self.assertEqual(f.lexical(x,"evidence_only"),f.lexical(y,"evidence_only"))
        self.assertEqual(f.lexical({**x,"context":""},"evidence_only"),[0,0])
        self.assertEqual(f.lexical({**x,"context":""},"question_answer_overlap"),[1])
        self.assertEqual(f.lexical({**x,"context":""},"legacy_lexical"),[0,0,1,0])
        # Set overlap is not semantic equivalence: polarity and role reversal.
        self.assertEqual(f.evidence_only("blue not red","red not blue"),[1,1])
        self.assertLess(f.evidence_only("three dots","four dots")[0],1)
    def test_ablation_fit_and_scoring_scope(self):
        p,d=small()
        m=f.fit(p,d["fit"],"evidence_only")
        full=f.score(m,p,d["selection"],"selection")
        rem=f.score(m,p,d["selection"],"selection",condition="removed")
        self.assertTrue(all(a==b for g,q,(a,b) in rem))
        self.assertGreater(f.b.metrics(full)["average_precision"],f.b.metrics(rem)["average_precision"])
        self.assertEqual(len(f.equal_length_pairs(d["selection"],full)),3)
        for bad in (d["selection"],d["calibration"]):
            with self.assertRaises(ValueError):f.fit(p,bad,"evidence_only")
        with self.assertRaises(ValueError):f.score(m,p,d["fit"],"fit")
        with self.assertRaises(ValueError):f.score(m,p,d["selection"][:-1],"selection")
        with self.assertRaises(ValueError):f.score(m,p,[{**x,"partition":"test"} for x in d["selection"]],"selection")
    def test_hybrid_and_neural_alignment(self):
        p,d=small();tr=d["fit"];dv=d["selection"]
        for mode,dim in (("neural",6),("hybrid",8),("hybrid_legacy",10)):
            m=f.fit(p,tr,mode,neural=neural(tr))
            self.assertEqual(len(m.head[0]),dim)
            scores=f.score(m,p,dv,"selection",neural=neural(dv))
            self.assertEqual(scores,f.score(m,p,dv[::-1],"selection",
                neural={k:v[::-1] for k,v in neural(dv).items()}))
        m=f.fit(p,tr,"hybrid",neural=neural(tr))
        for bad in ({}, {**neural(dv),"full":neural(tr)["full"]},
                    {**neural(dv),"full":[neural(dv)["full"][0]]*3}):
            with self.assertRaises(ValueError):f.score(m,p,dv,"selection",neural=bad)
        z=neural(dv);z["full"][0][2][0][0]=float("nan")
        with self.assertRaises(ValueError):f.score(m,p,dv,"selection",neural=z)
    def test_permutation_matches_original_label_blind_donor(self):
        p,d=small();dv=d["selection"]
        for i,x in enumerate(dv):x["context"]=("blue circle","red square","green triangle")[i]
        a=f.design(p,dv,"selection","permuted","evidence_only")
        donor={x["parent"]:x["context"] for x in f.b.context_permutation(dv,"selection",f.SEED)}
        for g,parent,fs in a:
            self.assertEqual(fs[0],f.evidence_only("blue circle",donor[parent]))

class Statistics(unittest.TestCase):
    def test_common_group_bootstrap_and_ties(self):
        p,d=small()
        scores=[(x["source_group"],x["parent"],(.2,.8)) for x in d["selection"]]
        out=f.compare_family(p,"selection",{"a":scores,"b":scores[::-1]},(("a","b"),),replicates=100)
        self.assertEqual(out["groups"],3)
        for v in out["contrasts"][0]["metrics"].values():
            self.assertEqual(v,{"difference":0.,"percentile_95":[0.,0.]})
        self.assertFalse(out["multiplicity_adjusted"])
        with self.assertRaises(ValueError):f.compare_family(p,"calibration",{"a":scores,"b":scores},(("a","b"),))
        with self.assertRaises(ValueError):f.compare_family(p,"selection",{"a":scores,"b":scores[:-1]},(("a","b"),))
    def test_calibration_frozen_scope_and_no_policy(self):
        p,d=small();m=f.fit(p,d["fit"],"evidence_only");policy=c.freeze(m)
        with patch.object(f,"fit",side_effect=AssertionError("no fitting")):
            out=c.calibration(policy,p,d["calibration"])
        self.assertEqual(out["status"],"NO_POLICY_MEETS_BOUNDS")
        self.assertIsNone(out["selected_threshold"])
        self.assertFalse(out["clinical_safety_certified"])
        self.assertEqual(len(out["policies"]),6)
        with self.assertRaises(ValueError):c.calibration(policy,p,d["selection"])
        with self.assertRaises(ValueError):c.calibration(dataclasses.replace(policy,thresholds=(.2,)),p,d["calibration"])
    def test_calibration_bounds_and_group_endpoint(self):
        # 1000 independent invented groups, no erroneous accepted answers.
        p,d=small();members=tuple(x for x in p.members if x[2]!="calibration")
        members+=tuple(("p"+str(i),"g"+str(i),"calibration") for i in range(1000))
        p=f.Plan(members,frozenset());m=f.fit(p,d["fit"],"evidence_only")
        rr=[row(a,g,k) for a,g,k in p.members if k=="calibration"]
        scores=[(x["source_group"],x["parent"],(.01,.99)) for x in rr]
        with patch.object(f,"score",return_value=scores):
            out=c.calibration(c.freeze(m),p,rr)
        self.assertEqual(out["status"],"BOUNDS_MET_UNDER_ASSUMPTIONS")
        self.assertEqual(out["selected_threshold"],.05)
        self.assertEqual(out["policies"][0]["error_groups"],0)
        self.assertEqual(out["policies"][0]["answer_coverage_descriptive"],.5)
        self.assertFalse(out["revision_evaluated"])
if __name__=="__main__":unittest.main()
