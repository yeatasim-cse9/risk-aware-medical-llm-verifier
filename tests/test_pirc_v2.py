import sys
from pathlib import Path
import unittest
from math import comb
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import pirc_v2_method as m
import pirc_v2_power as p
import pirc_calibration_prototype as v1

def row(a,b,changed=1):
    r=dict.fromkeys(m.FEATURES,0.0)
    r.update(original_support=a,revised_support=b,answer_changed=changed,evidence_available=1)
    return r

class TestPIRCV2(unittest.TestCase):
    def test_joint_changes_action_at_identical_marginals(self):
        x={"00":.05,"01":.25,"10":.05,"11":.65}
        y={"00":.0,"01":.30,"10":.10,"11":.60}
        # Both have p0=.70, p1=.90; different revision-induced error.
        # Choose penalty so revision benefit is between the two harm costs.
        args=dict(release_penalty=0,preservation_penalty=3,coverage_bonus=0)
        self.assertEqual(m.choose_action(x,**args)[0],"revise")
        self.assertEqual(m.choose_action(y,**args)[0],"accept")
    def test_joint_fit_and_structural_zeros(self):
        rows=[row(.9,.9,0),row(.1,.1,0),row(.1,.9),row(.9,.1)]*8
        targets=["11","00","01","10"]*8
        w=m.fit_joint(rows,targets)
        self.assertEqual(w,m.fit_joint(rows,targets))
        for r,t in zip(rows[:4],targets[:4]):
            q=m.predict_joint(w,r)
            self.assertAlmostEqual(sum(q.values()),1)
            self.assertEqual(max(q,key=q.get),t)
        self.assertEqual(m.predict_joint(w,rows[0])["10"],0)
        self.assertEqual(m.predict_joint(w,rows[0])["01"],0)
        with self.assertRaisesRegex(ValueError,"UNCHANGED_TRANSITION"):
            m.fit_joint([rows[0]],["01"])
    def test_missing_evidence_fails(self):
        r=row(.7,.8);r["evidence_available"]=0
        with self.assertRaisesRegex(ValueError,"MISSING_EVIDENCE"):
            m.features(r)
    def test_pmf_reference(self):
        for n in (0,10,60):
            for q in (0,.01,.2,.5,.95,1):
                pmf=m.binomial_pmf(n,q)
                self.assertAlmostEqual(sum(pmf),1)
                for x,y in enumerate(pmf):
                    self.assertAlmostEqual(y,comb(n,x)*q**x*(1-q)**(n-x),places=12)
    def test_pvalue_superuniform_at_boundary(self):
        for n in (5,20,50):
            for q in (.05,.1,.5):
                pmf=m.binomial_pmf(n,q)
                for alpha in (.001,.01,.05,.1):
                    self.assertLessEqual(sum(w for x,w in enumerate(pmf)
                        if m.lower_tail(n,x,q)<=alpha),alpha+1e-12)
                    self.assertLessEqual(sum(w for x,w in enumerate(pmf)
                        if m.upper_tail(n,x,q)<=alpha),alpha+1e-12)
    def test_feasibility_and_fail_closed(self):
        y=[True]*400+[False]*100
        good={"actions":["accept"]*400+["revise"]*100,"final":[True]*500}
        drop={"actions":["abstain"]*500,"final":[None]*500}
        harm={"actions":["revise"]*400+["accept"]*100,"final":[False]*500}
        a=m.certify_family(y,{"good":good,"drop":drop,"harm":harm})
        self.assertEqual(a["selected"],"good")
        self.assertFalse(a["policies"]["drop"]["passes"])
        self.assertFalse(a["policies"]["harm"]["passes"])
        self.assertFalse(a["clinical_safety_certified"])
        self.assertEqual(m.certify_family([False]*500,{"x":
            {"actions":["revise"]*500,"final":[True]*500}})["status"],"NO_CERTIFIED_POLICY")
    def test_holm_never_weaker_than_v1_on_same_family(self):
        for n in (50,100,200,500):
            y=[True]*n
            z={"actions":["accept"]*n,"final":[True]*n}
            policies={str(i):z for i in range(12)}
            a=v1.calibrate(y,policies);b=m.certify_family(y,policies)
            if a["selected"] is not None: self.assertIsNotNone(b["selected"])
    def test_zero_error_floors(self):
        for cap in (.05,.1):
            for delta in (.05,.05/12,.05/36):
                n=p.zero_error_minimum(cap,delta)
                self.assertLessEqual((1-cap)**n,delta)
                self.assertGreater((1-cap)**(n-1),delta)
    def test_planning_power(self):
        a=p.feasibility(200);b=p.feasibility(900)
        self.assertGreater(b["joint_power_lower_bound"],a["joint_power_lower_bound"])
        for x in (a,b):
            self.assertTrue(0<=x["joint_power_lower_bound"]<=x["joint_power_upper_bound"]<=1)
        self.assertEqual(p.mcnemar_power(10,0,0),0)
        self.assertGreater(p.mcnemar_power(500,.12,.04),p.mcnemar_power(500,.08,.04))
    def test_ap_ties_and_group_planning(self):
        self.assertEqual(p.average_precision([True,False],[.5,.5]),.5)
        self.assertEqual(p.average_precision([True,False],[.8,.2]),1)
        groups=[[(True,.9,.6),(False,.1,.4)],[(True,.8,.1),(False,.2,.9)],
                [(True,.7,.5),(False,.4,.3)],[(True,.6,.2),(False,.3,.8)]]
        a=p.pilot_ap_jackknife(groups)
        self.assertEqual(a["pilot_groups"],4)
        self.assertGreater(a["pseudovalue_sd"],0)
        self.assertGreater(p.ap_planning_groups(.35),p.ap_planning_groups(.15))
if __name__=="__main__":unittest.main()
