"""Strict invented metadata and hypothetical numerical planning tests only."""
import sys,unittest,math,dataclasses
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import e1_master01f_c_qualification_v1 as q
import e1_master01f_c_planning_v1 as p
from e1_master01f_b_calibration_v1 import THRESHOLDS

def meta(n,prefix,part):
    return [{"parent":prefix+"p"+str(i),"source_group":prefix+"g"+str(i),
             "partition":part} for i in range(n)]
def setup():
    tr=meta(140,"t","train");dv=meta(4,"d","dev")
    protected=frozenset(x["source_group"] for x in tr+dv)
    inv=q.CandidateInventory(protected|frozenset("new"+str(i) for i in range(20)),
        protected,frozenset({"new0"}),frozenset({"new1"}))
    return tr,dv,inv

class Exposure(unittest.TestCase):
    def test_all_prior_label_exposure_not_just_model_fit(self):
        tr,dv,inv=setup();s=q.existing_scope(tr,dv)
        self.assertEqual(len(s["prior_fit"]),128)
        self.assertEqual(len(s["label_development_exposed"]),144)
        self.assertFalse(q.eligible_inventory(inv,s)&s["label_development_exposed"])
        self.assertEqual(len(q.eligible_inventory(inv,s)),18)
        self.assertEqual(q.existing_scope(tr[::-1],dv[::-1]),s)
    def test_incomplete_protected_inventory_fails(self):
        tr,dv,inv=setup()
        with self.assertRaisesRegex(ValueError,"PROTECTED_SCOPE_INCOMPLETE"):
            q.report(tr,dv,dataclasses.replace(inv,protected_groups=frozenset()),target_groups=10)
    def test_wrong_scope_and_answer_fields_rejected(self):
        tr,dv,inv=setup()
        for bad in ([{**dv[0],"partition":"test"}]+dv[1:],
                    [{**dv[0],"answers":{}}]+dv[1:],
                    [{**dv[0],"source_group":"tg0"}]+dv[1:]):
            with self.assertRaises(ValueError):q.report(tr,bad,inv,target_groups=10)
    def test_report_strict_no_identifiers_or_approval(self):
        tr,dv,inv=setup()
        a=q.report(tr,dv,inv,target_groups=10)
        self.assertEqual(a["existing_confirmatory_calibration"],"BLOCKED_PRIOR_LABEL_EXPOSURE")
        self.assertEqual(a["candidate_structural_status"],"TARGET_MET_METADATA_ONLY")
        self.assertTrue(all(a[k] is False for k in q.FALSE_FLAGS))
        self.assertEqual(q.validate_report(a,tr,dv,inv,target_groups=10),a)
        for bad in ({**a,"execution_authorized":True},{**a,"private_rows":["example"]},
                    {**a,"schema_version":True},{**a,"eligible_count_band":"new0"}):
            with self.assertRaises(ValueError):q.validate_report(bad,tr,dv,inv,target_groups=10)
        self.assertEqual(q.report(tr,dv,inv,target_groups=19)["candidate_structural_status"],
                         "INSUFFICIENT_ELIGIBLE_GROUPS")
        for n,band in ((0,"zero"),(1,"small_suppressed"),(9,"small_suppressed"),(10,"at_least_10")):
            self.assertEqual(q._bucket(n),band)
    def test_exploratory_only_explicit(self):
        tr,dv,inv=setup()
        plan,status=q.exploratory_plan(tr,dv,calibration_groups=3,additional_model_exposure=frozenset())
        self.assertEqual(status,"EXPLORATORY_PRIOR_LABEL_EXPOSURE")
        self.assertTrue({g for a,g,k in plan.members if k=="calibration"}<=
                         q.existing_scope(tr,dv)["label_development_exposed"])

class Power(unittest.TestCase):
    def test_exact_joint_reference_enumeration(self):
        # Different hypothetical caps to permit nonzero probability at small n.
        n=15;rr=.15;cv=.8;cap=.6;floor=.2;alpha=.12;delta=alpha/(2*len(THRESHOLDS))
        def pm(n,k,x):return math.comb(n,k)*x**k*(1-x)**(n-k)
        expected=0.
        for m in range(1,n+1):
            if sum(pm(n,j,floor) for j in range(m,n+1))>delta:continue
            for e in range(m+1):
                if sum(pm(m,j,cap) for j in range(e+1))<=delta:
                    expected+=pm(n,m,cv)*pm(m,e,rr)
        actual=p.group_power(n,true_risk=rr,true_coverage=cv,risk_cap=cap,
                             coverage_floor=floor,alpha=alpha)
        self.assertAlmostEqual(actual["joint_power_one_fixed_threshold"],expected,places=11)
        self.assertGreater(expected,0)
    def test_power_boundary_and_planning(self):
        a=p.group_power(100,true_risk=.03,true_coverage=.75)
        b=p.group_power(100,true_risk=.07,true_coverage=.75)
        self.assertEqual(a["minimum_released_zero_errors"],53)
        self.assertGreater(a["joint_power_one_fixed_threshold"],b["joint_power_one_fixed_threshold"])
        self.assertEqual(p.group_power(10,true_risk=.03,true_coverage=.75)["joint_power_one_fixed_threshold"],0)
        self.assertEqual([x["groups"] for x in p.ap_sensitivity()],[442,1227,2404])
        for x in (0,True,float("nan"),1):
            with self.assertRaises(ValueError):p.group_power(100,true_risk=x,true_coverage=.75)

class Compute(unittest.TestCase):
    def inputs(self):
        return dict(measured_train_parents=20,measured_dev_parents=10,
            measured_train_inference_seconds=100.,measured_dev_inference_seconds=60.,
            measured_pre_inference_seconds=120.,measured_post_inference_seconds=10.,
            fit_parents=40,selection_parents=20,calibration_parents=30)
    def test_exact_budget_and_safety(self):
        x=self.inputs();a=p.compute_plan(**x)
        self.assertEqual(a["total_parents"],90)
        self.assertEqual(a["forward_examples"],8640)
        self.assertEqual(a["padded_positions"],4423680)
        self.assertEqual(a["seconds_per_parent_reference"],6.)
        self.assertEqual(a["projected_seconds"],670.)
        self.assertAlmostEqual(a["planning_seconds_with_margin"],871.)
        self.assertFalse(a["quota_certified"])
        self.assertEqual(p.validate_compute(a,**x),a)
        with self.assertRaises(ValueError):p.validate_compute({**a,"quota_certified":True},**x)
    def test_bad_inputs(self):
        for k,v in (("fit_parents",True),("calibration_parents",-1),
                    ("measured_train_inference_seconds",float("inf")),
                    ("safety_multiplier",.5)):
            with self.assertRaises(ValueError):p.compute_plan(**{**self.inputs(),k:v})
if __name__=="__main__":unittest.main()
