"""Synthetic MASTER 01C integration/negative tests; no medical data or credentials."""
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import e1_master01c_core as c
import e1_master01c_private as p
import e1_master01c_validate as v
import e1_master01c_workflow as w
import e1_master01_validation_kaggle as b
import e1_master01b_kaggle as prior
from test_e1_master01c_core import fixture,norm

COMMIT="a"*40
HASHES=dict(zip(v.MODULES,(p.BASE_SHA,p.PRIOR_SHA,"b"*64,"c"*64)))
CATS={"pqa_artificial":{"train":[5472,879,793,41],"dev":[675,115,98,2],"test":[682,117,97,3]},
      "pqa_labeled":{"train":[300,84,10],"dev":[47,8,2],"test":[33,14,2]}}
DIFF={"pqa_artificial":{"train":[2556,2333,2296],"dev":[320,291,279],"test":[304,298,297]},
      "pqa_labeled":{"train":[167,121,106],"dev":[29,14,14],"test":[20,18,11]}}

def seal(d):
    d.pop("integrity_sha256",None)
    d["integrity_sha256"]=c.digest(d)
    return d

def maximums():
    return {k:{view:dict(zip(p.MAX_KEYS,vals)) for view,vals in x.items()}
            for k,x in p.EXPECTED_MAX.items()}

def dist(x,y):
    x=list(x); y=list(y)
    a=x+[0]*max(0,len(y)-len(x));z=y+[0]*max(0,len(x)-len(y))
    tv=round(sum(abs(i/sum(a)-j/sum(z)) for i,j in zip(a,z))/2,8)
    return {"before_sorted":sorted(x,reverse=True),"after_sorted":sorted(y,reverse=True),
            "total_variation":tv}

def synthetic_report():
    # Counts mirror pinned input denominators; invented distribution of removals.
    # This is NOT a real-data result or proof of retained medical cohorts.
    counts={};balance={}
    for s,n in prior.SIZES.items():
        f=p.EXPECTED_ENDPOINTS[s];removed=f if s!="official_holdout" else 0
        m=n-removed
        counts[s]={"original":n,"flagged_endpoints":f,"retained":m,
                   "quarantined":removed,"paired_examples":2*m if s!="official_holdout" else 0}
        if s=="official_holdout": continue
        x=CATS["pqa_artificial"][s];y=CATS["pqa_labeled"][s]
        u=DIFF["pqa_artificial"][s];z=DIFF["pqa_labeled"][s]
        xx=[x[0]-removed,*x[1:]];uu=[u[0]-removed,*u[1:]]
        def add(a,b):
            return [i+j for i,j in zip(a,b+[0]*(len(a)-len(b)))]
        balance[s]={"config":dist([sum(x),sum(y)],[sum(xx),sum(y)]),
            "category":dist(add(x,y),add(xx,y)),"difficulty":dist(add(u,z),add(uu,z)),
            "by_config":{"pqa_artificial":{"category":dist(x,xx),"difficulty":dist(u,uu)},
                         "pqa_labeled":{"category":dist(y,y),"difficulty":dist(z,z)}}}
    strata={}
    for a,z in c.PAIRS:
        name=a+"__"+z;f=p.EXPECTED_UNION[name]
        strata[name]={"flagged_population":f,"unflagged_population":prior.SIZES[a]*prior.SIZES[z]-f,
                      "unflagged_review_sample":100}
    d={"schema_version":1,"status":"PROVISIONAL_REMEDIATION_REVIEW_REQUIRED",
       "previous_evidence_sha256":p.PRIOR_EVIDENCE,"previous_mapping_sha256":b.EXPECTED_MAPPING,
       "seed":c.SEED,"encoder_aggregate_sha256":p.EXPECTED_ENCODERS,"maxima":maximums(),
       "source_commit":b.BASE,"source_module_sha256":b.PINS,"dataset_pins":p.DATA_PINS,
       "model_asset_set_sha256":p.EXPECTED_ASSETS,
       "strata":strata,"review_queue_count":771,
       "quarantine":{"counts":counts,"mapping_sha256":"d"*64,
                     "residual_flagged_pairs":0,"source_proxy_conflicts":0},
       "balance":balance,"versions":p.OLD_VERSIONS,"device":"Tesla T4",
       "gates":{"missing_gates":list(c.GATES),"external_authentication_required":True,
                "final_split_approved":False,"final_split_frozen":False,
                "scientific_e1_experiment_executed":False},
       "package_source_commit":COMMIT,"package_hashes":HASHES,**dict.fromkeys(v.FALSE_FLAGS,False)}
    return seal(copy.deepcopy(d))

class TestMaster01CIntegration(unittest.TestCase):
    def test_validator_accepts_synthetic(self):
        d=synthetic_report()
        self.assertEqual(v.validate(d,COMMIT,HASHES),d)
    def test_integrity_and_false_approval(self):
        d=synthetic_report();d["human_review_completed"]=True
        with self.assertRaisesRegex(ValueError,"REPORT_INTEGRITY"):
            v.validate(d,COMMIT,HASHES)
        with self.assertRaisesRegex(ValueError,"FALSE_APPROVAL"):
            v.validate(seal(d),COMMIT,HASHES)
        d=synthetic_report();d["gates"]["final_split_frozen"]=True
        with self.assertRaisesRegex(ValueError,"FALSE_FREEZE"):
            v.validate(seal(d),COMMIT,HASHES)
    def test_no_private_fields(self):
        d=synthetic_report();d["review_pair_ids"]=["invented"]
        with self.assertRaisesRegex(ValueError,"AGGREGATE_SCHEMA"):
            v.validate(seal(d),COMMIT,HASHES)
        d=synthetic_report();d["balance"]["train"]["by_config"]["invented_private_key"]={}
        with self.assertRaisesRegex(ValueError,"AGGREGATE_SCHEMA"):
            v.validate(seal(d),COMMIT,HASHES)
    def test_holdout_and_paired_counts(self):
        d=synthetic_report();d["quarantine"]["counts"]["official_holdout"].update(
            retained=499,quarantined=1)
        with self.assertRaisesRegex(ValueError,"HOLDOUT_CHANGED"):
            v.validate(seal(d),COMMIT,HASHES)
        d=synthetic_report();d["quarantine"]["counts"]["train"]["paired_examples"]+=1
        with self.assertRaisesRegex(ValueError,"PAIRED_RETENTION"):
            v.validate(seal(d),COMMIT,HASHES)
    def test_pins_and_maximum_tolerance(self):
        d=synthetic_report();d["package_hashes"][v.MODULES[2]]="e"*64
        with self.assertRaisesRegex(ValueError,"PACKAGE_PIN"):
            v.validate(seal(d),COMMIT,HASHES)
        x=maximums();x["minilm"]["question"]["train__test"]+=1e-6
        p.compare_maxima(x)
        x["minilm"]["question"]["train__test"]+=1e-3
        with self.assertRaisesRegex(ValueError,"NUMERICAL_REPRODUCTION"):
            p.compare_maxima(x)
    def test_rights_gate_before_data_access(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,"RIGHTS_REQUIRED"):
                p.prepare_private(Path("not-used"))
            with self.assertRaisesRegex(ValueError,"RIGHTS_REQUIRED"):
                p.reconstruct_private(Path("not-used"))
        with patch.dict(os.environ,{"MASTER01C_PRIVATE_RIGHTS_CONFIRMED":"YES"},clear=True):
            with self.assertRaisesRegex(ValueError,"PRIVATE_REVIEW_REQUIRED"):
                p.prepare_private(Path("not-used"),return_private=True)
    def test_duplicate_submission_preflight(self):
        with patch.object(w.subprocess,"run",return_value=MagicMock(returncode=0)):
            with patch.object(w,"check",side_effect=["complete","missing"]):
                w.preflight()
            for state in ("queued","running","complete","error","cancelled"):
                with patch.object(w,"check",side_effect=["complete",state]):
                    with self.assertRaisesRegex(ValueError,"DUPLICATE"):
                        w.preflight()
            with patch.object(w,"check",return_value="error"):
                with self.assertRaisesRegex(ValueError,"PREVIOUS_RUN"):
                    w.preflight()
    def test_blinded_semantic_packets(self):
        rows,flags=fixture()
        queue,_=c.review_plan(rows,flags,norm,3)
        packet,key=c.semantic_packets(rows,queue,b"x"*32)
        self.assertEqual(len(packet),len(queue))
        self.assertEqual(len(key),len(queue))
        self.assertTrue(all(set(x)=={"review_id","cases"} for x in packet))
        self.assertTrue(all(set(y)=={"question","context","source"} for x in packet for y in x["cases"]))
        self.assertEqual(c.semantic_packets(rows,queue,b"x"*32),(packet,key))
        with self.assertRaisesRegex(ValueError,"PRIVATE_REVIEW_NONCE"):
            c.semantic_packets(rows,queue,b"x")
    def test_clinical_pairs_and_blinding(self):
        rows,_=fixture()
        answers={r["key"]:("invented A","invented B") for s in c.SPLITS[:3] for r in rows[s]}
        allocation={(s,r["config"],r["category"],r["difficulty"]):1
                    for s in c.SPLITS[:3] for r in rows[s]}
        packets,key,strata=c.clinical_review_plan(rows,answers,allocation,b"x"*32)
        self.assertEqual(len(packets),12)
        self.assertTrue(all(set(x)=={"review_id","question","context","source","answer"} for x in packets))
        self.assertEqual(len({k for k,role in key.values()}),6)
        for k in {k for k,role in key.values()}:
            self.assertEqual({role for kk,role in key.values() if kk==k},{0,1})
        answers[rows["official_holdout"][0]["key"]]=("invented","invented")
        with self.assertRaisesRegex(ValueError,"CLINICAL_ANSWER_COVERAGE"):
            c.clinical_review_plan(rows,answers,allocation,b"x"*32)
        d=c.reviewer_summary({1},{1:"supported"},{1:"ambiguous"},{1:"ambiguous"},clinical=True)
        self.assertEqual(d["unresolved"],1)
        self.assertFalse(d["scientific_approval"])

if __name__=="__main__":unittest.main()
