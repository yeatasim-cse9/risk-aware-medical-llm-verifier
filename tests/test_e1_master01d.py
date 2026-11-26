"""Invented inputs only: statistical coverage, adjudication, signature gates."""
from collections import Counter
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import e1_master01c_core as c
import e1_master01d_review as d
import e1_master01d_gate as g
from test_e1_master01c_core import fixture
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

def judgment(label,impact=None):
    x={"label":label,"rationale":"invented reason","evidence":"invented evidence"}
    if impact is not None: x["impact"]=impact
    return x

def trust():
    keys={role:Ed25519PrivateKey.generate() for role in g.ROLES}
    registry={role:{"identity":"synthetic-"+role,"public_key_hex":
              k.public_key().public_bytes(Encoding.Raw,PublicFormat.Raw).hex()} for role,k in keys.items()}
    return keys,registry,c.digest(registry)

def subject():
    return g.subject(environment_digest="a"*64,session_digest="b"*64,analysis_digest="c"*64)

def receipts(keys,roles,s,phase,overrides=None):
    out={}
    for gate,role in roles.items():
        p={"phase":phase,"gate":gate,"role":role,"subject":s,"decision":"approve",
           "evidence_sha256":(overrides or {}).get(gate,"d"*64)}
        out[gate]={"payload":p,"signature":keys[role].sign(g.canonical(p)).hex()}
    return out

class TestMaster01D(unittest.TestCase):
    def test_exact_intervals_exhaustive_coverage(self):
        # Enumerate all samples for small finite populations.
        for N in range(1,9):
            for n in range(1,N+1):
                for K in range(N+1):
                    covered=total=0
                    for sample in combinations(range(N),n):
                        x=sum(i<K for i in sample)
                        lo,hi=d.interval(N,n,x)
                        covered+=int(lo<=K<=hi);total+=1
                    self.assertGreaterEqual(Fraction(covered,total),Fraction(19,20))
        self.assertEqual(d.interval(5,5,2),(2,2))
        self.assertEqual(d.interval(100,20,0)[0],0)
        self.assertEqual(d.interval(100,20,20)[1],100)
        self.assertGreater(d.interval(7177253,100,0)[1],0)
        with self.assertRaisesRegex(ValueError,"INTERVAL_INPUT"):
            d.interval(5,6,1)

    def test_weighted_census(self):
        x=d.stratified_binary([(2,2,1),(8,8,2)])
        self.assertEqual(x["estimate"],0.3)
        self.assertEqual(x["ci95"],[0.3,0.3])

    def test_unresolved_and_impact_disagreement(self):
        a={"x":judgment("supported","minor"),"y":judgment("ambiguous","uncertain")}
        b={"x":judgment("supported","potentially_major"),"y":judgment("ambiguous","uncertain")}
        with self.assertRaisesRegex(ValueError,"REVIEW_COVERAGE"):
            d.resolve(set(a),a,b,{},clinical=True)
        final,summary=d.resolve(set(a),a,b,b,clinical=True)
        self.assertEqual(summary["adjudication_required"],2)
        self.assertEqual(summary["unresolved"],1)
        self.assertEqual(final["x"]["impact"],"potentially_major")
        self.assertFalse(summary["scientific_approval"])

    def test_semantic_uncertainty_requires_remediation(self):
        queue=[];key={};a={}
        for left,right in c.PAIRS:
            for flagged,n in ((True,d.prior.EXPECTED_UNION[left+"__"+right]),(False,100)):
                for i in range(n):
                    token=str(len(queue))
                    pair=((left,int(token)),(right,int(token)))
                    queue.append({"pair":pair,"stratum":(left,right),"flagged":flagged})
                    key[token]=pair;a[token]=judgment("topic_only")
        result=d.semantic_analysis(queue,key,a,first=a,second=a)
        self.assertFalse(result["requires_missed_pair_remediation"])
        self.assertFalse(result["complete_semantic_leakage_excluded"])
        self.assertGreater(result["unflagged_strata"]["train__dev"]["possible_positive_ci95_upper"],0)
        token=next(k for k,pair in key.items() if not next(x for x in queue if x["pair"]==pair)["flagged"])
        a[token]=judgment("unresolved")
        self.assertTrue(d.semantic_analysis(queue,key,a,first=a,second=a)["requires_missed_pair_remediation"])

    def test_clinical_census_retained_only(self):
        rows,_=fixture()
        answers={r["key"]:("invented A","invented B") for s in c.SPLITS[:3] for r in rows[s]}
        allocation=Counter((s,r["config"],r["category"],r["difficulty"])
                           for s in c.SPLITS[:3] for r in rows[s])
        packets,key,strata=c.clinical_review_plan(rows,answers,allocation,b"x"*32)
        mapping=sorted([*r["key"],s] for s in c.SPLITS[:3] for r in rows[s])
        cust={"records":rows,"mapping":mapping,"clinical_key":key,"clinical_strata":strata}
        final={t:judgment("supported" if role==0 else "contradicted","none")
               for t,(k,role) in key.items()}
        result=d.clinical_analysis(cust,final,first=final,second=final)
        for s in c.SPLITS[:3]:
            self.assertEqual(result[s]["strict_role_concordance"]["estimate"],1)
            self.assertEqual(result[s]["confirmed_role_discordance"]["estimate"],0)
            self.assertEqual(result[s]["both_roles_label_agree"]["ci95"],[1,1])
        cust["mapping"]=mapping[:-1]
        with self.assertRaisesRegex(ValueError,"CLINICAL_POPULATION"):
            d.clinical_analysis(cust,final,first=final,second=final)

    def test_preaccess_signature_and_tampering(self):
        keys,registry,sha=trust();s=subject()
        r=receipts(keys,g.PRE_ACCESS,s,"pre_access")
        self.assertFalse(g.pre_access(r,s,registry,sha)["gpu_submitted"])
        r["dataset_rights"]["payload"]["evidence_sha256"]="e"*64
        with self.assertRaisesRegex(ValueError,"INVALID_APPROVAL_SIGNATURE"):
            g.pre_access(r,s,registry,sha)
        with self.assertRaisesRegex(ValueError,"TRUST_ROOT_MISMATCH"):
            g.pre_access(r,s,registry,"0"*64)

    def test_independent_panel(self):
        _,registry,_=trust()
        registry["clinical_reviewer_2"]=registry["clinical_reviewer_1"].copy()
        with self.assertRaisesRegex(ValueError,"ROLE_SEPARATION"):
            g.verify_registry(registry,c.digest(registry))

    def test_signed_review_bound_to_packets(self):
        keys,registry,sha=trust();s=subject();packet_digest="f"*64
        submissions={}
        for role in ("semantic_reviewer_1","semantic_reviewer_2","semantic_adjudicator"):
            data={} if role.endswith("adjudicator") else {"x":judgment("topic_only")}
            p={"phase":"review","role":role,"subject":s,"packet_sha256":packet_digest,
               "judgments_sha256":c.digest(data)}
            submissions[role]={"judgments":data,"signature":keys[role].sign(g.canonical(p)).hex()}
        final,summary=g.review_signed({"x"},submissions,registry,sha,s,packet_digest=packet_digest)
        self.assertEqual(summary["agreement"],1)
        self.assertEqual(final["x"]["label"],"topic_only")
        with self.assertRaisesRegex(ValueError,"INVALID_APPROVAL_SIGNATURE"):
            g.review_signed({"x"},submissions,registry,sha,s,packet_digest="e"*64)

    def test_manual_freeze_even_with_signed_receipts(self):
        keys,registry,sha=trust()
        mapping=[["synthetic",0,"train"],["synthetic",1,"dev"],["synthetic",2,"test"]]
        mapping_hash=c.digest({"seed":c.SEED,"assignment":mapping})
        sem={"requires_missed_pair_remediation":False,"complete_semantic_leakage_excluded":False}
        clin={"invented":"summary"};ss={"n":771};cs={"n":6}
        with patch.object(d,"MAPPING",mapping_hash),patch.object(d,"COUNTS",
                {"train":1,"dev":1,"test":1,"official_holdout":500}):
            s=subject()
            r=receipts(keys,g.FREEZE,s,"freeze",{
                "semantic_disposition":c.digest([sem,ss]),
                "clinical_label_review":c.digest([clin,cs]),
                "private_manifest_review":mapping_hash,
                "locked_analysis_protocol":s["analysis_protocol"]})
            result=g.freeze_candidate(mapping,r,s,registry,sha,semantic_result=sem,
                     clinical_result=clin,semantic_review_summary=ss,clinical_review_summary=cs)
            self.assertFalse(result["final_split_frozen"])
            self.assertFalse(result["scientific_e1_experiment_executed"])
            sem["requires_missed_pair_remediation"]=True
            with self.assertRaisesRegex(ValueError,"MISSED_PAIR_REMEDIATION_REQUIRED"):
                g.freeze_candidate(mapping,r,s,registry,sha,semantic_result=sem,
                     clinical_result=clin,semantic_review_summary=ss,clinical_review_summary=cs)

if __name__=="__main__":unittest.main()
