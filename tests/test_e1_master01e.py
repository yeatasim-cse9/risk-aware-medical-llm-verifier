import sys,json,hashlib,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import e1_master01e_bioasq as bio
import e1_master01e_baselines as base

def pair(i,part="train",group=None):
    return {"parent":str(i),"source_group":group or str(i),"partition":part,
        "question":"invented query "+str(i),"context":"alpha beta",
        "answers":{"ground_truth":"alpha beta longer answer","hallucinated":"wrong"}}
def doc(i,pmid):
    u="https://pubmed.ncbi.nlm.nih.gov/"+str(pmid)+"/"
    return {"id":str(i),"type":"yesno","body":"invented question "+str(i),
        "documents":[u],"snippets":[{"document":u,"text":"invented context "+str(i)}],
        "exact_answer":"yes","ideal_answer":"never passed to features"}
def raw(qs):
    b=json.dumps({"questions":qs}).encode();return b,hashlib.sha256(b).hexdigest()
def protected():
    return [dict(dataset="prior01c",release="prior",id="private",task="protected",
        native_split="protected",sources=("pmid:11",),question="other question",
        context="other context",evidence_available=True,reference_available=False,provenance_known=True)]

class TestMaster01E(unittest.TestCase):
    def test_release_authorization_and_hash(self):
        b,h=raw([doc(1,11)])
        with self.assertRaisesRegex(ValueError,"PRIVATE_AUTHORIZATION"):
            bio.decode(b,release_sha256=h,expected_count=1,split="development",authorized=False)
        with self.assertRaisesRegex(ValueError,"RELEASE_PIN"):
            bio.decode(b,release_sha256="0"*64,expected_count=1,split="development",authorized=True)
        x,c=bio.decode(b,release_sha256=h,expected_count=1,split="development",authorized=True)
        self.assertEqual(c["yesno"],1);self.assertNotIn("exact_answer",x[0])
        self.assertNotIn("ideal_answer",x[0])
    def test_overlap_and_official_test(self):
        b,h=raw([doc(1,11),doc(2,22),doc(3,33)])
        t,j=raw([doc(4,22)])
        a=bio.inventory(b,dev_sha=h,dev_count=3,protected=protected(),authorized=True,
            test_raw=t,test_sha=j,test_count=1)
        self.assertEqual(a["inventory"]["excluded_rows"]["prior_protected_overlap"],1)
        self.assertEqual(a["inventory"]["excluded_rows"]["official_test_overlap"],1)
        self.assertEqual(a["inventory"]["eligible_group_representatives"]["test"],1)
        self.assertFalse(a["execution_authorized"])
        self.assertNotIn("candidate_manifest_sha256",a["inventory"])
    def test_url_and_duplicate(self):
        self.assertEqual(bio.pmid("http://www.ncbi.nlm.nih.gov/pubmed/123"),"pmid:123")
        for u in ("https://example.org/pubmed/123","https://pubmed.ncbi.nlm.nih.gov/123/?x=1"):
            with self.assertRaises(ValueError):bio.pmid(u)
        b,h=raw([doc(1,11),doc(1,22)])
        with self.assertRaisesRegex(ValueError,"QUESTION_SCHEMA"):
            bio.decode(b,release_sha256=h,expected_count=2,split="development",authorized=True)
    def test_train_only_and_question_null(self):
        train=[pair(i) for i in range(4)]
        dev=[pair(i,"dev") for i in range(4,8)]
        base.disjoint(train,dev)
        for mode in base.MODES:
            m=base.fit(train,mode,steps=10)
            scored=base.score_pairs(m,dev,"dev")
            self.assertEqual(len(scored),4)
            if mode=="question_only":self.assertEqual(base.metrics(scored)["paired_ranking"],.5)
            if mode=="length":self.assertEqual(base.metrics(scored)["paired_ranking"],1.)
        with self.assertRaisesRegex(ValueError,"PAIR_SCHEMA"):base.fit(dev,"length")
        with self.assertRaisesRegex(ValueError,"CROSS_PARTITION"):base.disjoint(train,[pair(9,"dev","1")])
    def test_frozen_context_control(self):
        a={"question":"alpha query","answer":"alpha beta","context":"alpha beta"}
        self.assertNotEqual(base.features(a,"evidence_lexical"),base.features({**a,"context":""},"evidence_lexical"))
        self.assertEqual(base.features(a,"answer_only"),base.features({**a,"context":""},"answer_only"))
        dev=[pair(i,"dev") for i in range(4)]
        for i,p in enumerate(dev):p["context"]="invented context "+str(i)
        sh=base.context_permutation(dev,"dev")
        by={p["parent"]:p["context"] for p in dev}
        self.assertTrue(all(by[p["parent"]]!=p["context"] for p in sh))
        self.assertEqual(sorted(by.values()),sorted(p["context"] for p in sh))
    def test_metrics_and_alignment(self):
        good=[("g1","a",(0.,1.)),("g2","b",(0.,1.))]
        bad=[("g1","a",(.5,.5)),("g2","b",(.5,.5))]
        self.assertEqual(base.metrics(good)["average_precision"],1.)
        self.assertEqual(base.metrics(bad)["average_precision"],.5)
        self.assertEqual(base.paired_bootstrap(good,bad,replicates=100)["percentile_95"],[.5,.5])
        with self.assertRaisesRegex(ValueError,"PARENT_GROUP_ALIGNMENT"):
            base.paired_bootstrap(good,[("g1","wrong",(.5,.5)),bad[1]],replicates=100)
if __name__=="__main__":unittest.main()
