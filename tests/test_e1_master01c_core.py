import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import e1_master01c_core as c

def norm(x): return " ".join(x.lower().split())

def fixture():
    rows={}
    for s in c.SPLITS:
        cfg="official_holdout" if s=="official_holdout" else "synthetic"
        offset=c.SPLITS.index(s)*10
        rows[s]=[dict(key=(cfg,offset+i),source=f"source-{s}-{i}",
                     question=f"invented question {s} {i}",context=f"invented context {s} {i}",
                     config=cfg,category=f"cat{i%2}",difficulty="easy") for i in range(4)]
    flags={p:set() for p in c.PAIRS}
    flags["train","dev"]={(0,0)}
    flags["test","official_holdout"]={(0,0)}
    return rows,flags

class TestMaster01C(unittest.TestCase):
    def test_closure_and_holdout(self):
        rows,flags=fixture()
        rows["train"][1]["source"]=rows["train"][0]["source"]
        rows["train"][2]["context"]=rows["train"][1]["context"]
        rows["dev"][1]["source"]=rows["official_holdout"][1]["source"]
        keep,out=c.quarantine(rows,flags,norm)
        self.assertEqual([len(keep[s]) for s in c.SPLITS],[1,2,3,4])
        self.assertEqual(out["residual_flagged_pairs"],0)
        self.assertEqual(out["counts"]["train"]["paired_examples"],2)
        self.assertEqual(keep["official_holdout"],rows["official_holdout"])
        self.assertEqual(c.quarantine(rows,flags,norm),(keep,out))
        m=sorted([*r["key"],s] for s in c.SPLITS[:3] for r in keep[s])
        self.assertFalse(c.freeze_preparation(m,out["mapping_sha256"],
                         dict.fromkeys(c.GATES,True))["final_split_frozen"])
        with self.assertRaisesRegex(ValueError,"MANIFEST_DIGEST"):
            c.freeze_preparation(m,"0"*64,dict.fromkeys(c.GATES,True))
    def test_review_sample(self):
        rows,flags=fixture()
        q,st=c.review_plan(rows,flags,norm,3)
        self.assertEqual(len(q),20)
        self.assertEqual(len({p["pair"] for p in q}),20)
        self.assertEqual(sum(p["flagged"] for p in q),2)
        self.assertEqual((q,st),c.review_plan(rows,flags,norm,3))
        self.assertEqual(st["train__dev"]["unflagged_population"],15)
    def test_full_complement(self):
        rows,flags=fixture()
        q,_=c.review_plan(rows,flags,norm)
        self.assertEqual(len(q),96)
        self.assertEqual(len({p["pair"] for p in q}),96)
    def test_missing_and_conflicts(self):
        rows,flags=fixture()
        rows["train"][0]["source"]=""
        with self.assertRaisesRegex(ValueError,"MISSING_SOURCE"):
            c.quarantine(rows,flags,norm)
        rows,flags=fixture()
        rows["train"][3]["source"]=rows["dev"][3]["source"]
        with self.assertRaisesRegex(ValueError,"RESIDUAL_SOURCE"):
            c.quarantine(rows,flags,norm)
        flags["train","dev"].add((4,0))
        with self.assertRaisesRegex(ValueError,"PAIR_INDEX"):
            c.review_plan(rows,flags,norm)
    def test_review_validation(self):
        first={1:"same_source",2:"unresolved"}
        second={1:"topic_only",2:"unresolved"}
        with self.assertRaisesRegex(ValueError,"ADJUDICATION_COVERAGE"):
            c.reviewer_summary({1,2},first,second,{1:"same_source"})
        d=c.reviewer_summary({1,2},first,second,{1:"same_source",2:"unresolved"})
        self.assertEqual(d["unresolved"],1)
        self.assertEqual(d["agreement"],0.5)
        self.assertFalse(d["scientific_approval"])
    def test_balance(self):
        rows,flags=fixture()
        keep,_=c.quarantine(rows,flags,norm)
        b=c.balance(rows,keep)
        self.assertEqual(b["train"]["category"]["before_sorted"],[2,2])
        self.assertAlmostEqual(b["train"]["category"]["total_variation"],1/6,places=7)
    def test_reject_holdout_manifest(self):
        with self.assertRaisesRegex(ValueError,"MANIFEST_ROW"):
            c.freeze_preparation([["official_holdout",0,"train"]],"0"*64,{})

if __name__=="__main__": unittest.main()
