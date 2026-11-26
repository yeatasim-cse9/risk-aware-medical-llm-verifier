import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import e1_master01f_r3_audit_v1 as a


def rows():
    return [{"id":str(i),"inner_id":str(i),"source":"sumpubmed",
             "context":"invented document "+str(i),"statement":"toy sentence "+str(i),
             "label":bool(i%2),"synthetic":bool((i//2)%2)} for i in range(100)]


class Audit(unittest.TestCase):
    def test_tied_ap(self):
        self.assertEqual(a.ap([1,0,1,0],[0,0,0,0]),.5)
        self.assertEqual(a.ap([1,0],[2,1]),1)

    def test_determinism_and_no_approval(self):
        r=rows()
        x=a.audit(r,20,20)
        self.assertEqual(x,a.audit(list(reversed(r)),20,20))
        self.assertEqual(x["proxy_groups"],100)
        self.assertFalse(x["execution_authorized"])
        self.assertFalse(x["article_independence_verified"])

    def test_source_id_and_context_components(self):
        r=rows()
        r[1]["inner_id"]=r[0]["inner_id"]
        r[2]["context"]=r[1]["context"]
        x=a.audit(r,20,20)
        self.assertEqual(x["proxy_groups"],98)

    def test_exclusion(self):
        r=rows()
        r[0]["source"]="acm"
        r[1]["context"]=""
        self.assertEqual(a.audit(r,20,20)["eligible_rows"],98)

    def test_no_explanation_feature(self):
        r=rows()
        r[0]["explanation"]="toy"
        with self.assertRaisesRegex(ValueError,"ROW_SCHEMA"):
            a.audit(r,20,20)

    def test_bad_label(self):
        r=rows()
        r[0]["label"]=1
        with self.assertRaisesRegex(ValueError,"ROW_ID_SOURCE_LABEL"):
            a.audit(r,20,20)

    def test_authorization_before_mount(self):
        with patch.object(a.native,"approved_plan") as plan:
            with self.assertRaisesRegex(ValueError,"AUDIT_AUTHORIZATION_REQUIRED"):
                next(a.verified_rows(Path("/nonexistent"),None,None,None,allow=False,label_semantics_confirmed=True))
            plan.assert_not_called()

    def test_protected_mount_rejected_before_hash(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"test.parquet").touch()
            with patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":"","HF_HUB_OFFLINE":"1"}), patch.object(a.native,"approved_plan",return_value={"train_files":[{"path":"data/train.parquet"}]}),patch.object(a.native,"file_identity") as h:
                with self.assertRaisesRegex(ValueError,"TRAIN_INVENTORY"):
                    next(a.verified_rows(root,None,None,None,allow=True,label_semantics_confirmed=True))
                h.assert_not_called()


if __name__=="__main__":
    unittest.main()
