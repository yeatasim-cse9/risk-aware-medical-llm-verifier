"""Verify archived public aggregate against original source and canonical hash."""
import hashlib,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import e1_master01d_v3_validate as v
import e1_master01d_v2_integrity as d

class VerifiedMaster01DV3(unittest.TestCase):
    def test_archived_actual_aggregate(self):
        a=json.loads((ROOT/"configs/master01dv3-verified-aggregate-2026-11-26.json").read_text())
        self.assertEqual(a["integrity_sha256"],"a8e159311f8829a5b5ece7e12e0ffd3a76812f20dd3f066c9d508bf23304ace8")
        hashes={k:hashlib.sha256((ROOT/"scripts"/(k+".py")).read_bytes()).hexdigest() for k in v.MODULES}
        # This regression test deliberately pins the archived audit source.
        # If modules change in future, retrieve the original source commit instead.
        v.validate(a,"c63169fc6e597b80d48fbcdaa01102c97abcddc8",hashes)
        old=json.loads((ROOT/"configs/master01c-verified-aggregate-2026-11-26.json").read_text())
        d.verify_checkpoint(old)
        self.assertEqual(a["maxima"],old["maxima"])
        self.assertEqual(a["versions"],old["versions"])
        self.assertEqual(a["pubmedqa_inventory"]["excluded_rows"]["prior_protected_overlap"],500)
        self.assertEqual(a["pubmedqa_inventory"]["eligible_group_representatives"]["calibration"],0)
        self.assertFalse(a["e1_execution_authorized"])

if __name__=="__main__":unittest.main()
