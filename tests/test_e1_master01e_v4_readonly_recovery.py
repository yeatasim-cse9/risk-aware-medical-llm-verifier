"""Regression: actual old writer output, strict inner validator, immutable recovery."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import e1_master01e_v4_readonly_recovery as rec
import e1_master01e_v4_workflow as w
from test_e1_master01e_v3 import report_fixture,seal
from test_e1_master01e_v4_controller import RecoveryController

def fixture():
    agg,commit,hashes,md=report_fixture()
    files={"scripts/fixture.py":"e"*64}
    value={"schema_version":1,"kind":"MASTER01EV4_PRIVATE_V1",
           "v4_source_commit":commit,"v4_file_sha256":files,"aggregate":agg}
    value["integrity_sha256"]=w.core.digest(value)
    return value,commit,hashes,files,md

def encode(value):return json.dumps(value,sort_keys=True).encode()

def reseal(value):
    value.pop("integrity_sha256",None)
    value["integrity_sha256"]=w.core.digest(value)
    return value

class ReadOnly(unittest.TestCase):
    def test_actual_original_writer_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,path=RecoveryController().fixture(Path(tmp))
            self.assertEqual(p.returncode,0)
            raw=path.read_bytes()
            v=json.loads(raw)
            self.assertEqual(set(v),rec.KEYS)
            agg=v["aggregate"]
            args=(v["v4_source_commit"],agg["package_hashes"],v["v4_file_sha256"],"b"*64)
            with patch.object(w.validator,"validate") as validate:
                self.assertEqual(rec.validate_original(raw,w,*args),v)
                validate.assert_called_once_with(agg,args[0],args[1],args[3])
                with self.assertRaisesRegex(ValueError,"RECOVERY_SCHEMA"):
                    w.validate_envelope(v,*args)

    def test_real_inner_validator_and_tamper(self):
        v,*args=fixture()
        self.assertEqual(rec.validate_original(encode(v),w,*args),v)
        for k,x in (("schema_version",True),("v4_source_commit","a"*40),
                    ("v4_file_sha256",{}),("input_budget_audit_passed",True)):
            z=copy.deepcopy(v);z[k]=x
            with self.assertRaises(ValueError):
                rec.validate_original(encode(reseal(z)),w,*args)
        for k,x in (("test_used",True),("private_rows",[]),("peak_reserved_bytes",0)):
            z=copy.deepcopy(v);z["aggregate"][k]=x;seal(z["aggregate"])
            with self.assertRaises(ValueError):
                rec.validate_original(encode(reseal(z)),w,*args)
        z=copy.deepcopy(v);z["integrity_sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"ORIGINAL_WRITER_INTEGRITY"):
            rec.validate_original(encode(z),w,*args)

    def test_duplicate_and_nonfinite_json(self):
        v,*args=fixture()
        for raw in (b'{"schema_version":1,"schema_version":1}',b'{"x":NaN}'):
            with self.assertRaises(ValueError):rec.validate_original(raw,w,*args)

    def test_read_only_preserves_original_bytes(self):
        v,_,hashes,files,md=fixture()
        v["v4_source_commit"]=rec.COMMIT
        v["aggregate"]["package_source_commit"]=rec.COMMIT
        seal(v["aggregate"]);reseal(v)
        raw=json.dumps(v,indent=3).encode()+b"\n"
        fake=SimpleNamespace(source=lambda c:(c,{},hashes,{}, {},md,files),
                             check=lambda k,s:"complete",core=w.core,validator=w.validator)
        def download(cmd,**kw):
            self.assertEqual(cmd[:4],["kaggle","kernels","output",rec.KERNEL])
            self.assertEqual(cmd[6:],["--file-pattern","^master01ev4_private_aggregate[.]json$"])
            (Path(cmd[5])/rec.REPORT).write_bytes(raw)
            return SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory() as tmp,patch.object(rec.subprocess,"run",side_effect=download) as call:
            out=Path(tmp)/"verified"
            rec.recover(fake,out)
            self.assertEqual(call.call_count,1)
            self.assertEqual((out/rec.REPORT).read_bytes(),raw)
            receipt=json.loads((out/"verification_receipt.json").read_text())
            self.assertEqual(receipt["original_file_sha256"],hashlib.sha256(raw).hexdigest())
            self.assertFalse(receipt["scientific_approval"])
            self.assertNotIn("input_budget_audit_passed",json.loads(raw))

    def test_noncomplete_never_downloads(self):
        fake=SimpleNamespace(source=lambda c:(c,{}, {},{}, {},None,{}),check=lambda *args:"running")
        with patch.object(rec.subprocess,"run") as run:
            with self.assertRaisesRegex(ValueError,"REMOTE_NOT_COMPLETE"):
                rec.recover(fake,Path("unused"))
            run.assert_not_called()

if __name__=="__main__":unittest.main()
