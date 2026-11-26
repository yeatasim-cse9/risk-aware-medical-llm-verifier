"""Synthetic wrapper subprocess tests; no actual Kaggle, GPU, model or dataset."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(SCRIPTS))
import e1_master01e_v3_recovery_workflow as w
import e1_master01e_v3_recovery_diagnostics as d

class RecoveryController(unittest.TestCase):
    def fixture(self,root,fail=False,authorized=True,tamper=False):
        work=root/"work";work.mkdir()
        scratch=root/"scratch";scratch.mkdir()
        runner=("import os,sys\n"
                "def run(root,report,manifest,md,recorder):\n"
                "    recorder.mark('AUTHORIZATION')\n"
                "    print('PRIVATE_FAKE_STDOUT',flush=True)\n"
                "    os.write(2,b'PRIVATE_FAKE_STDERR')\n")
        if fail:runner+="    raise RuntimeError('PRIVATE_FAKE_EXCEPTION')\n"
        else:
            for stage in d.STAGES[2:-3]:runner+="    recorder.mark("+repr(stage)+")\n"
            runner+="    return {'invented': 1}\n"
        bodies={"e1_master01e_v3_recovery_private":runner.encode(),
                "e1_master01e_v3_recovery_diagnostics":
                    (SCRIPTS/"e1_master01e_v3_recovery_diagnostics.py").read_bytes(),
                "e1_master01e_v3_validate":b"def validate(*args): pass\n",
                "e1_master01c_core":b"import hashlib,json\ndef digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()\n"}
        hashes={k:hashlib.sha256(v).hexdigest() for k,v in bodies.items() if k not in w.EXTRA}
        recovery={"scripts/"+k+".py":hashlib.sha256(bodies[k]).hexdigest() for k in w.EXTRA}
        if tamper:bodies[w.EXTRA[1]]+=b"#tampered"
        script=root/"runner.py"
        script.write_text(w.render("a"*40,bodies,hashes,{},{},"b"*64,recovery,authorized,
                                    work=str(work),scratch=str(scratch)))
        result=subprocess.run([sys.executable,str(script)],capture_output=True,text=True,timeout=15)
        return result,work/w.REPORT

    def test_wrapper_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,report=self.fixture(Path(tmp))
            self.assertEqual(p.returncode,0)
            self.assertEqual(p.stderr,"")
            self.assertEqual(p.stdout.splitlines(),["MASTER01EV3R_STAGE="+s for s in d.STAGES])
            self.assertNotIn("PRIVATE_FAKE",p.stdout)
            self.assertEqual(json.loads(report.read_text())["aggregate"]["invented"],1)

    def test_wrapper_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,report=self.fixture(Path(tmp),fail=True)
            self.assertNotEqual(p.returncode,0)
            self.assertEqual(p.stderr,"")
            self.assertEqual(p.stdout.splitlines()[-3:],[
                "MASTER01EV3R_FAILED_STAGE=AUTHORIZATION",
                "MASTER01EV3R_FAILURE_KIND=RUNTIME_ERROR",
                "MASTER01EV3R_FAILURE_CODE=UNCLASSIFIED"])
            self.assertNotIn("PRIVATE_FAKE",p.stdout)
            self.assertFalse(report.exists())

    def test_wrapper_unauthorized_and_tampered(self):
        for kwargs in ({"authorized":False},{"tamper":True}):
            with tempfile.TemporaryDirectory() as tmp:
                p,report=self.fixture(Path(tmp),**kwargs)
                self.assertNotEqual(p.returncode,0)
                self.assertEqual(p.stderr,"")
                self.assertFalse(report.exists())
                self.assertNotIn("PRIVATE_FAKE",p.stdout)

    def test_authorization_fail_closed(self):
        env={"MASTER01EV3R_RIGHTS":"true","MASTER01EV3R_MODEL_RIGHTS":"true",
             "MASTER01EV3R_GPU":"true","GITHUB_RUN_ATTEMPT":"1",
             "GITHUB_EVENT_NAME":"workflow_dispatch","GITHUB_REF":"refs/heads/main",
             "GITHUB_REPOSITORY":"yeatasim-cse9/risk-aware-medical-llm-verifier"}
        with patch.dict(os.environ,env,clear=True):self.assertTrue(w.authorization())
        for k in env:
            with patch.dict(os.environ,{**env,k:"invalid"},clear=True):
                self.assertFalse(w.authorization())

    def test_preflight_preserves_failed_run_and_blocks_duplicate(self):
        self.assertNotEqual(w.KERNEL,w.FAILED)
        for states,ok in ((["complete","error","missing"],True),
                          (["complete","error","error"],False),
                          (["complete","missing","missing"],False)):
            with patch.object(w,"source"),patch.object(w.subprocess,"run") as run,patch.object(w,"check",side_effect=states):
                run.return_value.returncode=0
                if ok:w.preflight()
                else:
                    with self.assertRaises(ValueError):w.preflight()
                self.assertEqual(run.call_args.args[0],["kaggle","quota"])

    def test_envelope_binding(self):
        with patch.object(w.validator,"validate"):
            args=("a"*40,{"base":"b"*64},{"recovery":"c"*64},"d"*64)
            e=w.envelope({"synthetic":True},*args)
            self.assertEqual(w.validate_envelope(e,*args),e)
            for k,value in (("recovery_source_commit","f"*40),("recovery_file_sha256",{}),("schema_version",True)):
                with self.assertRaises(ValueError):w.validate_envelope({**e,k:value},*args)
            with self.assertRaises(ValueError):
                w.validate_envelope({**e,"extra":"PRIVATE_FAKE"},*args)

if __name__=="__main__":unittest.main()
