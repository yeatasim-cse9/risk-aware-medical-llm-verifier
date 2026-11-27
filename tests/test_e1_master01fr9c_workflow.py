import sys
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import e1_master01fr9c_workflow as w
RAW='MODE = "PREFLIGHT"\nFRESH_GPU_AUTHORIZATION = False\nSUPERVISOR_PROTOCOL_REVIEW_CONFIRMED = False\nRIGHTS_AND_QUOTA_RECONFIRMED = False\n'
class Contract(unittest.TestCase):
    def test_flag_only(self):
        with patch.object(w,"SOURCE_SHA",w.sha(RAW.encode())):
            result=w.render(RAW,"gpu")
            self.assertEqual(result,RAW.replace('"PREFLIGHT"','"EXECUTE"').replace("False","True"))
            self.assertEqual(w.render(RAW,"cpu"),RAW)
    def test_drift(self):
        with self.assertRaisesRegex(ValueError,"R9C_SOURCE_DRIFT"):w.render(RAW,"gpu")
    def test_default_no_auth(self):
        with patch.dict(w.os.environ,{},clear=True):
            self.assertFalse(w.authorized("GPU"));self.assertFalse(w.authorized("CPU"))
    def test_attempt_gate(self):
        env={"GITHUB_EVENT_NAME":"workflow_dispatch","GITHUB_REPOSITORY":w.REPO,"GITHUB_REF":"refs/heads/main","GITHUB_RUN_ATTEMPT":"2","R9C_GPU":"true"}
        with patch.dict(w.os.environ,env,clear=True):self.assertFalse(w.authorized("GPU"))
    def test_cpu_package(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(w,"SOURCE_SHA",w.sha(RAW.encode())):
            p=Path(tmp)/"p";w.package(p,RAW,"cpu")
            self.assertFalse(json.loads((p/"kernel-metadata.json").read_text())["enable_gpu"])
            self.assertEqual(w.read_notebook(p)[1:],(RAW,True))
    def test_gpu_package(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(w,"SOURCE_SHA",w.sha(RAW.encode())):
            p=Path(tmp)/"p";w.package(p,RAW,"gpu")
            m=json.loads((p/"kernel-metadata.json").read_text())
            self.assertTrue(m["enable_gpu"] and m["is_private"])
            self.assertEqual(m["machine_shape"],"NvidiaTeslaT4")
            self.assertEqual(len(json.loads((p/"runner.ipynb").read_text())["cells"]),1)
    def test_existing_plan_stops(self):
        class Store:
            def read(self,p):return b"existing"
        with self.assertRaisesRegex(ValueError,"R9C_RUN_EXISTS_NO_RETRY"):w.fresh(Store())
    def test_qualifier_compiles(self):
        compile(w.QUAL_CODE,"qual","exec")

    def test_monitor_stale_complete_does_not_approve(self):
        class Store:
            def read(self,p):return None
        with patch.object(w,"check",return_value="complete"),patch.object(w.time,"monotonic",side_effect=[0,0,0,301]),patch.object(w.time,"sleep"),patch.object(w,"verify") as verify:
            with self.assertRaisesRegex(ValueError,"R9C_COMPLETE_WITHOUT_NEW_EVIDENCE"):
                w.monitor(Store(),"gpu")
            verify.assert_not_called()

    def test_source_normalization_preserves_pin(self):
        with patch.object(w,"SOURCE_SHA",w.sha(RAW.encode())):
            for candidate in (RAW,RAW.rstrip("\n"),RAW.replace("\n","\r\n")):
                self.assertEqual(w.canonical_source(candidate,w.SOURCE_SHA),RAW)
            with self.assertRaisesRegex(ValueError,"R9C_SOURCE_DRIFT"):
                w.canonical_source(RAW+"# changed",w.SOURCE_SHA)

    def test_live_pull_failure_reports_exact_stage(self):
        from io import StringIO
        from contextlib import redirect_stdout
        import subprocess
        with patch.object(w,"check",return_value="complete"),patch.object(w.subprocess,"run",return_value=subprocess.CompletedProcess([],2,b"",b"unrecognized arguments SECRET")),redirect_stdout(StringIO()) as out:
            try:w.pull(Path("."))
            except Exception as exc:w.report_failure(exc)
        result=out.getvalue()
        self.assertIn("R9C_FAILED_STAGE=KAGGLE_PULL",result)
        self.assertIn("R9C_CLI_ARGUMENT_REJECTION=TRUE",result)
        self.assertNotIn("SECRET",result)

    def test_live_metadata_failure_reports_exact_policy(self):
        from io import StringIO
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as tmp,patch.object(w,"SOURCE_SHA",w.sha(RAW.encode())),redirect_stdout(StringIO()) as out:
            p=Path(tmp)/"p";w.package(p,RAW,"cpu")
            f=p/"kernel-metadata.json";m=json.loads(f.read_text());m["is_private"]=False;f.write_text(json.dumps(m))
            try:w.read_notebook(p)
            except Exception as exc:w.report_failure(exc)
        self.assertIn("R9C_FAILURE_CODE=R9C_PRIVATE_NOTEBOOK_REQUIRED",out.getvalue())

    def test_remote_exception_not_leaked(self):
        from io import StringIO
        from contextlib import redirect_stdout
        with redirect_stdout(StringIO()) as out:w.report_failure(ValueError("R9C_PRIVATE_SECRET_IN_RAW_MESSAGE"))
        self.assertIn("R9C_UNCLASSIFIED_REQUIRES_EVIDENCE",out.getvalue())
        self.assertNotIn("PRIVATE_SECRET_IN_RAW_MESSAGE",out.getvalue())
