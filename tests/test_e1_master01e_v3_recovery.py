"""Synthetic-only recovery regression: no GPU, network, models or medical rows."""
import ast
from pathlib import Path
import sys
import unittest
SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(SCRIPTS))
import e1_master01e_v3_recovery_diagnostics as d

class RecoveryDiagnostics(unittest.TestCase):
    def test_unknown_exception_never_leaks(self):
        out=[];r=d.Recorder(out.append)
        r.mark("AUTHORIZATION")
        r.failure(RuntimeError("PRIVATE_EXAMPLE_DO_NOT_EXPORT"))
        self.assertEqual(out,["MASTER01EV3R_STAGE=BOOTSTRAP",
            "MASTER01EV3R_STAGE=AUTHORIZATION",
            "MASTER01EV3R_FAILED_STAGE=AUTHORIZATION",
            "MASTER01EV3R_FAILURE_KIND=RUNTIME_ERROR",
            "MASTER01EV3R_FAILURE_CODE=UNCLASSIFIED"])
    def test_allowlist_only(self):
        self.assertEqual(d.classify(ValueError("GPU_REQUIRED")),("VALUE_ERROR","GPU_REQUIRED"))
        self.assertEqual(d.classify(ValueError("PRIVATE_EXAMPLE")),("VALUE_ERROR","UNCLASSIFIED"))
        self.assertEqual(d.classify(ModuleNotFoundError("PRIVATE_EXAMPLE")),("MODULE_MISSING","UNCLASSIFIED"))
        class Untrusted(ValueError):
            def __str__(self): raise AssertionError("must not stringify")
        self.assertEqual(d.classify(Untrusted("GPU_REQUIRED")),("OTHER_EXCEPTION","UNCLASSIFIED"))
    def test_order_and_terminal(self):
        out=[];r=d.Recorder(out.append)
        with self.assertRaises(ValueError):r.mark("PRIVATE_EXAMPLE")
        for stage in d.STAGES[1:]:r.mark(stage)
        self.assertEqual(out,["MASTER01EV3R_STAGE="+s for s in d.STAGES])
        with self.assertRaises(ValueError):r.failure(ValueError("GPU_REQUIRED"))
    def test_failure_terminal(self):
        out=[];r=d.Recorder(out.append);r.failure(TypeError("private"))
        with self.assertRaises(ValueError):r.mark("AUTHORIZATION")
        with self.assertRaises(ValueError):r.failure(ValueError("GPU_REQUIRED"))
    def test_original_algorithm_unchanged(self):
        orig=ast.parse((SCRIPTS/"e1_master01e_v3_private.py").read_text())
        recovery=ast.parse((SCRIPTS/"e1_master01e_v3_recovery_private.py").read_text())
        old=next(n for n in orig.body if isinstance(n,ast.FunctionDef) and n.name=="run")
        new=next(n for n in recovery.body if isinstance(n,ast.FunctionDef) and n.name=="run")
        self.assertEqual(new.args.args.pop().arg,"recorder")
        gate=new.body.pop(0)
        self.assertIn("MASTER01EV3_RECOVERY_AUTHORIZED_ONE_PILOT",ast.unparse(gate))
        markers=[];body=[]
        for n in new.body:
            if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and ast.unparse(n.value.func)=="recorder.mark":
                markers.append(n.value.args[0].value)
            else:body.append(n)
        new.body=body
        self.assertEqual(tuple(markers),d.STAGES[1:-3])
        self.assertEqual(ast.dump(old),ast.dump(new))

if __name__=="__main__":unittest.main()
