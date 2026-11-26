"""Invented nonmedical inputs; no real model, dataset, network or GPU."""
import json,tempfile,time,unittest,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import e1_master01f_d_inputs_v1 as i
import e1_master01f_d_pilot_v1 as p
import e1_master01f_d_controller_v1 as c
from test_e1_master01e_v4 import ModernTokenizer,Model

def encoded(x):return json.dumps(x,sort_keys=True).encode()
def h(x):return i.sha(x.encode())
def fixture(root):
    meta={};files={};rows={}
    for k,part in i.ORIGINAL.items():
        meta[k]=[{"parent":h(k+str(j)),"source_group":h(k+"group"+str(j)),"partition":part}
                 for j in range(i.COUNTS[k])]
        for a,g,_ in i.select(meta[k],k):
            row={"parent":a,"source_group":g,"partition":k,"question":"Which shape",
                 "context":"blue circle and green triangle","answers":{"ground_truth":"blue circle","hallucinated":"red square"}}
            raw=encoded(row);name=a+".json";(root/name).write_bytes(raw);files[name]=i.sha(raw);rows[name]=row
    m={"schema":1,"mapping_sha256":i.hist.MAPPING,"checkpoint_sha256":i.hist.CHECKPOINT,
       "exposure":i.EXPOSURE,"metadata":meta,"selected_files":files}
    return seal(root,m),m,rows

def seal(root,m):
    raw=encoded(m);(root/"manifest.json").write_bytes(raw);return i.sha(raw)

class Engine(p.n.Engine):
    def __init__(self):
        model=Model();model.config.vocab_size=32
        super().__init__(ModernTokenizer(),model,"cpu");self.calls=0
    def infer(self,parts,hypothesis,*,removed=False,batch_size=4):
        self.calls+=1
        slots=[] if removed else parts
        a=sum(sum(self.encode(x,self.tokens(hypothesis))["attention_mask"])
              for x in slots+[[] for _ in range(p.n.SLOTS-len(slots))])
        return [.5]*6,dict(forward_examples=16,padded_positions=8192,active_tokens=a,real_windows=len(slots))

class Projected(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        for k,v in (("LIMITS",{"fit":3,"selection":2}),("COUNTS",{"fit":5,"selection":4})):
            q=patch.object(i,k,v);q.start();self.addCleanup(q.stop)
        self.pin,self.m,self.rows=fixture(self.root)
    def test_preflight_never_reads_answers(self):
        with patch.object(i,"read_pinned",wraps=i.read_pinned) as read:
            i.preflight(self.root,self.pin)
            self.assertEqual([x.args[1] for x in read.call_args_list],["manifest.json"])
        with patch.object(i,"read_pinned") as read:
            with self.assertRaises(ValueError):i.load_selected(self.root,self.pin,authorized=False)
            read.assert_not_called()
    def test_bad_pin_extra_file_and_test_metadata(self):
        with self.assertRaises(ValueError):i.preflight(self.root,"0"*64)
        extra=self.root/"test.json";extra.write_text("not read")
        with self.assertRaisesRegex(ValueError,"UNEXPECTED_MOUNT"):i.preflight(self.root,self.pin)
        extra.unlink()
        self.m["metadata"]["fit"][0]["partition"]="test"
        with self.assertRaises(ValueError):i.preflight(self.root,seal(self.root,self.m))
    def test_tampered_selected_and_symlink(self):
        name=next(iter(self.rows));path=self.root/name;path.write_text("changed")
        with self.assertRaisesRegex(ValueError,"FILE_INTEGRITY"):i.load_selected(self.root,self.pin,authorized=True)
        path.unlink();path.symlink_to(self.root/"manifest.json")
        with self.assertRaises(ValueError):i.preflight(self.root,self.pin)
    def test_scope_and_prior_exposure(self):
        self.m["exposure"]="fresh"
        with self.assertRaises(ValueError):i.preflight(self.root,seal(self.root,self.m))
    def test_end_to_end_actual_writer_roundtrip(self):
        chosen,_=i.preflight(self.root,self.pin)
        blocks=i.load_selected(self.root,self.pin,authorized=True)
        engine=Engine();marks=[]
        result=p.run(engine,p.Plan(chosen),blocks,authorized=True,deadline=time.monotonic()+60,mark=marks.append)
        self.assertEqual(engine.calls,30)
        self.assertEqual(marks,["AUDIT_FIT","AUDIT_SELECTION","INFER_FIT","INFER_SELECTION","METRICS"])
        self.assertEqual(len(result["metrics"]),29)
        self.assertIsNone(result["equal_length_metrics"])
        path=self.root/"out.json";commit="a"*40
        c.write_result(path,result,commit,self.pin)
        report=i.parse(path.read_bytes());self.assertEqual(c.validate_result(report,commit,self.pin),report)
        with self.assertRaises(FileExistsError):c.write_result(path,result,commit,self.pin)
        report["test_used"]=True;report.pop("integrity_sha256");report["integrity_sha256"]=c.digest(report)
        with self.assertRaises(ValueError):c.validate_result(report,commit,self.pin)
    def test_dev_overflow_before_any_forward(self):
        chosen,_=i.preflight(self.root,self.pin);blocks=i.load_selected(self.root,self.pin,authorized=True)
        blocks["selection"][0]["answers"]["ground_truth"]="alpha "*318
        e=Engine()
        with self.assertRaisesRegex(ValueError,"HYPOTHESIS_BUDGET_EXCEEDED"):
            p.run(e,p.Plan(chosen),blocks,authorized=True,deadline=time.monotonic()+60)
        self.assertEqual(e.calls,0)
    def test_deadline_and_budget(self):
        e=Engine();bounded=p.BoundedEngine(e,time.monotonic()-1,16)
        with self.assertRaisesRegex(ValueError,"COMPUTE_LIMIT"):bounded.infer([[4]],"blue")
        self.assertEqual(e.calls,0)
        bounded=p.BoundedEngine(e,time.monotonic()+10,16);bounded.infer([[4]],"blue")
        with self.assertRaisesRegex(ValueError,"COMPUTE_LIMIT"):bounded.infer([[4]],"blue")
if __name__=="__main__":unittest.main()
