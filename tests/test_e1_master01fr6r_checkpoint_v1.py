import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import e1_master01fr6r_checkpoint_v1 as c

class MemoryStore:
    def __init__(self):
        self.files={}; self.private=True; self.fail=False
    def verify(self):
        c.require(self.private,"PRIVATE_REQUIRED")
    def read(self,path):
        self.verify(); return self.files.get(path)
    def create(self,path,raw):
        self.verify()
        c.require(path not in self.files,"CONFLICT")
        self.files[path]=raw
        if self.fail:
            self.fail=False
            raise ValueError("AMBIGUOUS_WRITE")

class Checkpoint(unittest.TestCase):
    def setUp(self):
        self.plan=c.synthetic_plan("r6r-test","a"*40)
        self.pin=c.digest(c.canonical(self.plan))
        self.store=MemoryStore(); self.calls=[]; self.events=[]
        self.prefix=c.PREFIX+self.plan["run_id"]+"/"
    def compute(self,name,prior):
        self.calls.append(name)
        return c.synthetic_compute(name,prior)
    def run_it(self,**kw):
        return c.run(self.plan,self.pin,self.compute,self.store,
                     authorized=True,emit=lambda *a:self.events.append(a),**kw)
    def test_interruption_resume_and_completed_noop(self):
        self.assertIsNone(self.run_it(max_new=1))
        self.assertEqual(self.calls,["first_half"])
        result=self.run_it()
        self.assertEqual(result["aggregates"]["total"]["sum_squares"],385)
        self.assertEqual(self.calls,["first_half","second_half","total"])
        self.assertEqual(result,self.run_it())
        self.assertEqual(len(self.calls),3)
        self.assertEqual(self.events[-1],("already_complete",3,3))
    def test_ambiguous_write_stops_then_readback_resume(self):
        self.store.fail=True
        with self.assertRaises(ValueError):self.run_it()
        self.assertEqual(self.calls,[])
        self.assertEqual(self.run_it()["aggregates"]["total"]["sum_squares"],385)
    def test_gap(self):
        self.run_it()
        del self.store.files[self.prefix+"stage-000.json"]
        self.calls=[]
        with self.assertRaises(ValueError):self.run_it()
        self.assertEqual(self.calls,[])
    def test_modified_checkpoint(self):
        self.run_it(max_new=1)
        path=self.prefix+"stage-000.json"
        obj=json.loads(self.store.files[path]);obj["previous_sha256"]="0"*64
        self.store.files[path]=c.canonical(obj)
        with self.assertRaises(ValueError):self.run_it()
    def test_final_conflict(self):
        self.run_it()
        self.store.files[self.prefix+"complete.json"]=b"{}"
        with self.assertRaises(ValueError):self.run_it()
    def test_plan_pin_and_authorization(self):
        with self.assertRaises(ValueError):
            c.run(self.plan,self.pin,self.compute,self.store)
        with self.assertRaises(ValueError):
            c.run(self.plan,"b"*64,self.compute,self.store,authorized=True)
        self.assertEqual(self.store.files,{})
    def test_nonprivate_store(self):
        self.store.private=False
        with self.assertRaises(ValueError):self.run_it()
        self.assertEqual(self.calls,[])
    def test_raw_and_nonfinite_blocked(self):
        for value in ("raw text",float("nan"),True,[1,2]):
            with self.assertRaises(ValueError):c.metrics({"v":value},["v"])
    def test_remote_path_and_privacy(self):
        store=c.PrivateStore("synthetic-token")
        with self.assertRaises(ValueError):store.url("evidence/master01f-r6/report.json")
        with patch.object(c,"request_json",return_value={"id":c.REPO_ID,"full_name":c.REPO,
                "private":False,"visibility":"public","default_branch":"main","archived":False}):
            with self.assertRaises(ValueError):store.verify()
    def test_create_only_and_remote_readback(self):
        store=c.PrivateStore("synthetic-token")
        raw=b'{"a":1}'
        path=self.prefix+"plan.json"
        with patch.object(store,"verify"),patch.object(store,"read",return_value=raw),             patch.object(c,"request_json",return_value={"content":{"sha":c.blob(raw)}}) as req:
            store.create(path,raw)
            args=req.call_args.args
            self.assertNotIn("sha",json.loads(args[2]))
            self.assertEqual(args[3],"PUT")
        with patch.object(store,"verify"),patch.object(store,"read",return_value=b"wrong"),             patch.object(c,"request_json",return_value={"content":{"sha":c.blob(raw)}}):
            with self.assertRaises(ValueError):store.create(path,raw)
    def test_redirects_blocked(self):
        with self.assertRaises(ValueError):c.NoRedirect().redirect_request()
    def test_plan_collision(self):
        self.run_it(max_new=1)
        self.plan["input_sha256"]="b"*64; self.pin=c.digest(c.canonical(self.plan))
        with self.assertRaises(ValueError):self.run_it()
