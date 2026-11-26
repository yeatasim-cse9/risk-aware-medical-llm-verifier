"""Synthetic-only metadata, path, approval and transfer tests."""
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from io import BytesIO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e1_master01f_r2_native_v1 as n


class Native(unittest.TestCase):
    def setUp(self):
        self.rev = "a"*40
        self.info = {"id": n.p.CANDIDATE, "sha": self.rev, "private": False,
                     "gated": False, "cardData": {"license": "apache-2.0"}}
        self.raw = b"invented train bytes"
        self.tree = [{"type":"directory","path":"data"},
                     {"type":"file","path":"README.md","size":5}]
        for split, raw in (("train",self.raw),("validation",b"v"),("test",b"t")):
            self.tree.append({"type":"file","path":f"data/{split}-00000-of-00001.parquet",
                              "size":len(raw), "lfs":{"oid":hashlib.sha256(raw).hexdigest(),
                                                    "size":len(raw)}})
        self.meta = n.normalize(self.info, self.tree)
        self.pin = n.p.digest(self.meta)
        self.ap = {"schema":1, "repository":n.p.CANDIDATE, "revision":self.rev,
                   "metadata_sha256":self.pin}
        self.ap.update({k:True for k in n.APPROVAL-set(self.ap)})
        self.apin = n.p.digest(self.ap)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root/"out"
        env = patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES":""})
        env.start()
        self.addCleanup(env.stop)

    def acquire(self, allow=True):
        return n.acquire(self.meta,self.rev,self.pin,self.ap,self.apin,self.out,allow=allow)

    def fake_transfer(self, revision, entry, path):
        self.assertIn("/train-", n.resolve_url(revision,entry))
        path.write_bytes(self.raw)
        path.chmod(0o600)

    def test_probe_only_fixed_metadata_endpoints(self):
        with patch.object(n,"metadata_json",side_effect=[self.info,self.tree]) as get:
            self.assertEqual(n.probe(), self.meta)
        self.assertEqual([c.args[0] for c in get.call_args_list],
                         [n.API+"/revision/main",n.API+"/tree/"+self.rev+"?recursive=true"])

    def test_no_rows_endpoint(self):
        with self.assertRaisesRegex(ValueError,"METADATA_ENDPOINT"):
            n.metadata_json(n.API+"/rows")
        with self.assertRaisesRegex(ValueError,"URL_SCOPE"):
            n.metadata_json("https://example.invalid/")

    def test_ambiguous_file_rejected(self):
        self.tree.append({"type":"file","path":"extra.parquet"})
        with self.assertRaisesRegex(ValueError,"UNREVIEWED_FILE"):
            n.normalize(self.info,self.tree)

    def test_license_and_private_drift(self):
        self.info["cardData"]["license"]="other"
        with self.assertRaisesRegex(ValueError,"DECLARED_LICENSE_DRIFT"):
            n.normalize(self.info,self.tree)
        self.info["private"]=True
        with self.assertRaisesRegex(ValueError,"PUBLIC_RELEASE_REQUIRED"):
            n.normalize(self.info,self.tree)

    def test_lfs_identity(self):
        self.tree[-1]["lfs"]["size"]+=1
        with self.assertRaisesRegex(ValueError,"PUBLISHER_LFS_METADATA"):
            n.normalize(self.info,self.tree)

    def test_protected_split_urls_never_resolved(self):
        for item in self.meta["shards"]:
            if "/train-" not in item["path"]:
                with self.assertRaisesRegex(ValueError,"TRAIN_ONLY"):
                    n.resolve_url(self.rev,item)

    def test_bad_approval_before_transfer(self):
        self.ap["all_train_file_contents_rights_verified"]=False
        self.apin=n.p.digest(self.ap)
        with patch.object(n,"transfer") as tr:
            with self.assertRaisesRegex(ValueError,"APPROVAL_REQUIRED"):
                self.acquire()
            tr.assert_not_called()

    def test_unauthorized_no_transfer(self):
        with patch.object(n,"transfer") as tr:
            with self.assertRaisesRegex(ValueError,"AUTHORIZATION_REQUIRED"):
                self.acquire(False)
            tr.assert_not_called()

    def test_cpu_gate(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES":"0"}), patch.object(n,"transfer") as tr:
            with self.assertRaisesRegex(ValueError,"CPU_ENVIRONMENT_REQUIRED"):
                self.acquire()
            tr.assert_not_called()

    def test_pin_binding(self):
        self.ap["revision"]="b"*40
        self.apin=n.p.digest(self.ap)
        with self.assertRaisesRegex(ValueError,"APPROVAL_BINDING"):
            self.acquire()

    def test_success_only_train_and_no_overwrite(self):
        with patch.object(n,"transfer",side_effect=self.fake_transfer) as tr:
            report=self.acquire()
            self.assertEqual(tr.call_count,1)
            self.assertEqual([x.name for x in self.out.iterdir()],
                             ["train-00000-of-00001.parquet"])
            self.assertTrue(report["actual_file_bytes_verified"])
            self.assertFalse(report["rows_opened"])
            self.assertFalse(report["execution_authorized"])
            with self.assertRaisesRegex(ValueError,"DESTINATION_EXISTS"):
                self.acquire()

    def test_tampered_bytes_cleanup(self):
        def bad(rev,entry,path):
            path.write_bytes(b"x"*entry["size"])
        with patch.object(n,"transfer",side_effect=bad):
            with self.assertRaisesRegex(ValueError,"STORED_FILE_INTEGRITY"):
                self.acquire()
        self.assertEqual(list(self.root.iterdir()),[])

    def test_hardlink_rejected(self):
        def bad(rev,entry,path):
            outside=self.root/"outside"
            outside.write_bytes(self.raw)
            os.link(outside,path)
        with patch.object(n,"transfer",side_effect=bad):
            with self.assertRaisesRegex(ValueError,"STORED_FILE_POLICY"):
                self.acquire()
        self.assertFalse(self.out.exists())

    def test_redirect_scope(self):
        r=n.BoundedTransferRedirect()
        req=n.Request(n.HF+"/datasets/a/b/resolve/rev/data/train")
        for url in ("http://cas-bridge.xethub.hf.co/a",
                    "https://example.invalid/a",
                    "https://huggingface.co/datasets/a/b/resolve/rev/data/test"):
            with self.assertRaises(ValueError):
                r.redirect_request(req,None,302,"",{},url)

    def test_duplicate_json_rejected(self):
        with self.assertRaisesRegex(ValueError,"DUPLICATE_JSON_KEY"):
            n.parse(b'{"a":1,"a":2}')

    def test_byte_stream_actual_transfer(self):
        class Response(BytesIO):
            status=200
            headers={}
            def geturl(self):
                return "https://cas-bridge.xethub.hf.co/blob"
        entry=next(x for x in self.meta["shards"] if "/train-" in x["path"])
        class Opener:
            def open(self,req,timeout):
                self.url=req.full_url
                return Response(self.data)
        op=Opener()
        op.data=self.raw
        path=self.root/"file"
        with patch.object(n,"build_opener",return_value=op):
            n.transfer(self.rev,entry,path)
        self.assertEqual(path.read_bytes(),self.raw)
        self.assertTrue(op.url.endswith(entry["path"]))
        path.unlink()
        op.data=self.raw+b"x"
        with patch.object(n,"build_opener",return_value=op):
            with self.assertRaisesRegex(ValueError,"TRANSFER_SIZE"):
                n.transfer(self.rev,entry,path)

    def test_metadata_reason_is_allowlisted(self):
        self.assertEqual(n.metadata_reason(ValueError("PUBLISHER_REVISION")),
                         "PUBLISHER_REVISION")
        self.assertEqual(n.metadata_reason(n.MetadataUnavailable("METADATA_UNAVAILABLE")),
                         "METADATA_UNAVAILABLE")
        self.assertEqual(n.metadata_reason(ValueError("private server message")),
                         "UNCLASSIFIED_METADATA_ERROR")
        self.assertEqual(n.metadata_reason(ValueError({"arbitrary": "data"})),
                         "UNCLASSIFIED_METADATA_ERROR")

    def test_inventory_is_observation_only(self):
        observed = n.observe_inventory(self.info, self.tree)
        self.assertFalse(observed["qualified"])
        self.assertEqual(observed["revision"], self.rev)
        self.assertTrue(any(x["path"].startswith("data/test-") for x in observed["observed_entries"]))
        self.tree.append({"type": "file", "path": "../bad"})
        self.assertTrue(n.observe_inventory(self.info, self.tree)["inventory_truncated"])
        self.assertFalse(any(x["path"] == "../bad" for x in
                             n.observe_inventory(self.info, self.tree)["observed_entries"]))


if __name__=="__main__":
    unittest.main()
