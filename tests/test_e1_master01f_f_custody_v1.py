"""Synthetic nonmedical fixtures; real Ed25519 signing and bridge calls."""
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e1_master01f_f_custody_v1 as gate
import test_e1_master01f_e_v1 as previous


class Custody(previous.Provision):
    def setUp(self):
        super().setUp()
        self.trust = self.root / "trust"
        self.trust.mkdir()
        self.reqdir = self.root / "request"
        self.reqdir.mkdir(mode=0o700)
        self.dest = self.root / "destination"
        self.dest.mkdir(mode=0o700)
        self.private = self.root / "synthetic-private.pem"
        subprocess.run(["openssl", "genpkey", "-algorithm", "ED25519",
                        "-out", str(self.private)], check=True, capture_output=True)
        self.key = self.trust / "custodian.der"
        subprocess.run(["openssl", "pkey", "-in", str(self.private), "-pubout",
                        "-outform", "DER", "-out", str(self.key)],
                       check=True, capture_output=True)
        self.kpin = gate.d.sha(self.key.read_bytes())
        self.env = patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.req = self.reqdir / "request.json"
        self.bundle = self.dest / "bundle"
        self.sign("snapshot", snapshot_sha256=self.pin, source_release_verified=True,
                  covariates_only=True, processing_rights_verified=True)
        self.sign("answers", snapshot_sha256=self.pin,
                  request_sha256=self.index["request_sha256"], index_sha256=self.ipin,
                  selected_only=True, processing_rights_verified=True)

    def sign(self, phase, **fields):
        obj = {"schema": 1, "phase": phase,
               "checkpoint_sha256": gate.bridge.hist.CHECKPOINT,
               "mapping_sha256": gate.bridge.hist.MAPPING, **fields}
        path = self.trust / (phase + ".json")
        path.write_bytes(gate.bridge.encode(obj))
        subprocess.run(["openssl", "pkeyutl", "-sign", "-inkey", str(self.private),
                        "-rawin", "-in", str(path), "-out",
                        str(self.trust / (phase + ".sig"))],
                       check=True, capture_output=True)
        return gate.d.sha(path.read_bytes())

    def request(self):
        return gate.prepare_request(self.cov, self.trust, self.kpin, self.req, allow=True)

    def project(self):
        return gate.prepare_bundle(self.cov, self.ans, self.trust, self.kpin,
                                   self.req, self.bundle, allow=True)

    def approve(self, pin):
        return self.sign("approval", snapshot_receipt_sha256=gate.d.sha(
            (self.trust / "snapshot.json").read_bytes()),
            answers_receipt_sha256=gate.d.sha((self.trust / "answers.json").read_bytes()),
            bundle_manifest_sha256=pin, approved_for_readonly_preflight=True)

    def test_signed_end_to_end(self):
        self.assertEqual(self.request(), self.index["request_sha256"])
        pin = self.project()
        self.assertEqual(set(gate.d.load_selected(self.bundle, pin, authorized=True)),
                         {"fit", "selection"})
        with self.assertRaises(OSError):
            gate.approved_pin(self.trust, self.kpin)
        self.approve(pin)
        self.assertEqual(gate.approved_pin(self.trust, self.kpin), pin)
        with self.assertRaisesRegex(ValueError, "DESTINATION_EXISTS"):
            self.project()

    def test_wrong_key_no_snapshot_read(self):
        with patch.object(gate.bridge, "request") as request:
            with self.assertRaisesRegex(ValueError, "CUSTODIAN_KEY_PIN"):
                gate.prepare_request(self.cov, self.trust, "0" * 64, self.req, allow=True)
            request.assert_not_called()

    def test_tampered_signature_no_snapshot_read(self):
        path = self.trust / "snapshot.json"
        path.write_bytes(path.read_bytes() + b" ")
        with patch.object(gate.bridge, "request") as request:
            with self.assertRaisesRegex(ValueError, "CUSTODIAN_SIGNATURE_INVALID"):
                self.request()
            request.assert_not_called()

    def test_unauthorized_before_any_reads(self):
        with patch.object(gate, "regular") as read:
            with self.assertRaisesRegex(ValueError, "AUTHORIZATION_REQUIRED"):
                gate.prepare_request(self.cov, self.trust, self.kpin, self.req, allow=False)
            read.assert_not_called()

    def test_wrong_request_binding_before_answer_reads(self):
        self.request()
        self.req.write_bytes(b"changed")
        with patch.object(gate.bridge, "provision") as provision:
            with self.assertRaisesRegex(ValueError, "REQUEST_BINDING"):
                self.project()
            provision.assert_not_called()

    def test_hardlink_blocks_projection(self):
        self.request()
        first = next(iter(self.index["files"]))
        os.link(self.ans / first, self.root / "outside-link")
        with self.assertRaisesRegex(ValueError, "MOUNT_FILE_POLICY"):
            self.project()
        self.assertFalse(self.bundle.exists())

    def test_signed_approval_cannot_be_reused_after_receipt_change(self):
        self.request()
        pin = self.project()
        self.approve(pin)
        self.sign("answers", snapshot_sha256=self.pin,
                  request_sha256=self.index["request_sha256"], index_sha256="0" * 64,
                  selected_only=True, processing_rights_verified=True)
        with self.assertRaisesRegex(ValueError, "APPROVAL_BINDING"):
            gate.approved_pin(self.trust, self.kpin)

    def test_preflight_approval_gate_stat_only(self):
        self.request()
        pin = self.project()
        self.approve(pin)
        code = self.root / "code"
        assets = self.root / "assets"
        manifests = self.root / "manifests"
        for path in (code, assets, manifests):
            path.mkdir()
        manifest = manifests / "code.json"
        manifest.write_bytes(b"invented")
        with patch.object(gate.launch, "preflight", return_value="marker") as launch:
            with patch.object(gate.d, "load_selected") as read_answers:
                self.assertEqual(gate.preflight(self.trust, self.kpin, self.bundle,
                    code, manifest, assets, allow=True), "marker")
                read_answers.assert_not_called()
                self.assertEqual(launch.call_args.args[-1], pin)


if __name__ == "__main__":
    unittest.main()
