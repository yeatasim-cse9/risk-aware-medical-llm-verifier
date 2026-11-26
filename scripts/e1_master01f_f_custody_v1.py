"""Private CPU-only custody recovery. Verify external Ed25519 attestations.
Never acquire a mixed corpus, download a dataset, or launch inference.
The operator must independently authenticate the custodian and isolate mounts.
"""
import argparse
import os
from pathlib import Path
import stat
import subprocess
import tempfile

import e1_master01f_e_provision_v1 as bridge
import e1_master01f_f_preflight_v1 as launch

d = bridge.d
require = d.require
KEY_PREFIX = bytes.fromhex("302a300506032b6570032100")
COMMON = {"schema", "phase", "checkpoint_sha256", "mapping_sha256"}
FIELDS = {
    "snapshot": {"snapshot_sha256", "source_release_verified",
                 "covariates_only", "processing_rights_verified"},
    "answers": {"snapshot_sha256", "request_sha256", "index_sha256",
                "selected_only", "processing_rights_verified"},
    "approval": {"snapshot_receipt_sha256", "answers_receipt_sha256",
                 "bundle_manifest_sha256", "approved_for_readonly_preflight"},
}
CLAIMS = {"source_release_verified", "covariates_only",
          "processing_rights_verified", "selected_only",
          "approved_for_readonly_preflight"}


def regular(path, limit):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        st = os.fstat(stream.fileno())
        require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1
                and 0 < st.st_size <= limit, "FILE_POLICY")
        raw = stream.read(limit + 1)
    require(len(raw) == st.st_size, "FILE_SIZE")
    return raw


def isolated(*roots):
    resolved = []
    for root in roots:
        require(root.is_dir() and not root.is_symlink(), "MOUNT_POLICY")
        q = root.resolve()
        require(all(q != other and q not in other.parents and other not in q.parents
                    for other in resolved), "MOUNT_OVERLAP")
        resolved.append(q)
    return resolved


def inventory(root):
    for entry in root.iterdir():
        st = entry.lstat()
        require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1, "MOUNT_FILE_POLICY")


def private_parent(path):
    require(not path.is_symlink() and path.is_dir(), "PRIVATE_DESTINATION")
    st = path.stat()
    require(st.st_uid == os.getuid() and st.st_mode & 0o077 == 0,
            "PRIVATE_DESTINATION")


def authorized(flag):
    require(flag is True, "PRIVATE_PROCESSING_AUTHORIZATION_REQUIRED")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and
            os.environ.get("HF_HUB_OFFLINE") == "1" and
            os.environ.get("TRANSFORMERS_OFFLINE") == "1",
            "OFFLINE_CPU_ENVIRONMENT_REQUIRED")


def verify_signature(key, raw, signature):
    require(len(key) == 44 and key.startswith(KEY_PREFIX), "ED25519_PUBLIC_KEY")
    require(len(signature) == 64, "SIGNATURE_LENGTH")
    with tempfile.TemporaryDirectory(prefix="01ff-signature-") as tmp:
        root = Path(tmp)
        for name, data in (("key.der", key), ("receipt.json", raw), ("sig", signature)):
            path = root / name
            path.write_bytes(data)
            path.chmod(0o600)
        result = subprocess.run(
            ["openssl", "pkeyutl", "-verify", "-pubin", "-keyform", "DER",
             "-inkey", str(root / "key.der"), "-rawin", "-in", str(root / "receipt.json"),
             "-sigfile", str(root / "sig")],
            capture_output=True, timeout=15, check=False)
        require(result.returncode == 0, "CUSTODIAN_SIGNATURE_INVALID")


def receipt(trust, key_pin, phase):
    require(phase in FIELDS and d.valid_hash(key_pin), "EXTERNAL_KEY_PIN_REQUIRED")
    key = regular(trust / "custodian.der", 44)
    require(d.sha(key) == key_pin, "CUSTODIAN_KEY_PIN")
    raw = regular(trust / (phase + ".json"), 16384)
    signature = regular(trust / (phase + ".sig"), 64)
    verify_signature(key, raw, signature)
    value = d.parse(raw)
    require(type(value) is dict and set(value) == COMMON | FIELDS[phase]
            and type(value["schema"]) is int and value["schema"] == 1
            and value["phase"] == phase
            and value["checkpoint_sha256"] == bridge.hist.CHECKPOINT
            and value["mapping_sha256"] == bridge.hist.MAPPING,
            "CUSTODIAN_RECEIPT_SCOPE")
    for field in FIELDS[phase]:
        require(value[field] is True if field in CLAIMS else d.valid_hash(value[field]),
                "CUSTODIAN_RECEIPT_FIELD")
    return value, d.sha(raw)


def prepare_request(covariates, trust, key_pin, out, *, allow):
    authorized(allow)
    isolated(covariates, trust, out.parent)
    private_parent(out.parent)
    snap, _ = receipt(trust, key_pin, "snapshot")
    inventory(covariates)
    req = bridge.request(covariates, snap["snapshot_sha256"])
    raw = bridge.encode(req)
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return d.sha(raw)


def prepare_bundle(covariates, answers, trust, key_pin, request_file, out, *, allow):
    authorized(allow)
    isolated(covariates, answers, trust, request_file.parent, out.parent)
    private_parent(out.parent)
    snap, _ = receipt(trust, key_pin, "snapshot")
    ans, _ = receipt(trust, key_pin, "answers")
    require(ans["snapshot_sha256"] == snap["snapshot_sha256"], "SNAPSHOT_BINDING")
    inventory(covariates)
    raw = regular(request_file, d.MAX_MANIFEST)
    require(d.sha(raw) == ans["request_sha256"] and
            raw == bridge.encode(bridge.request(covariates, snap["snapshot_sha256"])),
            "REQUEST_BINDING")
    inventory(answers)
    return bridge.provision(covariates, snap["snapshot_sha256"], answers,
                            ans["index_sha256"], out, authorized=True)


def approved_pin(trust, key_pin):
    snap, sh = receipt(trust, key_pin, "snapshot")
    ans, ah = receipt(trust, key_pin, "answers")
    approval, _ = receipt(trust, key_pin, "approval")
    require(ans["snapshot_sha256"] == snap["snapshot_sha256"] and
            approval["snapshot_receipt_sha256"] == sh and
            approval["answers_receipt_sha256"] == ah, "APPROVAL_BINDING")
    return approval["bundle_manifest_sha256"]


def preflight(trust, key_pin, bundle, code_root, code_manifest, assets, *, allow):
    authorized(allow)
    isolated(trust, bundle, code_root, assets, code_manifest.parent)
    pin = approved_pin(trust, key_pin)
    inventory(bundle)  # stat only; no selected answer bytes are opened here.
    return launch.preflight(code_root, regular(code_manifest, 65536), bundle, assets, pin)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("operation", choices=("request", "provision", "preflight"))
    p.add_argument("--trust", required=True, type=Path)
    p.add_argument("--custodian-key-sha256", required=True)
    p.add_argument("--authorized-private-processing", action="store_true")
    for arg in ("covariates", "answers", "request-file", "out", "bundle",
                "code-root", "code-manifest", "assets"):
        p.add_argument("--" + arg, type=Path)
    a = p.parse_args()
    try:
        common = (a.trust, a.custodian_key_sha256)
        kw = {"allow": a.authorized_private_processing}
        if a.operation == "request":
            prepare_request(a.covariates, *common, a.out, **kw)
            marker = "MASTER01FF_PRIVATE_REQUEST_PREPARED"
        elif a.operation == "provision":
            prepare_bundle(a.covariates, a.answers, *common, a.request_file, a.out, **kw)
            marker = "MASTER01FF_PRIVATE_BUNDLE_PREPARED_APPROVAL_PENDING"
        else:
            marker = preflight(*common, a.bundle, a.code_root, a.code_manifest, a.assets, **kw)
        print(marker, flush=True)
    except Exception:
        # Do not emit private paths, row identifiers, pins or arbitrary exceptions.
        print("MASTER01FF_CUSTODY_GATE_FAILED", flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
