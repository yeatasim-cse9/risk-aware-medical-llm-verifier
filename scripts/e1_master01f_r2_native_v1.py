"""01F-R2: public metadata probe and separately gated private train-only transfer.

No dataset viewer, row API, load_dataset, snapshot_download, model or GPU.
Metadata is not evidence of rights, source independence or content correctness.
The acquisition operation is never invoked by repository CI.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import (Request, build_opener, HTTPRedirectHandler,
                            ProxyHandler)
import e1_master01f_r_protocol_v1 as p

MAX_JSON = 2_000_000
MAX_TOTAL = 4 * 1024**3
CHUNK = 1024 * 1024
HF = "https://huggingface.co"
API = HF + "/api/datasets/" + p.CANDIDATE
DOWNLOAD_HOSTS = frozenset({
    "huggingface.co", "cdn-lfs.huggingface.co", "cdn-lfs.hf.co",
    "cdn-lfs-us-1.hf.co", "cdn-lfs-eu-1.hf.co",
    "cdn-lfs-us-1.huggingface.co", "cdn-lfs-eu-1.huggingface.co",
    "cas-bridge.xethub.hf.co",
})
require = p.require


class MetadataUnavailable(ValueError):
    pass


def unique(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, "DUPLICATE_JSON_KEY")
        obj[key] = value
    return obj


def parse(raw):
    require(type(raw) is bytes and 0 < len(raw) <= MAX_JSON, "JSON_SIZE")
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("NONFINITE_JSON")))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def safe_https(url, hosts):
    q = urlsplit(url)
    require(q.scheme == "https" and q.hostname in hosts
            and q.port in (None, 443) and q.username is None and q.password is None
            and not q.fragment, "URL_SCOPE")
    return q


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("METADATA_REDIRECT")


class BoundedTransferRedirect(HTTPRedirectHandler):
    max_redirections = 4
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        q = safe_https(newurl, DOWNLOAD_HOSTS)
        # No redirect to another Hub object, dataset or API. The one exact
        # Hub resolve URL is constructed below; only blob CDNs may follow.
        require(q.hostname != "huggingface.co", "HUB_REDIRECT_SCOPE")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def metadata_json(url):
    safe_https(url, {"huggingface.co"})
    require(url == API + "/revision/main" or
            re.fullmatch(re.escape(API) + r"/tree/[0-9a-f]{40}[?]recursive=true", url),
            "METADATA_ENDPOINT")
    opener = build_opener(ProxyHandler({}), NoRedirect())
    req = Request(url, headers={"Accept": "application/json",
                                "User-Agent": "medllm-research-metadata/1"})
    try:
        with opener.open(req, timeout=25) as response:
            require(response.status == 200, "METADATA_HTTP")
            require(not response.headers.get("Link"), "PAGINATED_INVENTORY")
            raw = response.read(MAX_JSON + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise MetadataUnavailable("METADATA_UNAVAILABLE") from exc
    return parse(raw)


def normalize(info, tree):
    """Only extract public file inventory, not rows; unsupported layout fails."""
    require(type(info) is dict and info.get("id") == p.CANDIDATE
            and type(info.get("sha")) is str and p.REV.fullmatch(info["sha"]),
            "PUBLISHER_ID_REVISION")
    require(info.get("private") is False and info.get("gated") is False,
            "PUBLIC_RELEASE_REQUIRED")
    card = info.get("cardData")
    require(type(card) is dict and card.get("license") == "apache-2.0",
            "DECLARED_LICENSE_DRIFT")
    require(type(tree) is list and 0 < len(tree) <= 10005, "TREE_SCHEMA")
    paths, shards = set(), []
    for x in tree:
        require(type(x) is dict and type(x.get("path")) is str
                and x["path"] not in paths, "TREE_ENTRY")
        path = x["path"]
        paths.add(path)
        if x.get("type") == "directory":
            require(path == "data", "UNREVIEWED_DIRECTORY")
            continue
        require(x.get("type") == "file", "UNREVIEWED_TREE_TYPE")
        if path in {"README.md", ".gitattributes", "LICENSE", "LICENSE.txt", "NOTICE"}:
            continue
        require(p.PATH.fullmatch(path), "UNREVIEWED_FILE")
        lfs = x.get("lfs")
        require(type(lfs) is dict and type(lfs.get("oid")) is str
                and p.SHA.fullmatch(lfs["oid"]) and type(lfs.get("size")) is int
                and lfs["size"] > 0 and type(x.get("size")) is int
                and x["size"] == lfs["size"], "PUBLISHER_LFS_METADATA")
        shards.append({"path": path, "sha256": lfs["oid"], "size": lfs["size"]})
    metadata = {"repository": p.CANDIDATE, "revision": info["sha"],
                "shards": sorted(shards, key=lambda x: x["path"])}
    p.train_plan(metadata, info["sha"], p.digest(metadata))
    return metadata


def probe():
    info = metadata_json(API + "/revision/main")
    require(type(info) is dict and type(info.get("sha")) is str
            and p.REV.fullmatch(info["sha"]), "PUBLISHER_REVISION")
    tree = metadata_json(API + "/tree/" + info["sha"] + "?recursive=true")
    return normalize(info, tree)


APPROVAL = {"schema", "repository", "revision", "metadata_sha256",
            "supervisor_approved", "all_train_file_contents_rights_verified",
            "native_train_paths_reviewed", "private_cpu_processing_approved",
            "protected_splits_excluded"}


def approved_plan(metadata, revision, metadata_pin, approval, approval_pin):
    """Pins MUST come from a separate trusted review, never from this runner.

    A supplied receipt and digest do not certify the truth of its statements.
    Whole-shard rights are required BEFORE transfer, not only rights of rows
    later selected. No bypass for a single source selected after downloading.
    """
    plan = p.train_plan(metadata, revision, metadata_pin)
    require(type(approval_pin) is str and p.SHA.fullmatch(approval_pin),
            "EXTERNAL_APPROVAL_PIN_REQUIRED")
    require(type(approval) is dict and set(approval) == APPROVAL
            and p.digest(approval) == approval_pin, "APPROVAL_PIN")
    require(type(approval["schema"]) is int and approval["schema"] == 1
            and approval["repository"] == p.CANDIDATE
            and approval["revision"] == revision
            and approval["metadata_sha256"] == metadata_pin,
            "APPROVAL_BINDING")
    for key in APPROVAL - {"schema", "repository", "revision", "metadata_sha256"}:
        require(approval[key] is True, "APPROVAL_REQUIRED")
    require(sum(x["size"] for x in plan["train_files"]) <= MAX_TOTAL, "TRANSFER_BUDGET")
    return plan


def resolve_url(revision, entry):
    require(type(revision) is str and p.REV.fullmatch(revision), "REVISION_REQUIRED")
    require(type(entry) is dict and set(entry) == {"path", "sha256", "size"},
            "SHARD_SCHEMA")
    match = p.PATH.fullmatch(entry["path"])
    require(match is not None and match.group(1) == "train", "TRAIN_ONLY")
    return HF + "/datasets/" + p.CANDIDATE + "/resolve/" + revision + "/" + entry["path"]


def transfer(revision, entry, path):
    url = resolve_url(revision, entry)
    opener = build_opener(ProxyHandler({}), BoundedTransferRedirect())
    req = Request(url, headers={"Accept-Encoding": "identity",
                                "User-Agent": "medllm-research-train/1"})
    with opener.open(req, timeout=60) as response:
        safe_https(response.geturl(), DOWNLOAD_HOSTS)
        require(response.status == 200, "TRANSFER_HTTP")
        require(response.headers.get("Content-Encoding") in (None, "identity"),
                "TRANSFER_ENCODING")
        h = hashlib.sha256()
        count = 0
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as handle:
            while True:
                raw = response.read(min(CHUNK, entry["size"] - count + 1))
                if not raw:
                    break
                count += len(raw)
                require(count <= entry["size"], "TRANSFER_SIZE")
                h.update(raw)
                handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        require(count == entry["size"] and h.hexdigest() == entry["sha256"],
                "TRANSFER_INTEGRITY")


def private_parent(parent):
    require(parent.is_dir() and not parent.is_symlink(), "PRIVATE_DESTINATION")
    st = parent.stat()
    require(st.st_uid == os.getuid() and st.st_mode & 0o077 == 0,
            "PRIVATE_DESTINATION")


def file_identity(path, size):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as handle:
        st = os.fstat(handle.fileno())
        require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1
                and st.st_size == size, "STORED_FILE_POLICY")
        return hashlib.file_digest(handle, "sha256").hexdigest()


def acquire(metadata, revision, metadata_pin, approval, approval_pin, out, *, allow):
    require(allow is True, "PRIVATE_CPU_AUTHORIZATION_REQUIRED")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU_ENVIRONMENT_REQUIRED")
    plan = approved_plan(metadata, revision, metadata_pin, approval, approval_pin)
    private_parent(out.parent)
    require(not out.exists() and not out.is_symlink(), "DESTINATION_EXISTS")
    tmp = Path(tempfile.mkdtemp(prefix=".01fr2-", dir=out.parent))
    try:
        for entry in plan["train_files"]:
            name = Path(entry["path"]).name
            transfer(revision, entry, tmp / name)
            require(file_identity(tmp / name, entry["size"]) == entry["sha256"],
                    "STORED_FILE_INTEGRITY")
        require({x.name for x in tmp.iterdir()} ==
                {Path(x["path"]).name for x in plan["train_files"]},
                "STORED_INVENTORY")
        # Parent exclusively controlled by operator, no concurrent writers.
        require(not out.exists() and not out.is_symlink(), "DESTINATION_EXISTS")
        os.rename(tmp, out)
        return {"native_train_files": len(plan["train_files"]),
                "actual_file_bytes_verified": True, "rows_opened": False,
                "source_disjointness_verified": False,
                "clinical_validity_verified": False, "execution_authorized": False}
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)


def read_json(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as handle:
        st = os.fstat(handle.fileno())
        require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1
                and 0 < st.st_size <= MAX_JSON, "INPUT_FILE_POLICY")
        return parse(handle.read(MAX_JSON + 1))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="op", required=True)
    sub.add_parser("metadata")
    a = sub.add_parser("acquire")
    a.add_argument("--metadata", type=Path, required=True)
    a.add_argument("--revision", required=True)
    a.add_argument("--metadata-sha256", required=True)
    a.add_argument("--approval", type=Path, required=True)
    a.add_argument("--approval-sha256", required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--authorized-private-cpu", action="store_true")
    args = parser.parse_args()
    if args.op == "metadata":
        try:
            data = probe()
            print("MASTER01FR2_PUBLIC_METADATA=" + canonical(data).decode("ascii"))
            print("MASTER01FR2_PUBLIC_METADATA_SHA256=" + p.digest(data))
            print("MASTER01FR2_PUBLIC_METADATA_OBSERVED_REVIEW_REQUIRED")
        except Exception:
            # No arbitrary HTTP body, URL, query tokens or server diagnostics.
            print("MASTER01FR2_PUBLIC_METADATA_BLOCKED")
            raise SystemExit(2) from None
    else:
        try:
            require(args.authorized_private_cpu, "PRIVATE_CPU_AUTHORIZATION_REQUIRED")
            require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU_ENVIRONMENT_REQUIRED")
            # Private Kaggle operator only; output must be ephemeral, not
            # notebook version outputs. Code cannot certify account privacy.
            temp = Path("/kaggle/temp").resolve()
            require(temp.is_dir() and temp in args.out.parent.resolve().parents,
                    "KAGGLE_EPHEMERAL_DESTINATION")
            acquire(read_json(args.metadata), args.revision, args.metadata_sha256,
                    read_json(args.approval), args.approval_sha256, args.out,
                    allow=True)
            print("MASTER01FR2_PRIVATE_TRAIN_BYTES_VERIFIED_NO_ROWS_OPENED")
        except Exception:
            print("MASTER01FR2_PRIVATE_ACQUISITION_BLOCKED")
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
