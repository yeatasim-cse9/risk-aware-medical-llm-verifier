"""01F-F v1: strict offline preflight launcher. No data acquisition or GPU."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

import e1_master01f_e_package_v1 as old

CODE_MANIFEST_PIN = "131705c72edd4ba8d09019ff6ba604b2ac73b4050e5f4b25c41fb65c8d39324e"
D_COMMIT = old.D_COMMIT
MODULES = 84
SUCCESS = "MASTER01FE_READONLY_PREFLIGHT_PASSED_METADATA_ONLY"
CHILD_SUCCESS = b"MASTER01FD_READONLY_PREFLIGHT_PASSED_METADATA_ONLY"
require = old.require


def unique(pairs):
    out = {}
    for k, v in pairs:
        require(k not in out, "DUPLICATE_CODE_PIN")
        out[k] = v
    return out


def code_pins(raw):
    require(type(raw) is bytes and len(raw) <= 65536 and
            hashlib.sha256(raw).hexdigest() == CODE_MANIFEST_PIN, "CODE_MANIFEST_PIN")
    pins = json.loads(raw, object_pairs_hook=unique)
    require(type(pins) is dict and len(pins) == MODULES and all(
        type(k) is str and re.fullmatch(r"scripts/[a-zA-Z0-9_]+[.]py", k)
        and type(v) is str and re.fullmatch(r"[0-9a-f]{64}", v)
        for k, v in pins.items()), "CODE_MANIFEST_SCHEMA")
    return pins


def directory(path):
    require(isinstance(path, Path) and not path.is_symlink() and path.is_dir(),
            "DIRECTORY_POLICY")
    return path.resolve()


def strict_inventory(root, pins):
    scripts = root / "scripts"
    require(scripts.is_dir() and not scripts.is_symlink(), "SCRIPT_DIRECTORY")
    require({p.relative_to(root).as_posix() for p in scripts.iterdir()} == set(pins),
            "IMPORT_INVENTORY")
    for p in scripts.iterdir():
        require(stat.S_ISREG(p.lstat().st_mode), "IMPORT_FILE_POLICY")


def child(root, bundle, assets, manifest_pin):
    require(type(manifest_pin) is str and re.fullmatch(r"[0-9a-f]{64}", manifest_pin),
            "BUNDLE_PIN")
    # Use an operator-authenticated interpreter, Git installation, and read-only
    # private mounts. Environment filtering is not an OS security sandbox.
    env = {k: os.environ[k] for k in ("PATH", "LANG", "LC_ALL", "SYSTEMROOT")
           if k in os.environ}
    env.update(MASTER01FD_PRIVATE_ENVIRONMENT="YES", MASTER01FD_SOURCE_COMMIT=D_COMMIT,
               MASTER01FD_CUSTODIAN_MANIFEST_SHA256=manifest_pin,
               CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
               GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    return subprocess.run([sys.executable, "-I", "-B", "-c",
        "import sys,runpy;sys.path.insert(0,sys.argv[1]);sys.argv=sys.argv[2:];"
        "runpy.run_path(sys.argv[0],run_name='__main__')",
        str(root / "scripts"), str(root / "scripts/e1_master01f_d_controller_v1.py"),
        "preflight", "--bundle", str(bundle), "--assets", str(assets)],
        cwd=root, env=env, capture_output=True, timeout=300)


def preflight(root, manifest_bytes, bundle, assets, manifest_pin):
    root, bundle, assets = map(directory, (root, bundle, assets))
    pins = code_pins(manifest_bytes)
    strict_inventory(root, pins)
    old.check_pilot(root, pins)
    result = child(root, bundle, assets, manifest_pin)
    require(result.returncode == 0 and result.stderr == b"" and
            result.stdout.splitlines() == [b"MASTER01FD_STAGE=PREFLIGHT", CHILD_SUCCESS],
            "PRIVATE_PREFLIGHT_FAILED")
    return SUCCESS
