"""01F-E v1: independently checked code-only package and offline preflight.
Do not import candidate modules before verifying their bytes.
No dataset acquisition, network submission, or GPU launch.
"""
import hashlib,json,os,re,subprocess,sys
from pathlib import Path
D_COMMIT="4ca7c69f7e00feea4d3d5dcd102c38f563abb64c"
def require(ok,code):
    if not ok:raise ValueError(code)
def git(root,*args):
    return subprocess.check_output(["git",*args],cwd=root,stderr=subprocess.DEVNULL)
def hashes(root,commit):
    require(re.fullmatch(r"[0-9a-f]{40}",commit) is not None,"COMMIT")
    require(git(root,"rev-parse","HEAD").decode().strip()==commit,"CHECKOUT")
    paths=git(root,"ls-tree","-r","--name-only",commit,"scripts").decode().splitlines()
    paths=[p for p in paths if p.endswith(".py")]
    require(bool(paths),"EMPTY_SOURCE")
    actual={p.relative_to(root).as_posix() for p in (root/"scripts").rglob("*.py")}
    require(actual==set(paths),"EXTRA_OR_MISSING_MODULE")
    result={}
    for path in paths:
        p=root/path
        require(not p.is_symlink() and p.is_file() and p.resolve().is_relative_to(root.resolve()),"SOURCE_PATH")
        raw=p.read_bytes()
        require(raw==git(root,"show",commit+":"+path),"SOURCE_BYTES")
        result[path]=hashlib.sha256(raw).hexdigest()
    return result

def check_pilot(root,expected_hashes):
    require(type(expected_hashes) is dict and bool(expected_hashes),"EXTERNAL_SOURCE_HASHES")
    actual=hashes(root,D_COMMIT)
    require(actual==expected_hashes,"SOURCE_MANIFEST_MISMATCH")
    return actual

def preflight(root,expected_hashes,bundle,assets,manifest_pin):
    """D checkout must be separately pinned; no imports until verified.
    Child process executes read-only preflight, never run.
    """
    check_pilot(root,expected_hashes)
    require(re.fullmatch(r"[0-9a-f]{64}",manifest_pin) is not None,"MANIFEST_PIN")
    env={k:v for k,v in os.environ.items() if k not in
         ("PYTHONPATH","PYTHONHOME","MASTER01FD_AUTHORIZED_ONE_GPU_PILOT")}
    env.update(MASTER01FD_PRIVATE_ENVIRONMENT="YES",MASTER01FD_SOURCE_COMMIT=D_COMMIT,
               MASTER01FD_CUSTODIAN_MANIFEST_SHA256=manifest_pin,
               CUDA_VISIBLE_DEVICES="",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
    p=subprocess.run([sys.executable,"-I","-B","-c",
       "import sys,runpy;sys.path.insert(0,sys.argv.pop(1));runpy.run_path(sys.argv[0],run_name='__main__')",
       str(root/"scripts"),str(root/"scripts/e1_master01f_d_controller_v1.py"),
       "preflight","--bundle",str(bundle),"--assets",str(assets)],cwd=root,env=env,
       capture_output=True,timeout=300)
    require(p.returncode==0 and b"MASTER01FD_READONLY_PREFLIGHT_PASSED_METADATA_ONLY" in p.stdout.splitlines(),
            "PRIVATE_PREFLIGHT_FAILED")
    return "MASTER01FE_READONLY_PREFLIGHT_PASSED_METADATA_ONLY"
