"""01F-D v1: projected private input gate. No remote acquisition.
Trust anchor must be provided by an authorized custodian independently of bundle.
Only pre-separated selected answer files may be mounted; never mount raw corpus.
"""
import hashlib,json,os,re,stat
from pathlib import Path
import e1_master01f_b_v1 as f
import e1_master01d_v2_integrity as hist

LIMITS={"fit":256,"selection":128}
ORIGINAL={"fit":"train","selection":"dev"}
COUNTS={"fit":7427,"selection":890}
HEX=re.compile(r"[0-9a-f]{64}")
MAX_ROW=131072
MAX_MANIFEST=4000000
EXPOSURE="ALL_RETAINED_TRAIN_DEV_PREVIOUSLY_LABEL_EXPOSED"
require=f.require

def sha(raw):return hashlib.sha256(raw).hexdigest()
def valid_hash(x):return type(x) is str and HEX.fullmatch(x) is not None
def unique(pairs):
    d={}
    for k,v in pairs:
        require(k not in d,"DUPLICATE_JSON_KEY");d[k]=v
    return d
def invalid_constant(_):raise ValueError("NONFINITE_JSON")
def parse(raw):return json.loads(raw,object_pairs_hook=unique,parse_constant=invalid_constant)

def read_pinned(root,name,pin,limit):
    require(type(root) is Path or isinstance(root,Path),"ROOT")
    require(valid_hash(pin) and type(name) is str and re.fullmatch(r"[a-z0-9_.-]+",name)
            and name not in (".",".."),"FILE_PIN")
    # Open relative to an already verified directory, never follow a symlink.
    fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        item=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
        with os.fdopen(item,"rb") as h:
            s=os.fstat(h.fileno())
            require(stat.S_ISREG(s.st_mode) and 0<s.st_size<=limit,"FILE_SIZE")
            raw=h.read(limit+1)
        require(len(raw)==s.st_size and sha(raw)==pin,"FILE_INTEGRITY")
        return raw
    finally:os.close(fd)

def select(meta,part):
    rows=f.metadata(meta,ORIGINAL[part])
    require(len(rows)==COUNTS[part] and all(valid_hash(p) and valid_hash(g) for p,g in rows),
            "METADATA_COUNTS")
    chosen=[];seen=set()
    for p,g in sorted(rows,key=lambda x:(hashlib.sha256((str(f.SEED)+":"+x[0]).encode()).digest(),x[0])):
        if g in seen:continue
        seen.add(g);chosen.append((p,g,part))
        if len(chosen)==LIMITS[part]:break
    require(len(chosen)==LIMITS[part],"INSUFFICIENT_GROUPS")
    return tuple(chosen)

def manifest(root,expected):
    require(valid_hash(expected),"EXTERNAL_TRUST_ANCHOR_REQUIRED")
    m=parse(read_pinned(root,"manifest.json",expected,MAX_MANIFEST))
    require(type(m) is dict and set(m)=={"schema","mapping_sha256","checkpoint_sha256","exposure",
            "metadata","selected_files"} and type(m["schema"]) is int and m["schema"]==1
            and m["mapping_sha256"]==hist.MAPPING and m["checkpoint_sha256"]==hist.CHECKPOINT
            and m["exposure"]==EXPOSURE,"MANIFEST_SCOPE")
    meta=m["metadata"]
    require(type(meta) is dict and set(meta)==set(ORIGINAL),"METADATA_SCOPE")
    tr=f.metadata(meta["fit"],"train");dv=f.metadata(meta["selection"],"dev")
    require(not({p for p,g in tr}&{p for p,g in dv} or {g for p,g in tr}&{g for p,g in dv}),
            "CROSS_PARTITION_OVERLAP")
    chosen=select(meta["fit"],"fit")+select(meta["selection"],"selection")
    files=m["selected_files"]
    require(type(files) is dict and set(files)=={p+".json" for p,g,k in chosen}
            and all(valid_hash(v) for v in files.values()),"PROJECTED_FILE_SCOPE")
    return chosen,files

def preflight(root,expected):
    chosen,files=manifest(root,expected)
    # Metadata/stat only: never open an answer file here.
    require(set(x.name for x in root.iterdir())=={"manifest.json",*files},"UNEXPECTED_MOUNT")
    for name in files:
        s=(root/name).lstat()
        require(stat.S_ISREG(s.st_mode) and 0<s.st_size<=MAX_ROW,"FILE_SIZE")
    return chosen,files

def load_selected(root,expected,*,authorized):
    require(authorized is True,"AUTHORIZATION_REQUIRED")
    chosen,files=preflight(root,expected)
    blocks={k:[] for k in LIMITS}
    for p,g,k in chosen:
        name=p+".json"
        row=parse(read_pinned(root,name,files[name],MAX_ROW))
        require(type(row) is dict and row.get("parent")==p and row.get("source_group")==g
                and row.get("partition")==k,"ROW_MEMBERSHIP")
        f.b.validate_pairs([row],k)
        blocks[k].append(row)
    return blocks
