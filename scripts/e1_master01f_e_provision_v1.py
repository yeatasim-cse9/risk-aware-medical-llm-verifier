"""01F-E v1: offline custody bridge. No raw corpus, network, GPU or test answers.
Only independently pinned covariate snapshots and selected-only answer exports.
No self-issued trust anchors establish source provenance.
"""
import json,os,shutil,tempfile
from pathlib import Path
import e1_master01f_d_inputs_v1 as d
import e1_master01c_core as c
import e1_master01d_v2_integrity as hist
require=d.require
MAX_COVARIATES=96*1024*1024
MAX_INDEX=4*1024*1024
ROW_FIELDS={"key","source","question","context","config","category","difficulty"}
def encode(x):
    return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
def exact(obj,keys,code):
    require(type(obj) is dict and set(obj)==set(keys),code)
def snapshot(root,pin):
    require(set(x.name for x in root.iterdir())=={"covariates.json"},"COVARIATE_MOUNT_SCOPE")
    value=d.parse(d.read_pinned(root,"covariates.json",pin,MAX_COVARIATES))
    exact(value,{"schema","records","mapping"},"SNAPSHOT_SCHEMA")
    require(type(value["schema"]) is int and value["schema"]==1,"SNAPSHOT_VERSION")
    records=value["records"];exact(records,c.SPLITS,"COVARIATE_SPLITS")
    for s,rr in records.items():
        require(type(rr) is list and len(rr)==hist.ORIGINAL[s],"ORIGINAL_COUNTS")
        for row in rr:
            exact(row,ROW_FIELDS,"COVARIATE_FIELDS")
            require(type(row["key"]) is list and len(row["key"])==2,"KEY")
            row["key"]=tuple(row["key"])
    c.validate(records,{k:set() for k in c.PAIRS})
    mapping=value["mapping"]
    require(type(mapping) is list and all(type(x) is list and len(x)==3 and
            type(x[0]) is str and type(x[1]) is int and x[2] in c.SPLITS[:3] for x in mapping),
            "MAPPING_SCHEMA")
    require(mapping==sorted(mapping) and len({tuple(x[:2]) for x in mapping})==len(mapping)
            and c.digest({"seed":c.SEED,"assignment":mapping})==hist.MAPPING,"MAPPING_PIN")
    lookup={r["key"]:(s,r) for s,rr in records.items() for r in rr}
    kept={s:[] for s in c.SPLITS}
    for a,b,s in mapping:
        require((a,b) in lookup and lookup[a,b][0]==s,"MAPPING_MEMBERSHIP")
        kept[s].append(lookup[a,b][1])
    kept["official_holdout"]=records["official_holdout"]
    require({s:len(rr) for s,rr in kept.items()}==hist.RETAINED,"RETAINED_COUNTS")
    groups=hist.source_components(records)
    occupied={}
    for s,rr in kept.items():
        for row in rr:
            g=groups[row["key"]]
            require(occupied.setdefault(g,s)==s,"SOURCE_OVERLAP")
    metadata={};selected={}
    for part,s in d.ORIGINAL.items():
        m=[];by={}
        for row in kept[s]:
            p=c.digest(list(row["key"]));g=c.digest(list(groups[row["key"]]))
            require(p not in by,"PARENT_COLLISION")
            m.append({"parent":p,"source_group":g,"partition":s});by[p]=row
        metadata[part]=m
        for p,g,k in d.select(m,part):
            selected[p]=(g,k,by[p])
    require(len(selected)==sum(d.LIMITS.values()),"SAMPLE_COUNT")
    return metadata,selected

def request(root,pin):
    """Private request for custodian; only selected train/dev keys. Never public."""
    _,selected=snapshot(root,pin)
    return {"schema":1,"snapshot_sha256":pin,"exposure":d.EXPOSURE,
            "selected":[{"parent":p,"source_group":g,"partition":k,
                "key":list(row["key"])} for p,(g,k,row) in sorted(selected.items())]}

def provision(covariates,snapshot_pin,answers,answer_index_pin,out,*,authorized):
    require(authorized is True,"PROVISION_AUTHORIZATION_REQUIRED")
    require(not out.exists() and not out.is_symlink(),"DESTINATION_EXISTS")
    metadata,selected=snapshot(covariates,snapshot_pin)
    index=d.parse(d.read_pinned(answers,"index.json",answer_index_pin,MAX_INDEX))
    exact(index,{"schema","snapshot_sha256","request_sha256","files"},"INDEX_SCHEMA")
    require(type(index["schema"]) is int and index["schema"]==1
            and index["snapshot_sha256"]==snapshot_pin
            and index["request_sha256"]==d.sha(encode(request(covariates,snapshot_pin))),"INDEX_BINDING")
    files=index["files"]
    require(type(files) is dict and set(files)=={p+".json" for p in selected}
            and all(d.valid_hash(h) for h in files.values()),"ANSWER_INDEX_SCOPE")
    require(set(x.name for x in answers.iterdir())=={"index.json",*files},"ANSWER_MOUNT_SCOPE")
    # Validate inventory before opening any answer. No test, raw parquet, or extras.
    import stat
    for name in files:
        st=(answers/name).lstat()
        require(stat.S_ISREG(st.st_mode) and 0<st.st_size<=d.MAX_ROW,"ANSWER_FILE_POLICY")
    tmp=Path(tempfile.mkdtemp(prefix=".01fe-",dir=out.parent))
    try:
        written={}
        for p,(g,k,cov) in sorted(selected.items()):
            name=p+".json"
            v=d.parse(d.read_pinned(answers,name,files[name],d.MAX_ROW))
            exact(v,{"parent","key","partition","answers"},"ANSWER_SCHEMA")
            require(v["parent"]==p and v["key"]==list(cov["key"])
                    and v["partition"]==d.ORIGINAL[k],"ANSWER_MEMBERSHIP")
            row={"parent":p,"source_group":g,"partition":k,"question":cov["question"],
                 "context":cov["context"],"answers":v["answers"]}
            d.f.b.validate_pairs([row],k)
            raw=encode(row);require(len(raw)<=d.MAX_ROW,"ROW_SIZE")
            (tmp/name).write_bytes(raw);(tmp/name).chmod(0o600);written[name]=d.sha(raw)
        manifest={"schema":1,"mapping_sha256":hist.MAPPING,"checkpoint_sha256":hist.CHECKPOINT,
                  "exposure":d.EXPOSURE,"metadata":metadata,"selected_files":written}
        raw=encode(manifest);require(len(raw)<=d.MAX_MANIFEST,"MANIFEST_SIZE")
        (tmp/"manifest.json").write_bytes(raw);(tmp/"manifest.json").chmod(0o600)
        pin=d.sha(raw)
        d.preflight(tmp,pin)
        d.load_selected(tmp,pin,authorized=True)
        require(not out.exists() and not out.is_symlink(),"DESTINATION_EXISTS")
        # Parent must be exclusively controlled by custodian; no concurrent writers.
        os.rename(tmp,out)
        return pin
    finally:
        if tmp.exists():shutil.rmtree(tmp)
