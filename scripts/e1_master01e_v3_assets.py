"""Model-only metadata qualification and authorized local asset retrieval.
No medical data, GPU, inference, or mutable main revision.
Qualification verifies published weight/SentencePiece pointers, not weight bytes.
"""
import hashlib,json,re,shutil
from pathlib import Path
from e1_master01d_v2_integrity import require
from e1_master01e_v2_neural import check_assets

MODEL="cross-encoder/nli-deberta-v3-base"
REVISION="6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
LARGE={"model.safetensors":"d8148c6d49e0a7925134294c56326c71fe0ab1dc390e37355e00c7efbb488afa",
       "spm.model":"c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd"}
SMALL=("config.json","tokenizer_config.json","special_tokens_map.json","added_tokens.json")
FILES=set(LARGE)|set(SMALL)

def digest(m):
    return hashlib.sha256(json.dumps(m,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()
def sha(b):return hashlib.sha256(b).hexdigest()
def blob(b):return hashlib.sha1(("blob "+str(len(b))+"\0").encode()+b).hexdigest()
def validate_manifest(m):
    require(type(m) is dict and set(m)=={"model","revision","files"}
            and m["model"]==MODEL and m["revision"]==REVISION,"MODEL_REVISION")
    require(type(m["files"]) is dict and set(m["files"])==FILES
            and all(type(h) is str and re.fullmatch("[0-9a-f]{64}",h) for h in m["files"].values())
            and all(m["files"][k]==v for k,v in LARGE.items()),"MODEL_FILE_PINS")
    return m

def qualify(api,fetch_small):
    """Public metadata + four small JSON files only. No weight or spm download."""
    info=api.model_info(MODEL,revision=REVISION,files_metadata=True,token=False)
    require(info.sha==REVISION and not info.private and info.card_data is not None
            and info.card_data.license=="apache-2.0","MODEL_METADATA")
    entries={x.rfilename:x for x in info.siblings}
    require(FILES<=set(entries),"MODEL_FILE_SET")
    for k,h in LARGE.items():
        x=entries[k]
        require(x.lfs is not None and x.lfs.sha256==h and type(x.size) is int
                and 0<x.size<=800_000_000,"PUBLISHED_LFS_PIN")
    fs=dict(LARGE)
    for k in SMALL:
        x=entries[k]
        require(type(x.size) is int and 0<x.size<=65536 and x.lfs is None,"SMALL_FILE_POLICY")
        data=fetch_small(k)
        require(type(data) is bytes and len(data)==x.size and blob(data)==x.blob_id,"GIT_BLOB_PIN")
        obj=json.loads(data)
        require(type(obj) is dict,"MODEL_CONFIG")
        if k=="config.json":
            require(obj.get("model_type")=="deberta-v2" and obj.get("max_position_embeddings")==512
                and obj.get("id2label")=={"0":"contradiction","1":"entailment","2":"neutral"},
                "MODEL_LABEL_CONFIG")
        fs[k]=sha(data)
    return validate_manifest({"model":MODEL,"revision":REVISION,"files":fs})

def acquire(root,manifest,expected_digest,*,authorized):
    require(authorized is True,"PRIVATE_AUTHORIZATION_REQUIRED")
    validate_manifest(manifest)
    require(digest(manifest)==expected_digest,"MANIFEST_PIN")
    require(isinstance(root,Path) and not root.exists(),"ASSET_DESTINATION")
    from huggingface_hub import hf_hub_download
    root.mkdir(mode=0o700)
    for k in sorted(FILES):
        p=Path(hf_hub_download(MODEL,k,revision=REVISION,token=False))
        shutil.copyfile(p,root/k)
    check_assets(root,manifest,expected_digest)

if __name__=="__main__":
    from huggingface_hub import HfApi,hf_hub_download
    def small(k):
        require(k in SMALL,"SMALL_FILE_POLICY")
        return Path(hf_hub_download(MODEL,k,revision=REVISION,token=False)).read_bytes()
    m=qualify(HfApi(),small)
    print("MASTER01EV3_MODEL_MANIFEST="+json.dumps(m,sort_keys=True,separators=(",",":")))
    print("MASTER01EV3_MANIFEST_SHA256="+digest(m))
