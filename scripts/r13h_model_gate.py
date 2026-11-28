"""Publisher metadata gate; metadata-only in CI, exact bytes in authorized runtime."""
import hashlib, json, pathlib, re, urllib.request, urllib.parse

REPO="Qwen/Qwen3-4B-Instruct-2507"
REV="cdbee75f17c01a7cc42f958dc650907174af0554"
SHARDS={
"model-00001-of-00003.safetensors":"75311d91bb08cf0b882913da464a1e722a31fb44db35208663487efb7a3d8ed6",
"model-00002-of-00003.safetensors":"0b48adbb1f60e901153d91907ba11ce63bd4b8b584482e730f48808d055dfba1",
"model-00003-of-00003.safetensors":"7dd39ccca5e4de123c74c14af44c9bf2eb75df33b4614382af0134528e060d5d"}
ASSETS=("LICENSE","config.json","generation_config.json","merges.txt","model.safetensors.index.json","tokenizer.json","tokenizer_config.json","vocab.json")
FILES=tuple(SHARDS)+ASSETS

def gate(ok, code):
    if not ok: raise ValueError(code)

def manifest(info):
    gate(info.get("sha")==REV and info.get("id",info.get("modelId"))==REPO,"MODEL_REVISION")
    gate(info.get("cardData",{}).get("license")=="apache-2.0","MODEL_LICENSE")
    siblings=info["siblings"]
    by={x["rfilename"]:x for x in siblings}
    gate(len(by)==len(siblings) and set(FILES)<=set(by),"MODEL_FILES")
    result={}
    for name in FILES:
        x=by[name]; size=x.get("size")
        gate(type(size) is int and 0<size<=5_000_000_000,"ASSET_SIZE")
        if x.get("lfs"):
            h=x["lfs"]["sha256"]; alg="sha256"
            gate(re.fullmatch("[0-9a-f]{64}",h) is not None and x["lfs"]["size"]==size,"ASSET_LFS")
        else:
            h=x.get("blobId"); alg="git-sha1"
            gate(isinstance(h,str) and re.fullmatch("[0-9a-f]{40}",h) is not None,"ASSET_BLOB")
        if name in SHARDS: gate(alg=="sha256" and h==SHARDS[name],"SHARD_PIN")
        result[name]={"size":size,"algorithm":alg,"hash":h}
    gate(sum(v["size"] for v in result.values())<8_300_000_000,"TOTAL_SIZE")
    return result

def get_manifest():
    url="https://huggingface.co/api/models/"+REPO+"/revision/"+REV+"?blobs=true"
    with urllib.request.urlopen(url,timeout=45) as r:
        raw=r.read(2_000_001)
    gate(len(raw)<=2_000_000,"METADATA_SIZE")
    return manifest(json.loads(raw))

def acquire(folder, files):
    # No credentials, private data, remote code or broad snapshot. Explicit file allowlist.
    folder=pathlib.Path(folder)
    gate(set(files)==set(FILES),"DOWNLOAD_SCOPE")
    for name in FILES:
        a=files[name]; size=a["size"]
        h=hashlib.sha256() if a["algorithm"]=="sha256" else hashlib.sha1(("blob "+str(size)+"\0").encode())
        n=0; url="https://huggingface.co/"+REPO+"/resolve/"+REV+"/"+urllib.parse.quote(name)
        with urllib.request.urlopen(url,timeout=90) as r, (folder/name).open("xb") as f:
            while True:
                block=r.read(1024*1024)
                if not block: break
                n+=len(block); gate(n<=size,"DOWNLOAD_SIZE")
                h.update(block); f.write(block)
        gate(n==size and h.hexdigest()==a["hash"],"DOWNLOAD_HASH")
    cfg=json.loads((folder/"config.json").read_text())
    gate(cfg.get("model_type")=="qwen3" and cfg.get("architectures")==["Qwen3ForCausalLM"] and not cfg.get("auto_map"),"MODEL_ARCH")
    idx=json.loads((folder/"model.safetensors.index.json").read_text())
    gate(set(idx["weight_map"].values())==set(SHARDS),"WEIGHT_INDEX")

