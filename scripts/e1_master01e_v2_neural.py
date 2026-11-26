"""Offline NLI inference engine. No network, medical I/O, or automatic model selection.
Candidate asset manifest and its SHA256 must be separately reviewed and pinned.
Scores are evidence-alignment proxies, not clinical truth or calibrated probabilities.
"""
import hashlib,json,re
from pathlib import Path
from e1_master01d_v2_integrity import require

CANDIDATES=("cross-encoder/nli-deberta-v3-base","MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
LABELS={"entailment","contradiction","neutral"}
ALLOWED={"config.json","model.safetensors","tokenizer.json","tokenizer_config.json",
         "special_tokens_map.json","added_tokens.json","spm.model","vocab.txt"}
MAX_LEN=512
WINDOW=192
OVERLAP=48
SLOTS=16

def check_assets(root,manifest,expected_digest):
    require(isinstance(root,Path) and root.is_dir() and not root.is_symlink(),"ASSET_ROOT")
    require(type(manifest) is dict and set(manifest)=={"model","revision","files"}
            and manifest["model"] in CANDIDATES and type(manifest["revision"]) is str
            and re.fullmatch("[0-9a-f]{40}",manifest["revision"]),"ASSET_MANIFEST")
    require(type(expected_digest) is str and re.fullmatch("[0-9a-f]{64}",expected_digest)
        and hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":"),
                ensure_ascii=True).encode()).hexdigest()==expected_digest,"MANIFEST_PIN")
    fs=manifest["files"]
    require(type(fs) is dict and {"config.json","model.safetensors","tokenizer_config.json"}<=set(fs)
            and set(fs)<=ALLOWED and bool(set(fs)&{"tokenizer.json","spm.model","vocab.txt"}),"ASSET_FILE_SET")
    require(set(p.name for p in root.iterdir())==set(fs),"UNEXPECTED_ASSET")
    for name,h in fs.items():
        p=root/name
        require(type(h) is str and re.fullmatch("[0-9a-f]{64}",h) and
                p.is_file() and not p.is_symlink(),"ASSET_FILE")
        with p.open("rb") as f: actual=hashlib.file_digest(f,"sha256").hexdigest()
        require(actual==h,"ASSET_HASH")

def load_offline(root,manifest,expected_digest,*,authorized,device="cuda:0"):
    require(authorized is True,"PRIVATE_AUTHORIZATION_REQUIRED")
    check_assets(root,manifest,expected_digest)
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    require(device=="cpu" or device=="cuda:0" and torch.cuda.is_available(),"DEVICE")
    # Use explicit slow tokenizer pair construction; candidate smoke must pass
    # before any real rows are acquired. No silent fallback or remote code.
    tok=AutoTokenizer.from_pretrained(str(root),local_files_only=True,trust_remote_code=False,use_fast=False)
    model,info=AutoModelForSequenceClassification.from_pretrained(str(root),
            local_files_only=True,trust_remote_code=False,use_safetensors=True,
            output_loading_info=True)
    require(model.config.model_type=="deberta-v2","CANDIDATE_ARCHITECTURE")
    require(not any(info.get(k) for k in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs")),
            "MODEL_LOAD_MISMATCH")
    return Engine(tok,model,device)

class Engine:
    def __init__(self,tokenizer,model,device):
        self.tok=tokenizer;self.model=model.to(device).eval();self.device=device
        labels={int(k):str(v).lower() for k,v in model.config.id2label.items()}
        require(set(labels)=={0,1,2} and set(labels.values())==LABELS,"NLI_LABEL_MAP")
        self.labels=labels
        require(type(tokenizer.pad_token_id) is int and tokenizer.padding_side=="right" and
            model.config.max_position_embeddings>=MAX_LEN,"MODEL_SEQUENCE_CONTRACT")
        self.special=tokenizer.num_special_tokens_to_add(pair=True)
        require(1<=self.special<=8,"SPECIAL_TOKEN_CONTRACT")

    def tokens(self,text):
        require(type(text) is str and bool(text.strip()),"TEXT")
        ids=self.tok.encode(text,add_special_tokens=False,truncation=False,verbose=False)
        require(type(ids) is list and bool(ids) and all(type(x) is int and x>=0 for x in ids),"TOKENS")
        return ids

    def windows(self,text):
        ids=self.tokens(text);parts=[];start=0
        while start<len(ids):
            parts.append(ids[start:start+WINDOW])
            if start+WINDOW>=len(ids):break
            start+=WINDOW-OVERLAP
        require(len(parts)<=SLOTS,"CONTEXT_BUDGET_EXCEEDED")
        return parts,len(ids)

    def encode(self,premise,hypothesis):
        require(len(premise)<=WINDOW and len(hypothesis)<=MAX_LEN-WINDOW-self.special,
                "HYPOTHESIS_BUDGET_EXCEEDED")
        ids=self.tok.build_inputs_with_special_tokens(premise,hypothesis)
        require(len(ids)==len(premise)+len(hypothesis)+self.special and len(ids)<=MAX_LEN,
                "PAIR_TOKENIZATION")
        n=len(ids);pad=MAX_LEN-n
        out={"input_ids":ids+[self.tok.pad_token_id]*pad,"attention_mask":[1]*n+[0]*pad}
        if "token_type_ids" in self.tok.model_input_names:
            types=self.tok.create_token_type_ids_from_sequences(premise,hypothesis)
            require(len(types)==n,"TOKEN_TYPES")
            out["token_type_ids"]=types+[0]*pad
        return out

    def infer(self,parts,hypothesis,*,removed=False,batch_size=4):
        """Same fixed SLOTS x MAX_LEN padded tensor positions per answer/condition.
        Real window count and active tokens can differ; not equal wall-clock cost.
        Empty slots never influence full/permuted aggregation. For removal all
        premises empty and only first posterior is aggregated.
        """
        require(type(parts) is list and 1<=len(parts)<=SLOTS and type(removed) is bool
                and type(batch_size) is int and 1<=batch_size<=SLOTS,"INFERENCE_INPUT")
        hyp=self.tokens(hypothesis)
        real=1 if removed else len(parts)
        slots=[[] for _ in range(SLOTS)] if removed else parts+[[] for _ in range(SLOTS-len(parts))]
        enc=[self.encode(p,hyp) for p in slots]
        import torch
        post=[]
        with torch.inference_mode():
            for i in range(0,SLOTS,batch_size):
                batch=enc[i:i+batch_size]
                tensors={k:torch.tensor([row[k] for row in batch],device=self.device,dtype=torch.long)
                         for k in batch[0]}
                logits=self.model(**tensors).logits
                require(tuple(logits.shape)==(len(batch),3) and bool(torch.isfinite(logits).all()),"NLI_OUTPUT")
                for row in logits.float().softmax(dim=-1).cpu().tolist():
                    post.append({self.labels[j]:v for j,v in enumerate(row)})
        active=sum(sum(x["attention_mask"]) for x in enc)
        return aggregate(post[:real]),{"forward_examples":SLOTS,"padded_positions":SLOTS*MAX_LEN,
                "active_tokens":active,"real_windows":0 if removed else real}

def aggregate(rows):
    import math
    require(type(rows) is list and bool(rows),"NLI_ROWS")
    for x in rows:
        require(type(x) is dict and set(x)==LABELS and all(type(v) in (int,float) and
            math.isfinite(v) and 0<=v<=1 for v in x.values()) and abs(sum(x.values())-1)<1e-5,"NLI_PROBABILITY")
    e=[x["entailment"] for x in rows];c=[x["contradiction"] for x in rows]
    return [max(e),max(c),sum(e)/len(e),sum(c)/len(c),min(e),
            sum(x>=.5 for x in e)/len(e)]
