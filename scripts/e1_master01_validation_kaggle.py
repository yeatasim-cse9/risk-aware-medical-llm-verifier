"""MASTER 01: private GPU-assisted screening, never scientific E1 training.
Only aggregate output. Official holdout used solely for fixed contamination screen.
"""
from __future__ import annotations
import collections
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
import urllib.parse
import urllib.request

BASE = "c6e0f3637c9e7f753091e874f53a22ed2cc45f26"
REPO = "yeatasim-cse9/risk-aware-medical-llm-verifier"
EXPECTED_MAPPING = "9fa4a8d68d695c20447f25cb26c246f9b35ff355955850f9b26acb79197e1e2c"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REV = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
SEED = 20261125
THRESHOLDS = (0.80, 0.85, 0.90, 0.95)
SPLITS = ("train", "dev", "test", "official_holdout")
PINS = {
    "scripts/e1_pubmedqa_lineage_kaggle.py": "0a0501e85218f37d125467b23b13194e15c540a70ed6e3d5d8ab6d20345d8f40",
    "src/research_data/e1_pubmedqa_test_exclusion.py": "14cb454255ac1b9e09413742f0432da8382c596bccb3ee8a533c7342bfbacae6",
    "src/research_data/e1_advanced_lineage.py": "7c630df817118462790839c7314c929dc2b568a4db5a41dc07180cf6c39cacf2",
    "src/research_data/e1_candidate_splits.py": "7f4c8aac27a7b682346cc5c8df92cbceb62cb7821bbda0257931b9fbf545cd4a",
    "src/research_data/medhallu_native_audit.py": "a23b2b916c8165e4475bebe5c80a8d2aba39eee5494ef0514da234da98b1b30b",
}
STAGE = "BOOT"
NOTIFY = []
START = time.monotonic()

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()

def require(ok, code):
    if not ok:
        raise RuntimeError(code)

def notify(event):
    # Never transmit row-level data or exceptions. Failure is nonfatal and explicit.
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        token = secrets.get_secret("TELEGRAM_BOT_TOKEN")
        chat = secrets.get_secret("TELEGRAM_CHAT_ID")
        body = urllib.parse.urlencode({"chat_id": chat, "text": "E1 MASTER 01 | " + event}).encode()
        req = urllib.request.Request("https://api.telegram.org/bot" + token + "/sendMessage", data=body)
        with urllib.request.urlopen(req, timeout=10) as r:
            ok = bool(json.loads(r.read(10000)).get("ok"))
    except Exception:
        ok = False
    NOTIFY.append({"event": event, "delivered": ok})
    print("TELEGRAM_" + ("OK" if ok else "UNAVAILABLE") + "=" + event, flush=True)

def stage(name):
    global STAGE
    STAGE = name
    print("MASTER01_STAGE=" + name, flush=True)
    notify(name)

def download_module(path):
    req = urllib.request.Request("https://raw.githubusercontent.com/" + REPO + "/" + BASE + "/" + path,
                                 headers={"User-Agent":"private-e1-master01"})
    with urllib.request.urlopen(req, timeout=60) as r:
        b = r.read(100000)
    require(hashlib.sha256(b).hexdigest() == PINS[path], "PINNED_MODULE_HASH_MISMATCH")
    return b

def grouped_diagnostics(med, assignment):
    # No names, medical content or row IDs returned.
    groups = {}
    for config in sorted(med):
        groups[config] = {}
        for split in SPLITS[:3]:
            rows = [med[c][i] for (c,i), s in assignment.items() if c == config and s == split]
            categories = collections.Counter(r["Category of Hallucination"] for r in rows)
            difficulties = collections.Counter(r["Difficulty Level"] for r in rows)
            groups[config][split] = {
                "parents": len(rows), "benchmark_role_0": len(rows), "benchmark_role_1": len(rows),
                "category_frequency_sorted": sorted(categories.values(), reverse=True),
                "difficulty_frequency_sorted": sorted(difficulties.values(), reverse=True),
                "min_observed_category_count": min(categories.values(), default=0),
            }
    return groups

def overlap(a, b, block=256, thresholds=THRESHOLDS):
    """All cross-set vector pairs; no row identifiers returned. CPU or GPU tensors."""
    import torch
    require(a.ndim == b.ndim == 2 and a.shape[1] == b.shape[1] and len(a) and len(b),
            "INVALID_EMBEDDING_MATRIX")
    counts = [0] * len(thresholds)
    left = [0] * len(thresholds)
    right = [torch.zeros(len(b), dtype=torch.bool, device=b.device) for _ in thresholds]
    maximum = -1.0
    for i in range(0, len(a), block):
        sim = a[i:i+block] @ b.T
        require(bool(torch.isfinite(sim).all().item()), "NONFINITE_SIMILARITY")
        maximum = max(maximum, float(sim.max().item()))
        for k,t in enumerate(thresholds):
            mask = sim >= t
            counts[k] += int(mask.sum().item())
            left[k] += int(mask.any(dim=1).sum().item())
            right[k] |= mask.any(dim=0)
    return {"pairs_compared": len(a)*len(b), "maximum_cosine": min(1.0, round(maximum, 7)),
            "thresholds": {str(t): {"pairs": counts[k], "left_rows": left[k],
                         "right_rows": int(right[k].sum().item())} for k,t in enumerate(thresholds)}}

def encode(texts, tok, model, device, *, max_length=256):
    import torch
    result = []
    truncated = 0
    with torch.inference_mode():
        for i in range(0, len(texts), 64):
            batch = texts[i:i+64]
            lengths = tok(batch, add_special_tokens=True, truncation=False,
                          return_length=True, verbose=False)["length"]
            truncated += sum(n > max_length for n in lengths)
            encoded = tok(batch, padding=True, truncation=True, max_length=max_length,
                          return_tensors="pt").to(device)
            hidden = model(**encoded).last_hidden_state.float()
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
            result.append(torch.nn.functional.normalize(pooled, p=2, dim=1))
    return torch.cat(result), truncated

def scan(sets, tok, model, device):
    # Independent question and abstract-prefix views; truncation disclosed.
    output = {}
    for view in ("question", "context"):
        vectors = {}
        truncated = {}
        for s in SPLITS:
            v, n = encode([r[view] for r in sets[s]], tok, model, device)
            vectors[s] = v
            truncated[s] = n
        output[view] = {"truncated_rows": truncated, "comparisons": {}}
        for i,s in enumerate(SPLITS):
            for t in SPLITS[i+1:]:
                output[view]["comparisons"][s+"__"+t] = overlap(vectors[s], vectors[t])
        del vectors
    return output

def run():
    require(os.environ.get("MASTER01_PRIVATE_RIGHTS_CONFIRMED") == "YES",
            "PRIVATE_DATA_RIGHTS_CONFIRMATION_REQUIRED")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import torch
    from transformers import AutoTokenizer, AutoModel
    require(torch.cuda.is_available(), "KAGGLE_GPU_REQUIRED")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    scratch = Path("/kaggle/temp")
    work = Path("/kaggle/working")
    require(work.is_dir() and not work.is_symlink() and not scratch.is_symlink(),
            "PRIVATE_KAGGLE_STORAGE_REQUIRED")
    scratch.mkdir(mode=0o700, exist_ok=True)
    output = work / "master01_e1_validation_v1.json"
    require(not output.exists() and not output.is_symlink(), "REPORT_EXISTS_DO_NOT_OVERWRITE")
    with tempfile.TemporaryDirectory(dir=scratch, prefix="master01-") as tmp:
        root = Path(tmp)
        os.environ["HF_HOME"] = str(root/"hf")
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        stage("PINNED_CODE_AND_DATA")
        pkg = root/"research_data"
        pkg.mkdir()
        (pkg/"__init__.py").write_text("")
        for path in PINS:
            if path.startswith("src/"):
                (pkg/Path(path).name).write_bytes(download_module(path))
        sys.path.insert(0,str(root))
        helper_file = root/"helper.py"
        helper_file.write_bytes(download_module("scripts/e1_pubmedqa_lineage_kaggle.py"))
        spec = importlib.util.spec_from_file_location("master01_helper", helper_file)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        candidate = importlib.import_module("research_data.e1_candidate_splits")
        advanced = importlib.import_module("research_data.e1_advanced_lineage")
        hard = importlib.import_module("research_data.e1_pubmedqa_test_exclusion")
        snap,pq = helper.libraries()
        med = helper.checked_parquet(snap,pq,helper.MEDHALLU,helper.MEDHALLU_REVISION,
                                    helper.SOURCE_SHA256,root/"med",is_source=False)
        pub = helper.checked_parquet(snap,pq,helper.PUBMEDQA,helper.PUBMEDQA_REVISION,
                                    helper.PUBMEDQA_SHA256,root/"pub",is_source=True)
        ids, holdout_sha = helper.official_test_ids()
        require(holdout_sha == "939fe566f09017d13b1ca64d2ddfee0bc2374b366048152997669cccedc44d51",
                "HOLDOUT_HASH_MISMATCH")
        stage("RECONSTRUCT_PRIVATE_MAPPING")
        prior,old = candidate.candidate_plan(med)
        require(prior["candidate_assignment_sha256"] == helper.CANDIDATE_ASSIGNMENT_SHA256,
                "OLD_MAPPING_CHANGED")
        index = advanced.pubmed_index(pub)
        summary,assignment = hard.candidate_remediation(med,old,index,ids,seed=SEED)
        hard.verify_summary(summary)
        require(summary["private_mapping_sha256"] == EXPECTED_MAPPING and len(assignment)==9474,
                "REMEDIATED_MAPPING_CHANGED")
        stage("SOURCE_AND_LABEL_DIAGNOSTICS")
        diagnostics = grouped_diagnostics(med,assignment)
        sets = {s:[] for s in SPLITS}
        for (c,i),s in sorted(assignment.items()):
            row = med[c][i]
            sets[s].append({"question": row["Question"], "context": " ".join(row["Knowledge"])})
        # Official IDs only; no official answer labels enter the screen.
        held = {}
        for c in sorted(pub):
            for row in pub[c]:
                pid = str(row["pubid"])
                if pid in ids:
                    ctx = row["context"]
                    parts = ctx.get("contexts",[]) if isinstance(ctx,dict) else ctx
                    item = {"question":row["question"],"context":" ".join(parts)}
                    if pid in held:
                        require(held[pid] == item,"AMBIGUOUS_OFFICIAL_HOLDOUT_TEXT")
                    held[pid] = item
        require(set(held)==ids,"OFFICIAL_HOLDOUT_TEXT_NOT_COMPLETE")
        sets["official_holdout"] = [held[p] for p in sorted(held)]
        del pub,index,held,med,old,assignment,ids
        stage("GPU_SEMANTIC_SCREEN")
        tok = AutoTokenizer.from_pretrained(MODEL,revision=MODEL_REV,trust_remote_code=False,
                                             cache_dir=str(root/"model"))
        model = AutoModel.from_pretrained(MODEL,revision=MODEL_REV,trust_remote_code=False,
                     use_safetensors=True,cache_dir=str(root/"model")).to(device).eval()
        semantic = scan(sets,tok,model,device)
        del sets,model,tok
        stage("SANITIZED_REPORT")
        report = {"schema_version":1,"status":"SCREEN_COMPLETE_PENDING_SCIENTIFIC_REVIEW",
            "source_commit":BASE,"master_runner_commit":os.environ.get("MASTER01_RUNNER_COMMIT","UNSET"),
            "pinned_module_sha256":PINS,"mapping_sha256":EXPECTED_MAPPING,
            "medhallu_revision":helper.MEDHALLU_REVISION,"pubmedqa_revision":helper.PUBMEDQA_REVISION,
            "source_sha256":{"medhallu":{k:v[2] for k,v in helper.SOURCE_SHA256.items()},
                             "pubmedqa":{k:v[2] for k,v in helper.PUBMEDQA_SHA256.items()}},
            "official_500_sha256":holdout_sha,"remediation_summary":summary,
            "model":MODEL,"model_revision":MODEL_REV,"seed":SEED,"device":torch.cuda.get_device_name(0),
            "versions":{p:importlib.metadata.version(p) for p in
                        ("torch","transformers","huggingface_hub","pyarrow")},
            "diagnostics":diagnostics,"semantic_screen":semantic,
            "limitations":["generic encoder not validated for clinical equivalence",
              "256-token prefix truncation; not full abstract semantic search",
              "thresholds are screening heuristics, not proof of semantic equivalence",
              "no article-level external identity validation or clinician adjudication",
              "official holdout comparisons restricted to contamination screening"],
            "scientific_binary_labels_adjudicated":False,"source_lineage_fully_certified":False,
            "complete_semantic_leakage_excluded":False,"final_split_approved":False,
            "scientific_e1_experiment_executed":False,"private_row_data_exported":False,
            "notifications_before_finish":NOTIFY.copy()}
        report["integrity_sha256"] = hashlib.sha256(canonical(report)).hexdigest()
        payload = json.dumps(report,indent=2,sort_keys=True)
        require(len(payload.encode())<60000,"REPORT_SIZE_POLICY")
        with output.open("x",encoding="utf-8") as f:
            f.write(payload+"\n")
    notify("FINISH_SCREEN_COMPLETE_REVIEW_REQUIRED")
    print("MASTER01_SAFE_REPORT_SAVED="+str(output))
    print("MASTER01_FINAL_STATUS=BLOCKED_PENDING_SCIENTIFIC_APPROVAL")

if __name__ == "__main__":
    try:
        notify("START")
        run()
    except Exception:
        notify("FAILED_"+STAGE)
        print("MASTER01_FAILED_STAGE="+STAGE,flush=True)
        print("MASTER01_FAILED_CODE=SAFE_FAILURE_REQUIRES_REVIEW",flush=True)
        raise SystemExit(1)
