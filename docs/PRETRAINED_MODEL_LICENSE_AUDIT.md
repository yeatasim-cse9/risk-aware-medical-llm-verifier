# Step 10.5C — Pretrained model selection & declared-license audit

Audit date: 2026-11-25 (UTC). Scope: publicly accessible official Hugging Face model cards and Microsoft MIT LICENSE file; **metadata-only review**. This is not a complete legal clearance, not a real GPU/VRAM benchmark, and not permission to redistribute weight files.

## Shortlist

| Candidate | Planned role | Model-card declared license | What the card actually supports | Key limitation |
| --- | --- | --- | --- | --- |
| `sentence-transformers/all-MiniLM-L6-v2` | Dense embedding retrieval *baseline* | Apache-2.0 | Sentence/paragraph embeddings, 384-dimensional output; card warns inputs longer than 256 wordpieces truncate | Not biomedical evidence entailment; general-domain training and corpus-specific retrieval must be evaluated |
| `cross-encoder/nli-MiniLM2-L6-H768` | Pairwise three-way NLI *baseline* | Apache-2.0 | SNLI + MultiNLI fine-tuning; scores ordered [contradiction, entailment, neutral] | General-domain NLI, no established medical calibration; neutral does not equal the thesis' unsupported-claim gold label without a mapping protocol |
| `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` | Biomedical encoder and domain-adaptation research *candidate* | MIT, with published LICENSE.md | Biomedical BERT pretrained on PubMed abstracts; documented task is masked-token prediction | Not a standalone medical NLI classifier, retrieval-optimized embedding model or medical generative LLM |

Official model-card and license-evidence URLs:

- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768
- https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768/blob/main/README.md
- https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
- https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract/blob/main/LICENSE.md
- Hugging Face offline, pinned-revision guidance: https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache

## Selection decision

1. **First retrieval baseline**: all-MiniLM-L6-v2 for semantic matching, benchmark only against appropriately licensed evidence. Report Recall@k, MRR, passage attribution and mismatch cases. Use no MedVAL on Kaggle.
2. **First 3-way NLI baseline**: nli-MiniLM2-L6-H768 with a fixed, tested mapping `[0 contradiction, 1 entailment, 2 neutral]`. Evidence passage is the premise; claim is the hypothesis. Log all raw class probabilities before any research-specific threshold. Do NOT treat neutral as an automatic equivalent of an unsupported medical fact.
3. **Biomedical encoder follow-up**: BiomedBERT for domain comparison, not a drop-in 3-class NLI head. A downstream supervised head would require separately licensed training data, approved environment and held-out evaluation.

## License and data-policy boundary

These are **upstream declared licenses**. Apache-2.0 and MIT are compatible with many academic research workflows, but model redistribution/attribution/notice obligations, base checkpoints, source training-data terms, user-specific usage restrictions and any downstream derivative weights must be checked separately before downloading or transferring. The license metadata alone is NOT a blanket permission to upload model files to Kaggle, publish them on GitHub or process restricted third-party data.

- Never put restricted MedVAL-Bench text, physician labels, patient records or credentialed PhysioNet materials into public GitHub, Kaggle, Telegram or unaudited external model services.
- Any planned public MedHallu or BioASQ dataset needs its own access and redistribution audit first; their presence in a thesis plan is not evidence of license clearance.
- Use the public GitHub repository **only for code, non-sensitive audit metadata and aggregate permitted reports**. Weight files should be placed in an approved private input store after an explicit license decision, never in the public repository.
- For immutable reproduction, freeze the upstream Hugging Face commit SHA, exact tokenizer revision, model/config hashes, dependency lockfile, device, dtype, batch size, token limits and input dataset identifiers. Do not use moving `main` as experimental provenance.
- Prior Kaggle jobs deliberately set `enable_internet=false`. A new model pilot needs an explicitly reviewed, source-pinned offline model bundle or approved private input; **do not silently turn on internet**.

## GPU feasibility: evidence and unknowns

Earlier Kaggle synthetic jobs proved CUDA operation and a randomly initialized toy Transformer restore. **No tested candidate pretrained weight has been downloaded or executed in the user's Kaggle GPU account.** Current usable GPU model, VRAM, quota hours, CPU offload overhead and peak CUDA memory for each candidate remain unmeasured. Avoid making fit/throughput claims until a new isolated pilot records `torch.cuda.get_device_properties()`, weight-file sizes, `torch.cuda.max_memory_allocated()` and batch/sequence dimensions.

Do not assume that a small parameter count alone guarantees fitting into a given accelerator after tokenization, runtime overhead and other allocations. Start with tiny synthetic non-medical text pairs, batch size 1 and short token lengths; use `torch.inference_mode()` and fail rather than silently falling back to CPU.

## What must happen before any real pretrained GPU job

1. Recheck current official model cards and actual license files, attribution/NOTICE obligations and external training-data restrictions; note legal uncertainty where present.
2. Resolve and pin one exact upstream model & tokenizer commit; verify selected file manifests and individual SHA-256 checksums; prefer safetensors. Do not load untrusted pickle checkpoints when not necessary.
3. Confirm allowed model transfer route into a **private** Kaggle input or other approved environment. Document all repository/model license obligations and forbid restricted MedVAL material.
4. Get a fresh GPU quota check from the Kaggle account, then build a separate private notebook with user consent. The model must not auto-download or expose credentials.
5. Validate CUDA inference, peak VRAM, latency, tokenizer behavior, class mapping and the returned integrity-checked checkpoint using **synthetic nonclinical text**. Only then plan licensed, fully frozen scientific dataset evaluation.

## Repository verification

- Registry: `configs/pretrained-model-audit.json`. All `model_revision_sha`, binary hash, size and actual peak VRAM fields are **null**, because they were not independently measured or frozen.
- Safety gate: `scripts/audit_pretrained_models.py` rejects changed model IDs, swapped NLI labels, forged model revisions, premature download/approval flags, and any relaxation of the restricted-data boundary.
- Offline tests: `tests/test_pretrained_model_audit.py`. No imports of transformers, torch or network clients; all checks run on ordinary GitHub CPU without GPU time.
- Workflow: `.github/workflows/pretrained-model-license-audit.yml`, manual and one-shot push-on-change; `contents:read`, no Kaggle/Telegram/Hugging Face token and no external data transfers.

**Audit decision:** three candidates documented, all downstream deployment gates remain CLOSED. The next action is *one* separately approved, provenance-pinned model package feasibility review—not a real MedVAL or thesis experiment.
