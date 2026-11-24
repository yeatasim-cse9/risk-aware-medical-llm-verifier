# Step 10.5D — First pretrained model provenance and theoretical GPU feasibility

**Candidate:** sentence-transformers/all-MiniLM-L6-v2, general-domain embedding retrieval baseline. Not a medical NLI classifier or clinical model.

This phase queries public Hugging Face model and pinned file-tree **metadata only**. It does not fetch model weights, tokenizer bytes, clinical data, GPU time or private credentials.

Method:

1. Read official model card and the Hugging Face revision API for the exact revision SHA (not moving main). Require public, non-gated, Apache-2.0 license metadata and a sentence-similarity model.
2. Fetch immutable file-tree metadata at the returned revision and inspect specific required file entries: safetensors model, tokenizer, vocab, configuration, pooling module. Record advertised byte sizes; record LFS SHA-256 only when exposed as metadata. **Metadata SHA alone does not verify actual downloaded bytes.**
3. Record Hub-reported safetensors parameter count and derive FP32/FP16 parameter-storage lower bounds as n*4 and n*2 bytes. **This is not GPU peak VRAM**, does not include activations, attention temporaries, padding, CUDA context, runtime caches, optimizer and allocator overhead.
4. Run conservative no-network fixture tests for wrong model IDs, changed licenses, unpinned revision, missing files, fake sizes, corrupt LFS fields and attempted unauthorized data transfers.
5. Keep all application/experiment permissions closed. Do not upload MedVAL, private records, pretrained weight binaries or datasets. Do not spend quota or start a Kaggle job.

A later phase must freeze upstream model AND tokenizer byte-level SHA-256; inspect full Apache license and any NOTICE obligations, training-data and redistribution terms; stage the approved model in a private store with internet off in Kaggle; measure actual free VRAM, batch size, inference precision, throughput and peak allocations. A separate explicit GPU approval is required.

Relevant documentation:
- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api

**Status:** repository metadata-only preflight. No scientific E1-E7 results have been produced.
