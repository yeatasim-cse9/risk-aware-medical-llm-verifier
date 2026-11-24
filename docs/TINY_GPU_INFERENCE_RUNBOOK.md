# Step 10.5B — Private tiny GPU Transformer inference pilot

## Purpose and limits

This is an infrastructure pilot using a tiny **randomly initialized, UNTRAINED** PyTorch Transformer encoder. It is not a pretrained medical model or medical LLM. The predictions are synthetic and have NO clinical meaning. No real medical data, external weights, downloads, or restricted MedVAL-Bench materials are used.

One CUDA inference runs on 4 synthetic sequences of 8 integer token IDs. The model contains an embedding layer, one Transformer attention layer, and a 3-class head. Its outputs and a saved-and-reloaded model state are checked.

## Privacy and reproducibility

- Separate Kaggle notebook: yeatasimbillah/medical-llm-gpu-inference-pilot.
- Metadata: is_private=true, enable_gpu=true, enable_internet=false, enable_tpu=false. Dataset, model, kernel and competition sources are all empty.
- The CPU A/B notebooks and previous GPU smoke test remain untouched.
- Source code is frozen to an exact 40-character Git commit SHA at submission.
- Kaggle checkpoint files: checkpoints/tiny-transformer-weights.pt and checkpoints/tiny-transformer-inference.json.
- Kaggle re-creates the tiny model with the saved state dict and compares CUDA inference logits. The manifest stores weight SHA256, source SHA, logits, probabilities, predictions, and a manifest SHA256.
- GitHub's standalone verifier checks file hashes, output dimensions, finite probabilities, softmax consistency and source provenance without deserializing a potentially untrusted PyTorch checkpoint.

## GitHub workflows

- Manual consent-gated control: .github/workflows/kaggle-tiny-gpu-inference.yml
- Automatic first-push read-only preflight: .github/workflows/kaggle-tiny-gpu-inference-preflight.yml
- Control operations: preflight, prepare, submit, status, verify.

After read-only preflight, a user may approve ONE GPU notebook by running operation=submit on main with confirm_gpu_quota=true and source_commit blank. This consumes finite Kaggle accelerator time. The user must check GPU quota in Kaggle UI because success of 'kaggle quota' does not measure remaining allowance in this workflow.

For later verification, use operation=verify and supply the full TINY_GPU_INFERENCE_FROZEN_SOURCE SHA shown in the submission run. Do not re-submit if the notebook exists, even when a previous execution has failed. The GitHub workflow's concurrency group does not provide a global exactly-once Kaggle submission guarantee.

## Scientific boundary

Passing this test proves only that a small randomly initialized Transformer can perform CUDA inference, restore saved weights and produce verified non-sensitive outputs. It does not test model accuracy, pretrained checkpoints, tokenization of medical text, evidence retrieval, clinical hallucination detection, E1–E7, or unrestricted GPU availability. Those are separate, license-audited research stages.
