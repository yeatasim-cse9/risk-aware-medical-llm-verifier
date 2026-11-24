# Step 10.5A - Private Kaggle GPU Smoke Test

This step validates one isolated private Kaggle GPU notebook. No clinical datasets, patient records, MedVAL files or model weights are uploaded. The original CPU A/B checkpoints stay unchanged.

Source:
- scripts/gpu_smoke_runner.py: tiny 256x256 PyTorch matrix multiplication that requires CUDA (no CPU fallback)
- scripts/build_gpu_smoke.py: source SHA injection, metadata enforcing is_private=true, enable_gpu=true, enable_internet=false, empty input sources
- scripts/verify_gpu_smoke.py: independent SHA256/checkpoint/CUDA/source-provenance validation
- tests/test_gpu_smoke.py: offline no-GPU tests for metadata, tampering, CPU fallback, and wrong source commit

GitHub workflows:
- kaggle-gpu-smoke-readonly-preflight.yml: automatic first-push preflight and optional manual read-only status check; does not launch GPU
- kaggle-gpu-smoke.yml: strictly manual operation preflight, prepare, submit, status, verify

Separate target notebook: yeatasimbillah/medical-llm-gpu-smoke

To launch ONE GPU job, first inspect the current remaining GPU hours in Kaggle account UI. Run the manual controller using operation=submit, branch=main, confirm_gpu_quota=true and leave source_commit blank. This does consume some Kaggle GPU quota. A notebook already found at the expected slug blocks all duplicate submissions. The GitHub controller is serialized but cannot control unrelated manual uploads or concurrent tools.

After Kaggle completes, use operation=verify with source_commit set to the full 40-character GPU_SMOKE_FROZEN_SOURCE SHA printed in the successful submit run. This verifies the actual Kaggle persisted CUDA result, rather than treating successful submission as proof of GPU execution. A queued, failed, out-of-quota or CPU-fallback run will not pass verification.

The GPU script uses Kaggle's preinstalled torch (no pip or internet). It runs a tiny exact-value tensor calculation and saves checkpoints/gpu-smoke-result.json. It requires CUDA, records torch/GPU version information and checksums the result. It does not download datasets or contact third-party services.

Quota API accessibility is only a query, not evidence that GPU hours remain; user checks their account usage. This is a one-time manual smoke stage, not unattended GPU training or full thesis experiment automation. Do not submit restricted MedVAL-Bench to unauthorized Kaggle.
