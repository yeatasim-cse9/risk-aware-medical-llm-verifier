# Verified live infrastructure milestones

This file records **observed** cloud events, not claims of completed thesis experiments.

| Verification | GitHub Actions run | Observed outcome |
| --- | --- | --- |
| CPU smoke test | 45460024701 | Success |
| Kaggle API auth | 45467404272 | CLI authenticated successfully |
| Kaggle read-only preflight | 45471346800 | Remote slot A reported missing; success |
| First private CPU-only script submission | 45471589845 | Kernel version 1 successfully pushed |
| Kaggle remote execution and saved output | 45471757482 | Remote status COMPLETE; checkpoint downloaded and validated |

Pilot evidence:
- Notebook: `yeatasimbillah/medical-llm-cpu-auto-a`, version 1
- Frozen source commit: `1a90191ae2734a74807eb53fa7d5b27a4e44c78f`
- Task count: **40 / 100** synthetic tasks
- Verified saved status: `in_progress` (because more synthetic tasks remain)
- Synthetic checkpoint path: `checkpoints/cpu-pilot-automation.json`
- No restricted medical data, patient text, GPU, or paid LLM API used

Still open:
1. Remote A→B resume 40→80; B→A resume 80→100 and checkpoint integrity validation
2. Scheduled polling, quota-aware pause/backoff, safe automatic relaunch and error recovery
3. Telegram secret setup, notification delivery and monitoring
4. Real proposal datasets, evidence sources, E1–E7 models and statistical metrics

The test does not guarantee every future Kaggle run will be available, free or uninterrupted.
