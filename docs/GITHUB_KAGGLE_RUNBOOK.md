# GitHub → Kaggle CPU Pilot: safe operations

## What is already configured

- `Research Repository CI` runs Python unit tests on new `main` commits and pull requests.
- `Research CPU Smoke Test` runs manually.
- `Kaggle CPU Pilot Controller` is **manual-only** and starts with a credential-free `prepare` operation.
- The synthetic Kaggle worker is self-contained, uses no medical data, and executes up to **40 of 100** deterministic tasks per session.
- Two private CPU notebook slots are defined: `yeatasimbillah/medical-llm-cpu-auto-a` and `...-b`.
- **Automatic unattended re-launch is NOT deployed**. Kaggle API authentication, live preflight, first remote CPU-only launch, first saved output download and checkpoint validation have now been verified. Second-session remote resume remains unverified.

## First credential-free check

GitHub **Actions → Kaggle CPU Pilot Controller → Run workflow**.
Choose `operation=prepare`, `slot=a`, `resume=false`. Download the resulting `cpu-pilot-source-...` artifact to inspect `runner.py` and `kernel-metadata.json`. This builds but **does not submit** a Kaggle job.

## When ready for Kaggle API access

The account owner creates a Kaggle API token privately and stores it as the GitHub Actions repository secret named **`KAGGLE_API_TOKEN`**. Do not add it to a public repository, share it in chat, or print it in logs.
The `Kaggle CPU Pilot Controller` can then run:
- `operation=submit, slot=a, resume=false` to submit first CPU-only session;
- `operation=status, slot=a` to inspect Kaggle's status;
- `operation=collect, slot=a` only after a successful saved run, to validate and download synthetic checkpoint as a short-lived private GitHub Actions artifact.

For a second session only after inspecting slot A's output, select `submit, slot=b, resume=true`. The Kaggle metadata attaches A's output using `kernel_sources`. After success, alternate the slots to avoid self-dependency. Always check one source is attached and that both versions use the same `source_commit`; otherwise the worker fails closed. **Do not submit before the previous version has completed and its outputs are available.**

To reproduce a run after new commits, supply the *same 40-character source commit SHA* through the workflow's `source_commit` input.

## Optional notifications

If desired, add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` via GitHub Actions Secrets. The controller sends only status labels (submitted / checked / collected / failed), not medical content. Secrets are never placed in the repository. Missing Telegram secrets simply skip notifications.

## Limitations

Manual controller mode is intentionally a safety gate. It does not yet provide scheduled polling, quota-aware backoff, guaranteed zero-cost completion, automated A/B resubmission, or end-to-end remote verification. GitHub Actions can finish while Kaggle continues running, and public GitHub artifacts must never contain restricted datasets. Enable unattended re-launch only after a live CPU API test demonstrates reliable run-state detection, persistent checkpoint round-trip, quota compliance, and API error handling.


## Guardrail on remote submit

A live Kaggle status preflight now refuses an active or failed target slot. Resume also requires the opposite slot to be marked complete, downloads and integrity-checks its synthetic checkpoint, and confirms its immutable source commit matches the new submission. Unknown CLI statuses fail closed. The empty-result handling has **passed live read-only Kaggle API preflight**; duplicate/active-target guard remains fail-closed.

## Verified GitHub → Kaggle remote CPU proof (2026-11-25, Bangladesh local date)

- Source Git commit used for first remote kernel: `1a90191ae2734a74807eb53fa7d5b27a4e44c78f`
- Live Kaggle read-only preflight: GitHub Actions run `45471346800`, completed successfully.
- First remote Kaggle CPU-only private script submission: GitHub Actions run `45471589845`, Kaggle notebook `yeatasimbillah/medical-llm-cpu-auto-a`, **version 1 successfully pushed**.
- Read-only remote completion and output retrieval: GitHub Actions run `45471757482`, **complete**, checkpoint `cpu-pilot-automation.json` validated at **40 / 100** simulated task IDs, with checksum and exact source commit match.
- This validates initial remote job launch and persistent Kaggle saved output. **It does not yet validate cross-session remote resume, automated restart, Telegram messages, or actual medical experiments.**

### IMPORTANT: frozen commit for next manual resumed session

When testing slot B after successful slot A, run **GitHub Actions → Kaggle CPU Pilot Controller → Run workflow**:
- `operation = submit`
- `slot = b`
- `resume = true`
- `source_commit = 1a90191ae2734a74807eb53fa7d5b27a4e44c78f`

Do **not** leave `source_commit` empty on this particular resume test: a newer HEAD commit would cause intentional checkpoint provenance mismatch. The controller should preflight B as absent, verify A complete, download the A checkpoint, and attach A as an input to B. Never send medical records through this pilot.

Do not repeat the first-launch one-time workflow. It was removed after the successful run to prevent accidental duplicate submissions.

## Verified B output and next A resume

- Slot B run was successful and the 80-task saved checkpoint was independently downloaded and checksum-validated in GitHub Actions run `45473523954`.
- Last A session should use `operation=submit`, `slot=a`, `resume=true` and `source_commit=1a90191ae2734a74807eb53fa7d5b27a4e44c78f`.
- A already exists (version 1 at 40/100); the controller allows a version update only after checking that it has completed. The new Kaggle A version must take B's saved checkpoint as input and reach exactly 100/100, then GitHub should independently verify output.
- Do not substitute the current GitHub main commit for the frozen pilot commit, and never upload real restricted datasets into this synthetic pilot.

## Final verified three-session CPU pilot — stop submitting

The 3-session synthetic checkpoint test is **complete**:
- A v1 (40 tasks): GitHub verification `45471757482`.
- B v1 (80 tasks): GitHub verification `45473523954`.
- A v2 (100 tasks): GitHub verification `45474169382`.

The final checkpoint integrity, 100 unique IDs, correct source commit and completed status were verified independently. **Do not keep submitting more A/B versions** merely to repeat this finished pilot. Use the next infrastructure phase to implement monitored, quota-aware scheduling, error handling, and Telegram status notifications before declaring unattended operation ready.

These were **manually triggered submission stages**. The laptop did not execute Kaggle tasks, but automatic chaining without manual clicks is not yet implemented.
