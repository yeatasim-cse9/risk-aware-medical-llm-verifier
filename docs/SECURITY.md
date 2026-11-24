# Security and restricted data boundary

This repository is **public**. Only code, sanitized configuration, simulated pilot artifacts, and approved aggregate findings may be published.

- **Never commit** Kaggle credentials, Telegram bot token/chat identifiers, service keys, personal medical information, raw clinical text, MedVAL-Bench records/annotations, or unauthorized copies of licensed corpora.
- The `.gitignore` is a safeguard, **not** a complete control; review each `git add` and `git diff --cached` before pushing.
- The Kaggle CPU pilot intentionally uses **synthetic integer task IDs only**, with **private notebook** metadata, GPU off, internet off.
- Restricted MedVAL-Bench v1.0.1 remains in an approved environment. PhysioNet credentialing, DUA, approved storage, and third-party processing restrictions apply before any use.
- Do not transfer MedVAL data to Kaggle, GitHub Actions, Telegram, or hosted LLMs without explicit documented permission.
- Only setup secrets in GitHub **Settings → Secrets and variables → Actions → New repository secret**. Never paste tokens into a chat, issue, repository file, or workflow log.
- For external services, use `KAGGLE_API_TOKEN`, `TELEGRAM_BOT_TOKEN`, and `TELEGRAM_CHAT_ID`. Bot notification content is operational status only.
- GitHub workflows use read-only repository permissions and manual opt-in for external API calls. PR testing must not receive service secrets.
- Output collected by GitHub Actions is allowed **only for this synthetic pilot**. Do not repurpose the collector for restricted research outputs.
- Token rotation and deletion are manual account-owner responsibilities. Report accidental secret exposure and revoke credentials immediately.

No deployed code here should be interpreted as medical advice or a clinical decision support tool.
