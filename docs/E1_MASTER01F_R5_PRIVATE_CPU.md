# MASTER 01F-R5 — private CPU development preparation

Supervisor scope approval is user-confirmed in the conversation, not a
separately signed institutional approval. Applicable processing rights
and private runtime authorization remain external gates. Do not invoke
prepare until the operator confirms both; no actual approval is created.

Reviewed official terms https://clinicaltrials.gov/terms-conditions
(2023-01-31 update): international copyright and third-party content
restrictions apply; retain attribution, provider processing date,
modification description and as-of context. No broad public-domain
claim. Clarify archival snapshot versus current clinical information.
Use only minimal structured study facts and linkage identifiers, no
narrative, contacts or patient data. Institutional/legal review remains
necessary if applicable. Live version observed 2.0.5, provider timestamp
2026-11-25T10:00:05 via /api/v2/version; recheck at execution. Complete
API field metadata was unavailable to the web review. Actual projection,
pagination and Kaggle runtime have not been tested; fail closed on schema
drift, do not relax projection or use full-record fallback.

Fixed prospective development frame: studies first posted 2026-10-01
through 2026-10-07 inclusive; all frame IDs are development-exposed,
including unselected. Retrieve only ID, alias, organization/secondary
IDs and first-post dates for grouping. Connected exact ID proxies,
one representative per group by fixed hash rank, 128 selected. Unknown
links and unrelated ID collisions limit independence; no sample-size
or representativeness claim. Later holdouts must use independently
qualified distinct frames and exclude all exposed groups. Only selected
IDs are requested for answer-bearing fields, max 16 per request.
Check provider version/timestamp before and after acquisition; not a
transactional snapshot guarantee. No historical dataset access, no
future test or calibration allocation/access. If API rejects any fields
or returns unapproved fields, stop rather than broaden requests.

One standard-library script, no GPU or model packages. Use an immutable
reviewed commit in private Kaggle, accelerator None, internet enabled
for this API only, no concurrent writers, no saved raw outputs.
Operator creates an owner-only directory below /kaggle/temp (0700).
Metadata operation is public version only. The prepare operation
requires explicit operator confirmation of rights and CPU permission;
a flag cannot establish rights. Do not invoke in public CI.

Development artifacts: frame, selected IDs, source facts, questions,
separate references, report, hashes; private ephemeral only. Exact
typed independent scorer requires correct NCT ID and field value;
enrollment type is part of the answer. Deterministic lookup is only
a scorer sanity check, not evidence of LLM performance or reduction.
Actual LLM/generator selection, exact weights and usage rights, inference
runtime and frozen prompts remain to be qualified for the subsequent
experiment. Do not report a lookup self-check as empirical improvement.
No private raw outputs in public GitHub, no Google Drive, no GPU.

Official field mapping: https://clinicaltrials.gov/data-about-studies/csv-download
Official enrollment meaning: https://clinicaltrials.gov/policy/faq
API guidance: https://clinicaltrials.gov/data-api/how-download-study-records
