# Billing Log Data Dictionary — `openrouter_billing_log.csv`

Companion note for the CrucibleBench Run 2 billing export (43,372 rows, exported March 11, 2026). Read alongside Appendix E of the whitepaper, which documents the reconciliation methodology and its collision bounds.

## The `run2_status` column

The column tags rows by a **model-slug rule**, not by run attribution:

| `run2_status` value | Rows | Meaning |
|---|---|---|
| `run2_keep` | 27,883 | Primary model calls matched to the 650 scored Run 2 runs |
| `classifier` | 8,284 | **All** Gemini 3.1 Flash Lite rows, regardless of which run they served |
| `discard` | 7,205 | Primary model calls from discarded calibration attempts |

## Reconciling with the paper's counts

Appendix E reports `run2_keep (27,883)`, `classifier (6,962)`, `discard (8,527)`. Both classifications sum to 43,372 and both are correct; they differ only in where calibration-era classifier calls are counted:

- The paper's `classifier (6,962)` counts only Flash Lite calls tied to scored Run 2 dialogue turns. This figure is derived from the per-run classifier call counts in the released run JSONs (`classifier.calls`, summing to 6,962 across the 650 runs) and matches the paper's reported classifier spend ($0.74).
- The remaining **1,322** Flash Lite rows served discarded calibration attempts; the paper folds them into `discard` (7,205 + 1,322 = 8,527).
- Per-row attribution of individual Flash Lite calls to scored vs. calibration runs is **not preserved** in the log (the run JSONs store classifier usage as per-run aggregates), which is why the CSV column uses the coarser slug rule. Calibration and Run 2 activity interleave in time on March 10, so timestamps alone do not separate them.

## Verification notes (reproduced independently from the released artifacts)

- Per-model billed totals in Table 7 reproduce from this CSV to the cent; Grok 4 accounts for 42.2% of model spend.
- Fingerprint ambiguity is bounded: 97.9% of `run2_keep` rows carry a (model, prompt-token, completion-token) fingerprint occurring among no same-model discarded row; the 590 colliding rows total $1.03 (~1% of model spend), the worst-case misallocation.
- Turn denominators reconcile exactly: the run JSONs log 27,884 turns, of which 5 were API-error fallbacks with no model output; the paper's parse-reliability denominator (27,879) excludes those five, and this log retains 27,883 billed rows (±1 per-model differences reflect provider-side retries).

## Suggested Zenodo record note (paste-ready)

> The billing export's `run2_status` column tags all 8,284 Gemini Flash Lite rows as `classifier` by model slug. The whitepaper's Appendix E attributes 1,322 of these (calibration-era calls) to `discard`, yielding its reported 6,962/8,527 split; the 6,962 figure derives from per-run classifier call counts in the released run JSONs. Both classifications sum to 43,372 rows and are consistent; see Appendix E for the reconciliation methodology and its collision bounds.
