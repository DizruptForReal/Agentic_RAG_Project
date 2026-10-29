# Document corpus and domain

## Domain: product warranty claims for an industrial pump maker

The corpus belongs to **Aquaflow Pumps Ltd**, a fictional manufacturer of water pumps (all companies, people,
serial numbers and addresses are invented). The investigation the system is built around is a **warranty claim
assessment**: *is a given failure of a given unit covered?*

This domain was chosen because a warranty decision naturally needs everything the assignment asks for:

| Need in the assignment | Why warranty claims need it |
|---|---|
| Evidence from several documents | The answer combines a policy, its amendment, a product manual and the customer's contract |
| Contradictory information | The policy says 24 months; a later amendment says 12 months for commercial units |
| Hard-to-extract information | The contract's usage declaration exists only as a **scanned page** |
| Uploaded evidence mid-investigation | The purchase date is in **no** corpus document - only on the customer's receipt |
| Explicit refusals | A smudged receipt date can decide the verdict; questions about unknown models must not borrow other models' terms |

Products: **X100** domestic booster pump (named in the policy only - no specification sheet exists, on purpose),
**X200** industrial pump (64 kg, three-phase), **X300** residential circulation pump (7 kg).

The customer: **Acme Facilities Ltd**, which leases one X200, serial **AQ-X200-55871**, for commercial use.

## The indexed corpus (`samples/`, seeded by `python -m scripts.seed_corpus`)

All files are generated deterministically by `samples/make_samples.py`, which also runs a Tesseract self-check
so the OCR behaviour the tests rely on cannot silently change. Metadata (title, type, product, effective date,
what a document supersedes) comes from `samples/manifest.json` and is stored in SQLite.

| File | Pages | Type / product | Effective | What it contains |
|---|---|---|---|---|
| `warranty_policy_v1.pdf` | 4 (text) | policy / X-series | 2023-01-01 | Limited Warranty Policy AQF-WP-1.0: §1 scope and covered products (X100, X200, X300; leased units; other models not covered), §2 **24-month** warranty period, §3 what is covered (components, remedies, costs incl. labour, shipping, on-site call-out > 25 kg, replaced parts), §4 exclusions (wear, dry running, improper installation, unapproved fluids, lightning, flooding, consequential losses ...), §5 conditions (registration within 30 days, certified installer for X200 and commercial use, maintenance), §6 proof of purchase, §7 claims procedure (within 30 days, 2/10/15 business days), §8 disputes, §9 transfer, §10 liability, §11 definitions, §12 key time limits table |
| `warranty_amendment_2025.pdf` | 2 (text) | amendment / X-series | 2025-01-01 | Amendment AQF-WP-1.0-A1, **supersedes §2** for units bought from 1 Jan 2025: commercial units **12 months**, residential stays 24 months; usage class is decided by the customer contract; commercial labour covered for the first **6 months** only; maintenance log required for commercial claims; summary table |
| `service_manual_x200.pdf` | 3 (text) | manual / X200 | - | §1 specifications (10 bar, 42 m3/h, 5.5 kW 400 V, 80 C, 64 kg), §2 installation requirements, §3 approved fluids (water, ≤ 30% glycol), §4 maintenance schedule, §5 failure vs normal wear (**motor seizure is a failure**), §6 error codes E01-E05, §7 warranty note (*purchase date must come from the sales receipt*) |
| `customer_contract_acme.pdf` | 3 (pages 1-2 text, **page 3 scanned image**) | contract / X200 | 2025-03-01 | Equipment lease ACME-LEASE-2025-014: unit and serial, 36-month term (*"purchase date: as per the original sales receipt, which is not reproduced"*), GBP 310 monthly rental, maintenance, insurance, warranty clause, termination; **Schedule B (scanned)**: usage classification **COMMERCIAL**, with a deliberately smudged signature |
| `service_manual_x300.pdf` | 1 (text) | manual / X300 | - | X300 specifications, failure vs wear, maintenance, **Extended Care: 36 months** (X300 only, registration within 30 days + annual service, not for commercial use). A **decoy** for X200 questions |

Two receipts that are uploaded during the main demonstration (not indexed into the corpus):

| File | What it shows |
|---|---|
| `receipt_nonhinge.png` | Clear receipt for AQ-X200-55871 dated **05-03-2025** (receipt number smudged - irrelevant) |
| `receipt_hinge.png` | Same, dated 17-03-2025 but the **7 is smudged**, so OCR reads `1-03-2025` at low confidence |

### Built-in difficulties (on purpose)

1. **Contradiction resolved by versioning** - 24 vs 12 months. The amendment's metadata says what it supersedes;
   retrieval fetches an amendment whenever the document it amends is retrieved.
2. **Scanned page** - the commercial classification exists only as an image (Schedule B); OCR runs only on that
   page, and the value is verified word by word (the smudged signature on the same page must not block it).
3. **Missing fact** - no document contains the purchase date; the agent must ask for the receipt.
4. **Decisive uncertainty** - on the smudged receipt, 12 vs 17 March decides coverage (warranty ends 12 vs 17
   March 2026; the failure is on 14 March 2026), so no verdict can be given.
5. **Decoy** - the X300 manual's 36-month Extended Care must not be applied to the X200.
6. **Trap** - the contract's catalogue effective date (1 March 2025) is not a purchase date.
7. **Unknown models** - X500 / X900 appear in no document; the system refuses instead of borrowing policy terms.

### Ground truth for the main question

> *Is the motor seizure of Acme Facilities' Aquaflow X200 pump (serial AQ-X200-55871) on 14 March 2026 covered
> under warranty?*

Motor seizure is a failure (X200 manual §5) · the unit is commercial (contract Schedule B) · bought in 2025, so the
amendment's **12 months** apply · labour only for the first 6 months · the purchase date must come from the receipt:

| Receipt | Purchase date | Warranty ends | Verdict for a 14 March 2026 failure |
|---|---|---|---|
| none | unknown | - | **NEEDS_INPUT** (asks for the receipt), then MISSING_INFO if none is given |
| `receipt_nonhinge.png` | 05-03-2025 | 05-03-2026 | **Not covered** (cited answer) |
| `test_uploads/receipt_x200_clean_covered.png` | 20-03-2025 | 20-03-2026 | **Covered** - parts; labour not covered after month 6 |
| `receipt_hinge.png` | 1?-03-2025 | 12-03 or 17-03-2026 | **INSUFFICIENT_EVIDENCE** - the uncertain digit decides it |

## Test uploads (`samples/test_uploads/`, generated by `python -m samples.make_test_uploads`)

Used by the manual behaviour tests (`docs/behaviour_test_pack.md`, `docs/AgenticRAG_test_run.xlsx`):

| File | Behaviour it triggers |
|---|---|
| `receipt_x200_clean_covered.png` | Clean receipt, 20-03-2025 → covered |
| `receipt_x200_clean_expired.png` | Same file as `receipt_nonhinge.png` → not covered |
| `receipt_x200_smudged_digit.png` | Same file as `receipt_hinge.png` → uncertain OCR decides the verdict |
| `receipt_x200_blurred.png` | Whole receipt out of focus → every value low-confidence |
| `receipt_x200_wrong_serial.png` | Legible receipt for a **different** X200 (AQ-X200-60112) → must not be used |
| `warranty_card_x300.png` | X300 Extended Care card (AQ-X300-10457, bought 02-09-2024) → answer only possible from the upload |
| `installation_certificate_x200.pdf` | Text PDF (no OCR needed): certified installation of Acme's X200 |
| `purchase_records_conflicting.pdf` | Two records with different purchase dates for the same unit → CONFLICT |
| `maintenance_log_x200_scanned.pdf` | 6 scanned pages (OCR on every page) → indexing an upload; interrupt-and-resume |
| `receipt_x200_clean_covered_COPY.png` | Same bytes, different name → duplicate detection |
| `unsupported_notes.csv` | Unsupported file type → rejected |

## Question sets (`samples/`)

| File | Questions | Purpose |
|---|---|---|
| `heldout_questions.json` | 42 | Written per document section and question type (lookup, procedure, yes/no, casual wording, cross-document, versions, out-of-scope); **never used for tuning**; expected section, outcome and required facts per question |
| `golden_questions.json` | 23 | Questions that failed during live testing, kept as a regression set |

Run them with `python -m scripts.eval_retrieval` (retrieval only, no LLM) and `python -m scripts.eval_answers`
(the whole agent).
