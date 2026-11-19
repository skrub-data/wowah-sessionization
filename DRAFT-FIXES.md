# Draft fixes

Discrepancies between `draft.qmd` and the code, which is the ground truth.
Line numbers refer to the working tree at the time of writing (`draft.qmd` as of 2026-11-18, after the new
outline sections were added; earlier line numbers in this file were stale).

Legend: **D** = `draft.qmd`, **C** = code.

## Resolved by the `only_prev_active` change (2026-11-19)

`build_churn_dataset(historical_data, only_prev_active=False)` now has an opt-in
filter that keeps only rows where `prev_played` is true; `make_data_op` uses it for
`X` and `y`, while `build_feature_table` keeps the full grid (lags need it).

| # | Status |
|---|---|
| 5 | Fixed in code: `remove_unrealistic_entries` removed (output identical up to row order; the table is now sorted by `char, month`). Draft text updated. |
| 9 | Fixed in code: dead `is_not_null()` clause removed. |
| 12 | Fixed in code and draft: new-churner target, with the 75% / 0.63 numbers in a callout. The "stopped playing / suspended subscription" wording in the draft's definition section is still to be revisited. |
| 14 | Partly fixed: `add_churn.py` module, `make_user_month` and `add_churn` docstrings updated. |
| 6 | Draft text updated to "last observed month and the three padding months". |

New numbers on the 2008 parquet: 81,199 rows (was 221,237), 36.5% positive (was 54.5%).
On the 2008 raw files through `HISTORICAL_END` (10% sample, default hyperparameters),
ROC AUC per test month: old definition Mar–Sep 0.50 0.55 0.75 0.79 0.79 0.83 0.79
(7 folds); new definition Apr–Sep 0.82 0.82 0.82 0.83 0.87 0.88 (6 folds, March is
lost because January has no previous month). The full dataset keeps 31 folds
(2006-03 … 2008-09) with either definition. Not comparable populations; the full-dataset
results were not rerun.

Why the scores went *up*: the "easy" assumption was wrong. The already-absent rows have
no features, so the model cannot rank within them (AUC 0.43–0.80), and they are 40–70% of
the old test rows and most of the training rows. On the same test rows (active in M−1), a
model trained on all rows scores 0.64–0.82 (Apr–Sep) vs 0.82–0.90 trained on active rows
only. Single diagnostic run, default features (no lags), cause not tested. In the
2008-only window the Jan 2008 rows are also all `prev_played=False` (December 2007 is
missing), which hurts the old early folds.

Items 1 and 11: the draft no longer contains a results table; `REPORT.md` has been
updated with the 2008-only table above, but its fold counts for the full dataset (33)
and `readme.md` are still stale.

## Summary

| # | Severity | What is wrong | Draft | Code |
|---|---|---|---|---|
| 1 | high | Results table has 9 folds (Mar–Nov 2008); the pipeline produces 31 (Mar 2006 – Sep 2008) | D:535–539 | `main.py:60`, `main.py:312–336` |
| 2 | high | "Last test month" is asserted, not derived; `MAX_DATE` is a split point, not a data bound | D:429–433 | `main.py:44`, `main.py:60`, `main.py:78–121` |
| 3 | medium | Post opens on the parquet; the pipeline parses raw `.txt` files | D:156–165, D:524 | `main.py:248–257`, `main.py:259–308`, `src/parse_to_single_table.py:78–134` |
| 4 | medium | Headline row/character counts describe the exploration sample, not the training window | D:156–158 | `main.py:60` |
| 5 | medium | `remove_unrealistic_entries` is presented as the place where pre-first-month rows are dropped; it drops nothing | D:364–367, D:371 | `add_churn.py:128`, `add_churn.py:156–180` |
| 6 | low | `churn_next` is nulled for four months, not one | D:341–345 | `add_churn.py:148–152`, `add_churn.py:190` |
| 7 | low | Diagram and "Our objective" callout use different month conventions | D:225–242 | `add_churn.py:1–9`, `main.py:78–121` |
| 8 | low | `diff_lags` and `add_gini` are fixed, not tuned | D:464–466, D:509–510 | `adding_features.py:299–357`, `main.py:295` |
| 9 | low | Dead `is_not_null()` clause in the churn expression | — | `add_churn.py:143` |
| 10 | low | `main-unpacked.py` is referenced but deleted | — | `readme.md`, `REPORT.md` |
| 11 | low | `readme.md` / `REPORT.md` repeat the stale fold counts and a removed CLI flag | — | `main.py:386–425` |
| 12 | medium | Churn is described as "stopped playing"; the label is a rolling, non-absorbing flag, and ~75% of positive rows are long-absent characters | D:187–211, D:706–710 | `add_churn.py:139–152` |
| 13 | low | `main.py` comments still say the last test month is November | — | `main.py:41–44`, `main.py:52–59` |
| 14 | low | `add_churn.py` docstrings contradict the code (padding months, "last month" null) | — | `add_churn.py:55–75`, `add_churn.py:108–115` |
| 15 | low | `interval` is accepted but not wired through to the churn grid | — | `main.py:129–134`, `add_churn.py:181–190` |
| 16 | medium | Unseen-environment churn labels are computed on a different population and a partial January | D:765–790 (outline) | `main.py:62–72`, `main.py:312–335`, `src/utils.py:3–13` |

---

## 1. Results table is stale

**D:535–539** — the sentence "The ROC AUC of each test month (March to November
2008 …)" and the 9-column table `| Mar | … | Nov |`.

**C:** `main.py:60` sets `HISTORICAL_END = 2008-10-31`, and `main.py:312–336`
(`get_env`) feeds the *train* environment every raw file up to that date — 132,042
files starting 2006-01-01, not the 2008 parquet. `main.py:40` and `main.py:44`
(`MIN_DATE = 2006-02-01`, `MAX_DATE = 2008-10-01`) then drive `Splitter`
(`main.py:78–121`).

Verified fold counts:

| Input | Folds | Test months |
|---|---|---|
| 2008 parquet, no cutoff | 9 | 2008-03 … 2008-11 |
| 2008 parquet, `HISTORICAL_END` | 7 | 2008-03 … 2008-09 |
| full parquet, no cutoff | 33 | 2006-03 … 2008-11 |
| full parquet, `HISTORICAL_END` | 31 | 2006-03 … 2008-09 |

The table was committed in `b9c21af` (2026-11-05); `main.py` gained the
raw-file/`HISTORICAL_END` machinery afterwards in `539dcea` and `91664b2`
(2026-11-13).

**Fix:** regenerate the table from `pixi run python main.py --env train --fraction 0.1`
and label the window explicitly (31 folds, March 2006 – September 2008). The
frozen copy in `_freeze/draft/execute-results/html.json` must be re-rendered too.

## 2. "Last test month" is asserted rather than derived

**D:429–433** — the inline `display(Markdown(...))` block claims split points run
from `MIN_DATE` to `MAX_DATE` and that "the last test month is the latest one
whose target is fully observed".

**C:** `MAX_DATE = 2008-10-01` (`main.py:44`) would make November the last test
month, but `HISTORICAL_END = 2008-10-31` (`main.py:60`) truncates the parsed data,
so October's `churn_next` needs November and is null. `build_churn_dataset`
(`add_churn.py:190`) drops those rows, and `Splitter` (`main.py:78–121`) skips
folds with an empty test set. The last usable test month is **September 2008**.

The draft also never says that `MIN_DATE`/`MAX_DATE` are *split points*, not data
bounds — which is what makes the sentence read as if November were reachable.

**Fix:** compute the last test month from the data in the inline block instead of
asserting it, and add one sentence distinguishing split points from data bounds.

## 3. Data source narrative

**D:156–165** — "We begin by loading the historical dataset … `df =
pl.scan_parquet("data/wowah_data_2008.parquet")`".

**D:524** — "The data op is evaluated by passing a list of raw log file paths
as input", which contradicts the parquet framing above.

**C:** the pipeline never reads the parquet. `make_data_op` (`main.py:259–308`)
starts from `skrub.var("raw_files")` and calls `load` (`main.py:248–257`), which
calls `parse_files` (`src/parse_to_single_table.py:104–134`) on `.txt` files
selected by `list_raw_files` (`src/parse_to_single_table.py:78–101`). The parquet
is used only for the exploratory `TableReport` and the standalone
`build_churn_dataset` demo.

**Fix:** say up front that the parquet is a convenience copy for exploration and
that the pipeline parses the raw logs; keep the two narratives visibly separate.

## 4. Headline counts describe the wrong dataset

**D:156–158** — "this post uses the year 2008 (about 10.8M rows, 37k characters)".

**C:** those numbers are correct for `data/wowah_data_2008.parquet`
(10,826,734 rows / 37,354 chars), but the train environment covers 2006-01-01 →
2008-10-31: **34,038,187 rows / 84,981 characters**. The full parquet is
35,604,542 rows / 90,456 chars.

**Fix:** either state both numbers with their scope, or run the post on the 2008
slice only and say so.

## 5. `remove_unrealistic_entries` is redundant

**D:364–367** — "It also calls `remove_unrealistic_entries`, a safety check that
removes the rows preceding a character's first month and selects the final
columns."

**C:** `add_churn` already filters `first_month <= month` at `add_churn.py:128`.
`remove_unrealistic_entries` (`add_churn.py:156–180`) re-joins `first_month` and
re-filters. Verified on the 2008 parquet: 370,653 rows in, 370,653 rows out — it
removes nothing and only reorders/renames columns.

**Fix:** either delete the function and its call in `build_churn_dataset`
(`add_churn.py:181–192`), or reword the draft to say it only normalises the
column order.

## 6. `churn_next` is nulled for four months, not one

**D:341–345** — "The last month in the data has no 'next month' to look at, so its
`churn_next` is left empty."

**C:** `add_churn.py:148–152` nulls `churn_next` whenever `month + 1mo >
last_month`, i.e. for the last observed month **and** the three padding months
added by `make_user_month`. Verified on the 2008 parquet: nulls in
`2008-12, 2009-01, 2009-02, 2009-03`.

**Fix:** say "the last observed month and the three padding months after it".

## 7. Diagram and callout use different month conventions

**D:225–242** — the "Our objective" callout says: given June, use data through May,
predict churn as of July. The diagram (`images/diagram-months-churn.png`) shows
training months January/February/March, April as the gap month, May as the target.

**C:** both are consistent with `add_churn.py:1–9` (April → act in April, target
May) and with `Splitter` (`main.py:78–121`), but the draft never reconciles them,
and the diagram's "Training months: January, February, March" reads as if training
stops at March, contradicting the callout's "all months until May included".

**Fix:** pick one convention and use it in both places; relabel the diagram's
training box as "all months up to the gap month".

## 8. Fixed, non-tuned settings are presented as tunable

**D:464–466** — lagged features described as "the value of every feature one or
more months earlier, plus the month-to-month difference of numeric features".

**D:509–510** — "the session gap, whether to use location features, how many lags
to add, and the learning rate of the model".

**C:** `add_lagged_features` (`adding_features.py:299–357`) hard-codes
`diff_lags=(1,)`, and `make_data_op` (`main.py:295`) hard-codes `add_gini=True`.
Neither is exposed as a `skrub.choose_*`.

**Fix:** mention that the diff lag and the Gini switch are fixed.

## 9. Dead clause in the churn expression

**C only:** `add_churn.py:143` — `churn=pl.col("prev_played").is_not_null() & …`.
`prev_played` is already `fill_null(False)` on the line above, so `is_not_null()`
is always true. Verified: `prev_played` null count is 0.

**Fix:** drop the clause.

## 10. `main-unpacked.py` is gone but still referenced

`git status` shows `D main-unpacked.py`. `readme.md` and `REPORT.md` describe it
as an older variant. It also still used `has_played` as the target
(`main-unpacked.py:236`), which is the leaky setup the draft's "Why the target is
shifted" callout describes.

**Fix:** remove the references from `readme.md` and `REPORT.md`.

## 11. Secondary docs repeat the stale numbers

- `readme.md:29` — "The first test month is March 2006 and the last is November
  2008". True without the cutoff; with `HISTORICAL_END` the last is September 2008.
- `readme.md:79–90` — documents `--dataset 2008|all`, but `main.py:386–425` now
  takes `--env {train,unseen,predict-unseen}` and `--fraction`.
- `REPORT.md:98–99` — "the 2008 file gives 9 folds … and the full dataset 33".
  With `HISTORICAL_END`: 7 and 31.
- `REPORT.md:234–242` — the 9-fold results table, same staleness as item 1.

**Fix:** update both after regenerating the results.

## 12. Churn is a rolling flag, not "stopped playing"

**D:187–211** — churn is "avatars that are not found online for at least two
consecutive months … users that stopped playing, and consequently suspended their
subscription"; "a character that skips a month and then comes back is not
considered churned". **D:706–710** (outline) quotes the 54.5% base rate without
saying what it contains.

**C:** `add_churn.py:139–152` defines `churn = ~has_played & ~prev_played` per month
with no memory, so:

- it is **not absorbing**: a character absent for N consecutive months gets N−1
  positive `churn_next` rows, and one that returns after a two-month gap flips back
  to negative. "Skips one month and comes back" is correct, but "skips two and
  comes back" is still labelled churn at the second absent month.
- positives are dominated by long-absent characters. Verified on the 2008 parquet
  (221,237 rows, 54.5% positive): **75.4% of positive rows have `prev_played ==
  False`**, i.e. the character was already absent in M−1 (versus 48.8% of
  negative rows). The rule "absent last month" alone scores **ROC AUC 0.63**.
  Features built from M−1 are empty for these rows, so they are the easy ones.

This also makes the 54.5% base rate and the AUC table hard to interpret without
the "already absent" split planned in outline section 7 (D:722+).

**Fix:** reword the definition as "absent in two consecutive months, evaluated
independently for every month", state that a long-gone character keeps
contributing positive rows, and add the 75% / 0.63 numbers (or report AUC split
by `prev_played`) next to the results table. Don't use "suspended their
subscription" as a property of the label; it is an assumption.

## 13. Stale comments in `main.py`

**C:** `main.py:41–43` says "The last split point is October 2008, so the last test
month is November"; `main.py:52–59` justifies `HISTORICAL_END` by December 2008's
target needing January 2009. With `HISTORICAL_END = 2008-10-31` the last labelled
row is September 2008 (see item 2), so the last test month is September. The
comments describe the pre-`HISTORICAL_END` full-dataset behaviour.

**Fix:** update both comments (and keep them consistent with item 2 in the draft).

## 14. `add_churn.py` docstrings disagree with the code

- `make_user_month` (`add_churn.py:55–75`) says the 3 padding months exist so lagged
  features don't crash. In the pipeline they never reach the lag code:
  `build_churn_dataset` filters out rows with null `churn_next`, including all
  padding rows, and `build_feature_table` (`main.py:162`) builds its grid from that
  filtered output. The padding only matters for the standalone helper functions.
- `add_churn` (`add_churn.py:108–115`) says `churn_next` is null "for the last month
  in the data"; it is null for the last month **and** the 3 padding months
  (same as item 6, but in the code docstring).
- The module docstring (`add_churn.py:1–26`) says the "has played" column is
  filled with False and that rows before the first month are filtered, but never
  mentions `prev_played`, `churn`, or `churn_next`.

**Fix:** correct the docstrings; consider dropping the padding (it is dead for the
pipeline) or documenting that it is kept for the helpers only.

## 15. `interval` is not wired through

`build_feature_table(interval=...)` (`main.py:129–134`) and `Splitter.split`
(`main.py:90`) take an `interval`, but `build_churn_dataset` (`add_churn.py:181–190`)
calls `make_data`/`make_user_month` with their default `"1mo"`. Passing another
interval would give a churn grid and features on different time bases, and the `"3mo"`
padding in `make_user_month` is hard-coded.

**Fix:** either thread `interval` through `build_churn_dataset`, or remove the
parameter from the other functions and say churn is monthly only.

## 16. Unseen-environment labels are not comparable to the training ones

*(From code reading; not run.)* **D:765–790** (outline section 9) and `REPORT.md:113–132`
describe the unseen environment as the same pipeline on new files. The churn labels
differ in three ways:

- `first_month` (`add_churn.py:46–47`, `128`) is recomputed on the files given, so
  every character active in October 2008 looks new in the unseen run, and
  characters inactive for the whole of Oct 2008 – Jan 2009 have no rows at all
  (in the training run they would appear as churners).
- `last_month` is January 2009, which stops on the 10th (`main.py:62–72`).
  December's `churn_next` is therefore "absent in December **and** in the first 10
  days of January", so some December rows labelled churn are characters that return
  later in January.
- `sample_by_user` (`src/utils.py:3–13`) draws the 10% sample, `seed=42`, from the
  unique characters of whichever files it is given, so the unseen sample is not a
  subset of the training sample.

**Fix:** state these caveats in outline section 9, and don't quote unseen
performance as directly comparable to the cross-validated AUCs.
