# WoWAH Player Churn Prediction

A pipeline that predicts, for each World of Warcraft character and month,
whether the character will have churned, i.e. stopped playing, by the end of the
following month. It uses the public
[World of Warcraft Avatar History (WoWAH)](https://qualinet.github.io/databases/world_of_warcraft_avatar_history_dataset/)
dataset: roughly one snapshot every 10 minutes of every online Horde character
between 2006 and 2009.

The pipeline is built with [polars](https://pola.rs/) and
[skrub](https://skrub-data.org/) DataOps, with a scikit-learn model on top. A
custom month-by-month splitter makes sure features and targets never use data
from the future. See [REPORT.md](REPORT.md) for a detailed walkthrough.

## Task definition

Let `M` be the month in which we score a character (and could act on it).

- **Unit of analysis**: a `(char, month)` pair, from the character's first month
  on.
- **Churned as of a month** (`churn`): the character was **not seen in that
  month and not seen in the previous month**.
- **Target** (`churn_next`): the character has churned as of `M + 1`, i.e. it is
  absent in both `M` and `M + 1`. Only characters that played in `M - 1` are scored
  (`build_churn_dataset(..., only_prev_active=True)`): the others have no features.
- **Features** for a row at month `M` are computed from the heartbeats of month
  `M - 1` only (plus fixed attributes such as race and class).
- **Validation**: `Splitter` tests on a month `T` and trains on the rows up to
  `T - 2`. The target of a row at `T - 1` depends on `T`, so it is left out. The
  first test month is March 2006 and the last is November 2008. Metric: ROC AUC.

## Repository layout

| Path | Purpose |
|---|---|
| `main.py` | Builds the skrub DataOps plan (target, feature table, `TableVectorizer`, `HistGradientBoostingClassifier`), the `Splitter`, and the `evaluate`, `cross_validate` and `random_search` entry points |
| `add_churn.py` | Builds the `(char, month)` table with `has_played`, `churn` and the target `churn_next` (`build_churn_dataset`) |
| `explore_data.py` | Exploratory profiling of the heartbeat parquet: attributes, per-character activity, heartbeat gaps, sessions, monthly activity, and a `skrub` report on a character sample (`pixi run explore`; `pixi run explore-plots` redraws the figures from the cached tables) |
| `adding_features.py` | Session, monthly player, class, location (zone rarity, hubs, Gini) and lag/diff features |
| `src/utils.py` | `sample_by_user` (sample whole character histories) and `get_session_duration` |
| `src/parse_to_single_table.py`, `src/convert_to_parquet.py` | Parse the raw WoWAH `.txt` logs to CSV, then to parquet |
| `location_features.py`, `lagged_features.py`, `main-unpacked.py` | Exploration scripts and an older variant of `main.py` split into more DataOps nodes |
| `plotting/` | Plots: users per month, monthly stats, polar plots of session start/end times, the session-gap example, zone rarity and the row-vs-character sampling comparison |
| `index.qmd`, `draft.qmd`, `skrub-dataops.qmd` | Website landing page, blog post index and slide deck; rendered output goes to `docs/` and is published to GitHub Pages |
| `posts/*.qmd` | The blog post, split into six standalone parts (`01-data` … `06-results-and-runtime`); `draft.qmd` is the index that links to them |
| `sections/_setup.qmd`, `sections/_load_data.qmd` | Shared `#\| include: false` cells (the `show()` helper and the parquet load) included by every part |
| `OUTLINE.md` | Internal backlog of material that exists in the code but is not yet in any part; not published |
| `wowah_churn_study.db` | SQLite storage with the Optuna studies from `random_search()` |
| `report.dot`, `report.mmd`, `report.svg`, `build_feature_table.*` | Diagrams of the DataOps plan |

## Setup

```bash
pixi install            # the only environment: runs and renders everything
```

A single environment (aliased as `docs`, so `pixi run -e docs ...` still works)
holds both the pipeline dependencies and the rendering stack (Quarto, Jupyter,
ipykernel, jupyter-cache). It can therefore run every script and render every
`.qmd` file; there is no need for a separate docs environment.

`graphviz` and `pydot` are included because `data_op.skb.report()`, called at
the end of `main.py`, needs the `dot` command. Use `pixi run` or `pixi shell` so
that it is on the `PATH`.

## Data

Everything lives in `data/`, which is git-ignored. Two old parquet files
(`wowah_data_raw.parquet`, `wowah_user_month_with_churn.parquet`) are still
tracked by git from before that.

| File | Content |
|---|---|
| `wowah_data_2008.parquet` | Heartbeats for 2008 only (about 10.8M rows, 37k characters). Default input of the pipeline |
| `wowah_data_all.parquet` | Full dataset, 2005-12 to 2009-01 (about 35.6M rows, 90k characters) |
| `wowah_parsed_all.csv` / `.parquet` | Output of the parsing scripts (column `avatar_id` instead of `char`) |
| `wowah_churn_data.parquet`, `wowah_user_month_with_churn.parquet` | Legacy churn tables, no longer produced or used by `main.py`. Some older scripts still read the first one |
| `wowah.rar`, `no_matches_files.log`, `index.html`, `img/` | Original archive, parser log and the dataset's documentation page |

The 2008 and "all" parquet files have columns `timestamp, char, guild, level,
race, charclass, zone`. No script in this repository creates them from
`wowah_parsed_all.*`. The parsing scripts also contain hard-coded paths from an
earlier layout and need adjusting before use.

## Usage

```bash
pixi run evaluate                       # fit on the historical raw files, 10% of characters
pixi run evaluate-all                   # same, keeping every character
pixi run evaluate-unseen                # evaluate on the held-out raw files
pixi run predict-unseen                 # fit on historical, predict Nov 2008
pixi run python main.py --env train --fraction 1   # same as evaluate-all
```

`--env` is `train` (default), `unseen` or `predict-unseen`, and `--fraction` is
the share of characters to sample. The same options exist as arguments of
`evaluate()`, `cross_validate()` and `random_search()` (Optuna backend, results
in `wowah_churn_study.db`).

### Tuning

The choices are grouped by what they cost to evaluate, and `--stage` selects
which group a run declares:

| `--stage` | Choices | Model | Sample | Cost/trial |
|---|---|---|---|---|
| `features` | the eight feature-table choices | HGB, defaults | 2% | 145 s |
| `gaps` | the three noisy feature choices, re-tested | logistic regression | 10% | 766 s |
| `structural` | the six post-feature-table choices | logistic regression | 10% | 342 s |
| `capacity` | the four model hyperparameters | HGB | 10% | 356 s |
| `full` | all of them | HGB | — | — |
| `none` | no choice at all; every knob at its default | HGB | — | — |

Costs are sequential, measured on this machine with the parse cached. The
`gaps` stage is the most expensive per trial because each of its 18 combinations
is a cold feature table; it is also the smallest.

```bash
# Stage 1: which features are worth building (2% sample, ~2 min/trial warm)
pixi run python main.py --env train --fraction 0.02 --stage features \
    --search --n-iter 40 --study-name wowah_churn_1_features

# Stage 2: re-test the three noisy feature choices at 10%, rest frozen
pixi run python main.py --env train --fraction 0.1 --stage gaps \
    --search --n-iter 18 --study-name wowah_churn_2_gaps \
    --frozen '{"lags": [1,2,3], "diff_lags": [1,2], "location_features": true,
               "add_gini": false, "use_monthly_features": true}'

# Stage 3: structural choices, with the feature choices pinned
pixi run python main.py --env train --fraction 0.1 --stage structural \
    --search --n-iter 40 --study-name wowah_churn_3_structural \
    --frozen '{"session_gap": 45, "zone_session_gap": null,
               "use_class_features": false, "lags": [1,2,3], "diff_lags": [1,2],
               "location_features": true, "add_gini": false,
               "use_monthly_features": true}'

# Stage 4: model capacity, with everything else pinned
pixi run python main.py --env train --fraction 0.1 --stage capacity \
    --search --n-iter 60 --study-name wowah_churn_4_capacity \
    --frozen '{"session_gap": 45, "lags": [1,2,3], "drop_char": "DropCols", ...}'

# Final: refit the winning configuration on every character
pixi run python main.py --env train --fraction 1 --stage none \
    --frozen '{"session_gap": 45, "lags": [1,2,3], ...}'
```

`--frozen` takes a JSON object mapping choice names to values; JSON has no
tuples, so `lags` and `diff_lags` are given as lists and converted on load.
`--study-name` is what makes a run resumable — without it skrub generates a
fresh UUID per call. `--timeout` bounds the search in seconds.

The search runs **sequentially**. Parallel trials are not offered: `parse_files`
spawns a process pool sized to the core count, so N workers on a cold cache
means N × cores processes, and 8 workers drove a 24 GB machine into 40 GB of
swap. See AGENTS.md, "Do not parallelise the search", for the measurements.

`--timeout` bounds the search in wall-clock seconds. It is safe because the
search is sequential — skrub only switches to Optuna's threaded
parallelisation when a timeout is set *and* `n_jobs > 1`, and threads deadlock
against the parser's process pool. Note that a sleeping machine burns the
timeout, so `--n-iter` is the more predictable bound.

Searched hyperparameters: the session gap (30, 45 or 60 minutes), the
zone-session gap, the lags and diffs, whether to use location features and the
Gini coefficient, whether to add the monthly and class-level blocks, whether to
drop `char` and `month`, the vectorizer's cardinality threshold, high-cardinality
encoder and null-drop fraction, and the model's regularisation and capacity.

Other tasks: `pixi run churn` prints the churn table, `pixi run explore` profiles
the full heartbeat dataset (HTML report and plots in `reports/exploration/`),
`pixi run plot-user-months`, `plot-stats-by-month`, `plot-session-times`,
`plot-sampling` and `plot-zone-rarity` redraw the plots, and `pixi run render`
(or `render-draft` / `render-slides`) renders the documents into `docs/`.
`render-draft` renders only the series index (`draft.qmd`); to render one part,
pass its path: `pixi run quarto render posts/03-sessions.qmd`.

## Website

The documents are published as a Quarto website at
<https://skrub-data.org/wowah-sessionization/>: `index.qmd` is the
landing page and links to the blog post and the slides
(`skrub-dataops.qmd`). The blog post itself is a series: `draft.qmd` is the
index and each part lives in `posts/`. A push to `main` triggers
`.github/workflows/publish.yml`, which renders the site and pushes `docs/` to
the `gh-pages` branch. `_freeze/` is committed so the workflow can render the
blog post without the raw dataset; refresh it locally after changing a code
cell. To publish by hand, run `pixi run quarto publish gh-pages`.

Iterating on the exploration figures does not need the full scan: the monthly
table and the gap histogram are cached under `reports/exploration/cache/`, so

```bash
pixi run explore-plots --plots characters --show
```

redraws a single figure from the cache and opens it in a window. `--plots`
accepts `gaps`, `characters`, `levels` or `all`, and `--dpi` controls the
resolution of the saved files.
