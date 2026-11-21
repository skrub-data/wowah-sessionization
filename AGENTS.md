# AGENTS.md

## Use the pixi environment for everything

This project is managed with [pixi](https://pixi.sh). The environment is already
solved and checked in (`pixi.toml`, `pixi.lock`), and it holds **every** tool the
project needs: Python, Quarto, Jupyter, Graphviz, Polars, skrub, scikit-learn,
Optuna and the rest.

Always run commands through pixi, from the repository root:

```bash
pixi run python ...        # any Python script, module or one-off snippet
pixi run quarto render     # render the site or a single document
pixi run jupyter ...
```

Prefer the pixi tasks declared in `pixi.toml` when one exists:

```bash
pixi run render
pixi run render-slides
pixi run render-draft
pixi run churn
pixi run explore
pixi run evaluate
```

### Do not call binaries inside `.pixi/envs/` directly

`pixi run` sets up the environment (PATH, `PYTHONPATH`, native plugins) before
executing a command. Invoking `.pixi/envs/default/bin/quarto` or
`.pixi/envs/default/bin/python` by path skips that setup and fails with
confusing errors such as a missing `tools/<arch>/deno` or `dart-sass`. The same
applies to spawning the interpreter from Python subprocesses.

### Do not install tools globally

`pip install`, `pipx`, `npx`, Homebrew and system packages are not the way to
get a tool here. Quarto, `python`, `pandoc`, `dot` and friends are already in
the environment; if a package is genuinely missing, first ask the user for confirmation
and then use `pixi add <package>` so 
that `pixi.toml` and `pixi.lock` stay in sync. A temporary `npx
@mermaid-js/mermaid-cli` run to convert a diagram is fine; it must not be
treated as a project dependency.

## Rendering

- Quarto lives in the environment: `pixi run quarto --version`.
- `pixi run quarto render` renders every `*.qmd` listed in `_quarto.yml` into
  `docs/`, which is git-ignored.
- To check one document without touching `docs/`, render it to a scratch
  directory:

  ```bash
  pixi run quarto render skrub-dataops.qmd --output-dir /tmp/slides-out
  ```

- `execute: cache: true` and `jupyter-cache` are configured, so re-renders reuse
  cached results. Use `pixi run quarto render <file> --no-cache` when a code
  cell must actually re-execute.
- `skrub-dataops.qmd` is a RevealJS slide deck;
  `draft.qmd` and `skrub-dataops.qmd` are the write-ups of record.
- `index.qmd` is the landing page: it links to the blog post and the slides.
  The navbar in `_quarto.yml` mirrors those links.
- The blog post is a series. `draft.qmd` is the index and each part is a
  standalone document in `posts/` (`01-data` … `06-results-and-runtime`), with
  a `Part N ·` title prefix. Parts are self-contained: each one includes the
  shared `sections/_setup.qmd` and `sections/_load_data.qmd` cells and uses
  `../` paths for images and includes.
- `sections/*.qmd` are draft fragments; they are excluded from the site build
  (`!sections/**` in `_quarto.yml`). Only the two `_`-prefixed files are still
  included, by the parts in `posts/`.
- `OUTLINE.md` is the internal backlog of material not yet written into a part.
  It is not rendered.

## Publishing

The site is published to GitHub Pages at
<https://skrub-data.org/wowah-sessionization/> (the custom domain of the
`skrub-data.github.io` organisation site).

- `.github/workflows/publish.yml` renders the site on every push to `main` and
  force-pushes `docs/` to the `gh-pages` branch, which Pages serves.
- `_freeze/` is committed on purpose. The blog post's code cells need the 4 GB
  raw dataset in `data/` (git-ignored), so the frozen results are what lets CI
  render without the data. Re-render locally with the data present and commit
  the updated `_freeze/` whenever a code cell changes.
- To publish by hand instead: `pixi run quarto publish gh-pages`.

## Python

- Run scripts as `pixi run python path/to/script.py`. Files under `plotting/`
  are cell-style scripts (`# %%`) and some read paths relative to their own
  directory, so respect the `cwd` set by the corresponding pixi task.
- `PYTHONPATH` is set to the project root by `pixi.toml`, so `from src.utils
  import ...` works without installing the package.
- Data is large and lives in `data/` (git-ignored). Do not add new large
  artefacts to the repository; write generated figures to `images/` or
  `reports/`.

## Data: do not re-parse the raw logs

The raw corpus is **132,042 `.txt` heartbeat files (~4 GB)** under
`data/raw/WoWAH/`, organised as `YYYY_MM_DD/YYYY-MM-DD/HH-MM.txt`. Parsing all of
them takes ~75 s and is the single most expensive step in the project.

### The DataOp cache is on by default

`main.py` calls `skrub.set_config(cache=".skrub_cache")` at import time, so any
plan built with `make_data_op` reuses previously computed nodes.

Measured on this machine (Apple M3, 8 cores) at `load_fraction=0.1`, over the 31
folds the `Splitter` yields:

| | wall time | `parse_files` calls |
|---|---:|---:|
| cache off | ~75 min (extrapolated) | 63 |
| cache cold | 289 s | 1 |
| cache warm | 213 s | 0 |

A single `parse_files` over the 132,042 historical files takes **68 s** and
returns 34.9M rows. `cross_validate` evaluates the plan once per fit and once
per score, so an uncached run parses the same files **63 times** (31 folds × 2
+ 1) — about 71 minutes of parsing alone. That is the "game changer": the cache
turns a ~75-minute run into a ~5-minute one, roughly **15× faster**.

The cache-off figure is extrapolated rather than measured end to end: an
uncached `cross_validate` was still running after 30 minutes and was stopped.
The 63-call count and the 68 s per-call cost were measured directly (the parse
was memoized so the count could be taken without paying 63 × 68 s).

Once the cache is cold, the remaining saving is smaller: 289 s → 213 s, about
26%. The 213 s warm run is ~68 s of parsing plus ~145 s of model fitting and
encoding. So the cache is worth having for two distinct reasons — it removes
the 63× re-parse on a fresh run, and it skips the single parse on a repeat run.

The `Splitter` yields 31 folds over the historical window (2006-03 → 2008-09).

The cache is git-ignored and lives at `.skrub_cache/` in the repository root.
Delete that directory to force a full recompute.

**The cache does not invalidate itself when the code changes.** joblib keys a
cached call on the *cached function's own source* plus its arguments — it does
not look at the functions that function calls. Editing `adding_features.py`
would therefore keep serving the old feature table forever. `main.py` works
around this by passing a source fingerprint as an extra argument of
`build_feature_table`:

| Node | Fingerprinted? | Covers |
|---|---|---|
| `load` | no | — (parser is frozen) |
| `build_feature_table` | yes | `main.py`, `add_churn.py`, `adding_features.py` |

**If you add a new module that `build_feature_table` calls into, add it to the
`_source_fingerprint(...)` call**, or edits to it will be silently ignored.

The parse node is not fingerprinted because the parser is frozen: the guard
would only ever fire on a change that is not going to happen. The cost is that
editing `src/parse_to_single_table.py` is not detected — delete `.skrub_cache/`
by hand if you ever do. This is a tidiness trade-off, not a correctness one, and
the fingerprint may be removed entirely later for the same reason.

### A full parse already exists

Check these before writing any code that calls `parse_files` or `load`:

| File | Rows | Coverage |
|---|---|---|
| `data/wowah_data_all.parquet` | 36.5M | 2005-12-31 → 2009-01-10 (everything) |
| `data/wowah_data_2008.parquet` | 10.8M | 2008 only |
| `data/wowah_parsed_all.parquet` | 35.6M | **stale** — pre-dates the parser fix |

For analysis, experiments and one-off checks, read one of these with
`pl.read_parquet(...)`. Only go through the raw `.txt` files when you are
specifically testing the parsing node itself.

**`wowah_data_all.parquet` is generated by `build_parquet.py`, which calls
`parse_files`.** Regenerate it whenever `src/parse_to_single_table.py` changes:

```bash
pixi run python build_parquet.py     # ~75 s, needs the raw corpus
```

The plan reads it by default (`make_data_op(source="parquet")`), which takes
~1 s instead of ~75 s. `source="raw"` parses the `.txt` files instead; the two
are interchangeable and give the same results, with one deliberate difference:
the parquet is a strict **superset**. A heartbeat at 23:57 is stored in the
*next* day's folder, so `list_raw_files`'s folder-name filter drops it while the
parquet keeps it. Measured on the same config: parquet 208 s / AUC 0.8724,
raw 323 s / AUC 0.8734.

### The parser's two drop rules are intentional

`src/parse_to_single_table.py` drops a line when either:

- the **race or class** field contains non-ASCII characters (e.g. `373族`,
  `547人`) — these are corrupted records and the field is a fixed attribute, so
  a bad value is worse than a missing row; or
- the line does not match the heartbeat shape at all.

Zone names containing `:` or `'` (`Auchindoun: Shadow Labyrinth`,
`Ahn'kahet: The Old Kingdom`, `Zul'Drak`) **are** parsed correctly — `zone_field`
is `[^,"]+`, which allows them. If you suspect a parsing bug, test the regex
directly against a literal line before concluding anything:

```bash
pixi run python -c "
from src.parse_to_single_table import line_re
m = line_re.match('\t[1] = \"0, 01/01/06 00:03:22, 41,373, , 60, Orc, Mage, Auchindoun: Shadow Labyrinth, yes, 0\",')
print(m.group(8) if m else 'DROPPED')
"
```

## Evaluating the model: use the DataOp, not a hand-rolled loop

`main.py` defines the splitter and the plan. Re-implementing either in a scratch
script is how wrong numbers get into the write-up. Two specific traps:

### Use `main.Splitter`, never `StratifiedKFold`

The pipeline's splitter is an **expanding-window time split**: for each month it
trains on everything up to `split_point - 1` and tests on `split_point`. Shuffled
K-fold trains on future months to predict past ones and gives different — and
wrong — answers. A comparison that looked like a clear win under
`StratifiedKFold` reversed sign under the real splitter.

### Prefer `data_op.skb.cross_validate()` / `.eval()`

```python
op = main.make_data_op(load_fraction=0.1)
results = op.skb.cross_validate(environment=main.get_env("train"))
```

This runs the real graph, the real splitter and the real scoring. A hand-rolled
loop over `build_feature_table` output reproduces the splitter's off-by-one
wrongly (features for a row at month *M* come from *M-1*, so the frame you test
on is not the frame you think).

### Bound long runs before starting them

`cross_validate` on the full corpus is slow even with a warm cache. Before
launching anything that touches the raw files or the full feature table:

- estimate the cost first (time one month, one fold, or a 300-file subset);
- run it in the background and keep working, rather than blocking;
- never leave a command running for tens of minutes without output — if it
  produces nothing for a few minutes, stop it and find out why.

Measured `cross_validate` wall time on this machine:

| `load_fraction` | cold | warm |
|---|---:|---:|
| 0.02 | 170 s | 86 s |
| 0.1 | 289 s | 213 s |

A single `.eval()` (one fit, no cross-validation) is ~5 s warm. If a run is
slow, check whether the cache is actually being hit before assuming the code is
at fault — and note that "warm" is only ~26% faster at 0.1, because the
remaining time is model fitting and encoding, not parsing. The big win is
cache-off vs cache-on, not cold vs warm.

### Do not parallelise the search

`random_search` runs trials **sequentially** (`n_jobs=1`) and there is no
option to change that. This is deliberate, and the reason is worth recording so
nobody re-adds it.

`parse_files` uses a `ProcessPoolExecutor` sized to `os.cpu_count()`. On a cold
cache, N parallel search workers means N × cores processes, each building a
35M-row table before subsampling. Measured on a 24 GB, 8-core machine at a 2%
sample:

| workers | wall time | outcome |
|---|---|---|
| 1 | 145 s per trial | fine |
| 2 | 206 s for 2 trials | 1.4× speedup |
| 4 | 222 s for 4 trials | 2.6× speedup |
| 4, cold parse | 961 s for 4 trials | **slower than sequential** |
| 8 | — | **40 GB of swap** |

The 8-worker case is the one that went wrong: the arithmetic said it should
have fit, and it did not. A single trial peaks at ~10 GB while parsing and
~1 GB for a cold feature-table build, and the second is multiplied by the
worker count. The machine swapped hard enough to need a reboot.

There is a second, independent trap. skrub switches to Optuna's **threaded**
parallelisation whenever `timeout` is set *and* `n_jobs > 1` (see
`skrub/_data_ops/_optuna.py`). Threads cannot run the parser's
`ProcessPoolExecutor` safely — the workers deadlock at 0% CPU and never
finish. So `timeout` and parallel trials are mutually exclusive here.

If you ever do want to try it, the safe shape is: warm the cache with one
sequential evaluation first, then start at 2 workers and watch memory. Do not
use `-1`.

### macOS needs a `__main__` guard

`parse_files` uses `ProcessPoolExecutor`. On macOS the default start method is
`spawn`, so any script that triggers parsing **must** guard its entry point:

```python
if __name__ == "__main__":
    main()
```

Without it you get `BrokenProcessPool` or a `RuntimeError` about
`freeze_support()`, which looks like a bug in the parser but is not.

## Diagrams and figures

- Mermaid sources live at the repository root (`*.mmd`, for example
  `session_features.mmd`, `build_feature_table.mmd`) and are rendered next to
  the existing diagrams in `images/`.
- Generated figures are committed when a document embeds them, so update both
  the source and the rendered file.
- Keep plotting scripts reproducible: fixed data paths, no interactive display
  by default, and a `# %%` cell layout consistent with the other files in
  `plotting/`.
- **Every matplotlib figure uses the `seaborn-v0_8-talk` preset.** Start the
  plotting code with `plt.style.use("seaborn-v0_8-talk")` and do not hard-code
  `fontsize=` values that undercut it. The preset already sets 14.3 pt axis
  labels, 15.6 pt titles and 13 pt ticks and legends, which is what keeps the
  figures readable once they are scaled down to a blog column or a slide. Only
  override a size when the preset genuinely does not fit (for example the 24
  hourly labels of a polar clock), and then stay close to it. Size the figure
  to the larger fonts rather than shrinking the fonts back down.

## Before handing off a change

Validate with the same environment, not a system interpreter:

```bash
pixi run python -c "import skrub, polars, sklearn"
pixi run quarto render path/to/changed.qmd --output-dir /tmp/check
```

Remove scratch output from `/tmp` and do not leave generated files untracked
unless they are intended deliverables.
