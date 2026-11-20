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
- `skrub-dataops.qmd` and `pain_points.qmd` are RevealJS slide decks;
  `draft.qmd` and `skrub-dataops.qmd` are the write-ups of record.

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
them takes minutes and is the single most expensive step in the project.

### The DataOp cache is on by default

`main.py` calls `skrub.set_config(cache=".skrub_cache")` at import time, so any
plan built with `make_data_op` reuses previously computed nodes. Measured on the
full corpus at `load_fraction=0.1`: a cold `cross_validate` parses 63 times and
takes ~413s; with the cache warm it parses once and takes ~69s.

The cache is git-ignored and lives at `.skrub_cache/` in the repository root.
Delete that directory to force a full recompute.

**The cache does not invalidate itself when the code changes.** joblib keys a
cached call on the *cached function's own source* plus its arguments — it does
not look at the functions that function calls. Editing `parse_files` would
therefore keep serving the old parse forever. `main.py` works around this by
passing a source fingerprint as an extra argument of the two expensive nodes:

| Node | Fingerprint covers |
|---|---|
| `load` | `src/parse_to_single_table.py`, `src/utils.py` |
| `build_feature_table` | `main.py`, `add_churn.py`, `adding_features.py` |

They are separate on purpose: editing feature code must not throw away the
(much more expensive) parsed logs. **If you add a new module that either node
calls into, add it to the corresponding `_source_fingerprint(...)` call**, or
edits to it will be silently ignored.

### A full parse already exists

Check these before writing any code that calls `parse_files` or `load`:

| File | Rows | Coverage |
|---|---|---|
| `data/wowah_data_all.parquet` | 35.6M | 2005-12-31 → 2009-01-10 (everything) |
| `data/wowah_data_2008.parquet` | 10.8M | 2008 only |
| `data/wowah_parsed_all.parquet` | 35.6M | same as `_all`, pre-rename columns |

For analysis, experiments and one-off checks, read one of these with
`pl.read_parquet(...)`. Only go through the raw `.txt` files when you are
specifically testing the parsing node itself.

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

A cold run of the whole plan is ~86s at `load_fraction=0.02` and ~7 minutes at
`load_fraction=0.1`; a warm run of the same is 1–3s. If a run is slow, check
whether the cache is actually being hit before assuming the code is at fault.

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

## Before handing off a change

Validate with the same environment, not a system interpreter:

```bash
pixi run python -c "import skrub, polars, sklearn"
pixi run quarto render path/to/changed.qmd --output-dir /tmp/check
```

Remove scratch output from `/tmp` and do not leave generated files untracked
unless they are intended deliverables.
