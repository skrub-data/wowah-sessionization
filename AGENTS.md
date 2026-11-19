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
