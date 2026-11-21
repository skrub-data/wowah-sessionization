# What Data Ops make easier

A companion to `skrub-dataops.qmd`. The deck argues that Data Ops simplify the
implementation; this document is the catalogue behind that claim, organised by
the operation a data scientist actually performs rather than by API.

Every example is taken from this repository's plan and was run against it.

---

## 1. Writing the pipeline

**The operation.** Turn a sequence of data transformations into something that
can be fitted, scored, and replayed on new data.

**Without Data Ops.** The transformations live in a script. To replay them you
either re-run the script or hand-write a transformer that wraps each step. The
two drift apart, and the drift is silent.

**With Data Ops.** The script *is* the pipeline. `skrub.var` marks the input,
`.skb.apply_func` and `.skb.apply` are the steps, and the result is a `DataOp`
that can be fitted:

```python
raw_files = skrub.var("raw_files")
historical_data = raw_files.skb.apply_func(load)
user_month = historical_data.skb.apply_func(build_churn_dataset)
X = user_month["char", "month"].skb.mark_as_X(cv=Splitter())
y = user_month["churn_next"].skb.mark_as_y()
```

Nothing is executed at this point. The same object is the specification and the
implementation, so there is no second copy to keep in sync.

**What it buys.** `make_learner()` gives a fitted object with `fit`/`predict`
that re-runs the whole graph — parsing included — on whatever data you pass it.
That is what makes the held-out evaluation in this project possible: the model
is fitted on the historical files and then predicts from raw `.txt` files it has
never seen, through the identical code path.

---

## 2. Splitting data the way the problem requires

**The operation.** Choose which rows train and which rows are scored, when the
answer is not "random 80/20".

**Without Data Ops.** `cross_validate` takes a `cv` argument, so a custom
splitter is possible — but it only sees `X` and `y` as arrays. Anything the
splitter needs beyond that (a month column, a group id) has to be smuggled into
`X` and stripped out again downstream.

**With Data Ops.** The splitter is attached where the data is marked, and it
receives the actual frame:

```python
X = user_month["char", "month"].skb.mark_as_X(cv=Splitter())
```

`Splitter.split` filters on a `month` column with polars. It is ordinary
dataframe code, not array indexing.

**What it buys.** The splitter is honoured everywhere — `cross_validate`,
`train_test_split`, `make_randomized_search`, and scoring all use it. There is
one definition of "a fold" and no way for the search to use a different one than
the evaluation.

**Also available.** `op.skb.iter_cv_splits(env)` yields each fold as a dict with
`X_train`, `X_test`, `y_train`, `y_test` and the row indices, which is how you
inspect what the splitter actually did without re-deriving it.

---

## 3. Tuning decisions that are not model hyperparameters

**The operation.** Decide things like "how long a gap ends a session", "should
this feature block exist", "how many lags to add" — choices that change the
*shape* of the pipeline, not a number inside an estimator.

**Without Data Ops.** These are either hard-coded and never tested, or explored
by editing the script and re-running it, which means the comparison lives in
your memory rather than in a results table.

**With Data Ops.** Any value can be replaced by a choice, inline, where it is
used:

```python
session_gap = skrub.choose_from([30, 45, 60], name="session_gap")
use_location = skrub.choose_bool(name="location_features")
lags = skrub.choose_from({"no": None, "1": (1,), "1_2": (1, 2)}, name="lags")
```

and the whole graph becomes a search space:

```python
search = data_op.skb.make_randomized_search(backend="optuna", n_iter=40)
```

**What it buys.** The search space is declared next to the code that consumes
it, so it cannot drift from the implementation. `describe_param_grid()` prints
the space, `describe_defaults()` prints what the untuned pipeline will use, and
`get_choices()` returns the choice objects.

**The four choice types**, each with a different meaning:

| Call | Meaning |
|---|---|
| `choose_from([...])` | one of a fixed set of values |
| `choose_bool()` | `True` / `False` |
| `choose_float(lo, hi, log=True)` | a number from a range |
| `optional(step)` | a **step** is present or absent |

`optional` is the one that is easy to miss: it makes a pipeline *step*
conditional, so the grid contains `[DropCols(...), None]` and `None` means the
step is skipped entirely. `choose_bool` makes a *value* conditional — the step
always runs and receives `True`/`False`.

---

## 4. Expressing dependencies between choices

**The operation.** Some knobs only make sense when another knob is on. The
zone-session gap is meaningless without the location block; the Gini coefficient
is computed from zone sessions.

**Without Data Ops.** A flat grid contains those invalid cells. Each one costs a
trial and, here, a cache entry.

**With Data Ops.** `choice.match({...})` expresses the dependency:

```python
add_gini = use_location.match({
    True: skrub.choose_bool(default=False, name="add_gini"),
    False: False,
})
```

**What it buys.** Both the grid description and the sampler respect it. The grid
shows `gini` only under the `loc: True` branch, and an Optuna trial that draws
`loc=False` records no `gini` parameter at all — the invalid cell is never
sampled.

**One constraint.** Two *different* choice objects sharing a name, both
reachable, raise `ValueError: Choice and node names must be unique`. Build the
nested choice once and reuse the object in the `match`. An unused choice does
not leak into the plan, so eager construction is safe.

---

## 5. Carrying a result from one stage to the next

**The operation.** Run a search, take its winner, and use it as the starting
point for the next search.

**Without Data Ops.** Copy the winning values into the script by hand. The
transcription is where mistakes happen, and the search that produced them is no
longer connected to the code that uses them.

**With Data Ops.** Two mechanisms, for two different situations.

*Pinning a choice* — bind its name in the environment:

```python
op.skb.eval({"date_window": window, "session_gap": 45})
```

*Pinning a whole group* — pass a `frozen` mapping to the plan builder, which is
what this repo does so that a stage's winner is a value rather than a choice.

**What it buys.** The winner of the feature search becomes an input to the
structural search without being retyped, and the study database keeps the record
of where it came from.

**The trap.** The environment override is silent: `describe_param_grid()` still
lists the choice, so a plan can look tunable while every trial receives the same
value. Keep the override out of the environment during a search — that is why
`frozen` is a separate argument here.

---

## 6. Reading a finished search

**The operation.** Turn "the search finished" into "the pipeline uses the
result".

**Without Data Ops.** The search returns a dict of parameter names that mean
nothing to the pipeline builder.

**With Data Ops.** The study is queryable, and the plan can be rebuilt to
translate its output:

```python
study = optuna.load_study(study_name="wowah_churn_gaps", storage=...)
study.best_params   # {'0:session_gap': '2:60', '1:zone_session_gap': '0:same'}
```

The key is `<choice id>:<name>` and the value is `<outcome index>:<label>`.
Rebuild the plan for the same stage, read each choice's `outcomes` list, and map
the index back — about twenty lines, in `main.best_params_from_study`.

**What it buys.** The search result becomes a `frozen` mapping that the next
stage consumes directly, with no hand transcription.

**Also available.** `study_name` plus `storage` makes a run resumable:
`load_if_exists=True` picks up where an interrupted search stopped. Without a
name, skrub generates a fresh UUID per call and nothing can be resumed.

---

## 7. Inspecting a pipeline

**The operation.** Find out what the pipeline actually does, and what it did on
this data.

**Without Data Ops.** Print statements, or re-run steps by hand to look at
intermediate frames.

**With Data Ops.** Several levels, from cheap to thorough:

| Call | What you get |
|---|---|
| `describe_steps()` | the graph as a numbered list of steps |
| `draw_graph()` | the graph as a diagram |
| `find("name")` | the node with that name, for further use |
| `report(environment=...)` | an HTML report: graph, per-node timing, fitted state |
| `preview()` | a cheap look at intermediate results |

**What it buys.** The report is the one that matters for a slow pipeline: it
shows per-node wall time, so the expensive step is identified rather than
guessed. In this project it is the location block, which is why that block is a
choice rather than a fixed part of the plan.

**Naming nodes** is what makes the rest work. `set_name("features")` on the
assembled feature table is what lets `find("features")` and
`truncated_after("features")` locate it later.

---

## 8. Reusing part of a fitted pipeline

**The operation.** A model search re-fits the expensive prefix — here the
feature table and the vectorizer — on every trial, even though it is identical
across all of them.

**Without Data Ops.** Cache the intermediate frame by hand, and remember to
invalidate it when the code changes.

**With Data Ops.** Name the node, then truncate the fitted learner at it:

```python
all_features = X.skb.apply_func(add_features, feature_table).skb.set_name("features")

learner = op.skb.make_learner().fit(env)
prefix = learner.truncated_after("features")
prefix.transform(env)      # the feature table, no model refit
```

**What it buys.** The expensive prefix is computed once and reused. It is also
the cleanest way to answer "what does the model actually see?" — the output is
the assembled feature table, before encoding.

**Related.** `freeze_after_fit()` on a node computes it during `fit()` and
reuses it during `transform()`/`predict()`, which is the same idea expressed
inside the plan rather than outside it.

---

## 9. Iterating quickly

**The operation.** Run the pipeline many times while developing.

**Without Data Ops.** Every run pays for every step, including the ones that did
not change.

**With Data Ops.** Two independent mechanisms.

*Caching* — one configuration call, and every node's result is memoised:

```python
skrub.set_config(cache=".skrub_cache")
```

*Subsampling* — configure a cheap preview, and opt in to it explicitly:

```python
op.skb.subsample(500)
op.skb.eval(env, keep_subsampling=True)
```

**What it buys.** Caching is what makes a search over a slow plan feasible at
all: without it, every trial re-runs the parse. Subsampling is what makes the
*development* loop fast, without changing what a real run does — subsampling is
off unless you ask for it.

**The sharp edge.** joblib keys a cached call on the cached function's own
source plus its arguments, not on the functions it calls. Editing a helper
leaves the cache key unchanged and the cache keeps serving stale results,
silently. The workaround is to pass a source fingerprint as an extra argument of
the cached function, which is what `FEATURES_FINGERPRINT` does here.

---

## 10. Scoring and metrics

**The operation.** Decide what "better" means, and make every part of the
pipeline agree on it.

**Without Data Ops.** The metric is passed separately to each call, and it is
easy for the search to optimise one thing while the report shows another.

**With Data Ops.** The metric is attached to the plan:

```python
data_op = encoded.skb.apply(model, y=y).skb.with_scoring("roc_auc")
```

**What it buys.** `cross_validate`, `make_randomized_search` and `eval` all use
it. The search optimises the metric the report shows, by construction.

---

## Summary

| Operation | Data Ops feature |
|---|---|
| Writing the pipeline | `var`, `apply_func`, `apply`, `mark_as_X`/`mark_as_y` |
| Splitting data | `mark_as_X(cv=...)`, `iter_cv_splits` |
| Tuning architecture | `choose_from`, `choose_bool`, `choose_float`, `optional` |
| Dependent choices | `choice.match({...})` |
| Carrying results forward | environment override, `frozen` mapping |
| Reading a search | `study.best_params`, `get_choices`, `describe_param_grid` |
| Inspecting | `describe_steps`, `draw_graph`, `find`, `report`, `preview` |
| Reusing a prefix | `set_name`, `truncated_after`, `freeze_after_fit` |
| Iterating | `set_config(cache=...)`, `subsample` |
| Metrics | `with_scoring` |

The through-line: each of these is a thing you would otherwise do by hand, in a
second script, with no connection to the pipeline. Data Ops put it in the graph,
where it is declared once and honoured everywhere.
