# Candidate pain points for Act 3

Not part of the deck. Each entry is a pain point the implementation actually hit,
with the Data Ops feature that answers it and the evidence behind the claim.

The five pain points currently in `skrub-dataops.qmd` are: data preparation as a
pipeline node, time-aware CV, tuning architecture rather than knobs, debugging,
and caching. The entries below are either new or sharpen one of those.

Everything marked **verified** was run against this repository's plan.

---

## 1. A choice's default is not the value you wrote

**Pain.** The untuned pipeline — `cross_validate`, `eval`, the final fit — uses
each choice's default. That default is not the value you passed first, and it
differs per choice type:

| Call | Default |
|---|---|
| `choose_from([15, 30, 60])` | `15` — the **first** outcome |
| `choose_float(0.1, 0.5)` | `0.3` — the **middle** |
| `choose_float(0.02, 0.3, log=True)` | `0.0775` — the geometric middle |
| `choose_int(15, 63, log=True)` | `31` |
| `choose_bool()` | `True` |
| `optional(step)` | the step is **present** |

`choose_from` has no `default=` parameter at all, so the only way to change its
default is to reorder the outcomes — which changes what the search reports.

**Why it bites.** Writing `choose_from([15, 30, 60])` reads as "30 is the
default, 15 and 60 are alternatives". It means the opposite. The pipeline you
report as "default hyperparameters" is silently the first outcome.

**Data Ops answer.** `choose_float`/`choose_int`/`choose_bool`/`optional` all
take `default=`. For `choose_from`, order the outcomes deliberately and say so.

**Verified.**

---

## 2. The environment can silently override a choice

**Pain.** Binding a choice's name in the environment pins it, bypassing the
search entirely:

```python
op.skb.eval({"v": 10, "k": 3})   # k is a choice; this returns 30, not 10
```

Nothing warns you. `describe_param_grid()` still lists the choice, so the plan
looks tunable while every trial gets the same value.

**Why it bites.** This is the mechanism for carrying one stage's winner into the
next — and it is also how you accidentally make a search meaningless. The two
uses are indistinguishable at the call site.

**Data Ops answer.** It is the right mechanism; the discipline is to keep the
override out of the environment during a search. In this repo `frozen=` is a
separate argument from the environment for exactly that reason.

**Verified.**

---

## 3. Nested choices keep invalid cells out of the search

**Pain.** Some knobs only exist when another knob is on: the zone-session gap is
meaningless without the location block, and the Gini coefficient is computed
from zone sessions. A flat grid contains those invalid cells, and each one costs
a trial and a cache entry.

**Data Ops answer.** `choice.match({...})` expresses the dependency, and both
the grid description and the sampler respect it:

```
- loc: True
  gini: [False, True]
- loc: False          <- no gini under this branch
```

An Optuna trial that draws `loc=False` records no `gini` parameter at all.

**Verified.**

---

## 4. `optional()` and `choose_bool()` are not the same thing

**Pain.** "Turn this feature off" can mean two different things: skip a *step*,
or pass a different *value* to a step that still runs.

**Data Ops answer.**

- `skrub.optional(transformer)` makes a **step** conditional — the grid contains
  `[PCA(...), None]`, and `None` means the step is skipped.
- `skrub.choose_bool()` makes a **value** conditional — the step always runs and
  receives `True`/`False`.

In this repo `drop_char` is an `optional(DropCols(["char"]))` because the step
itself should disappear; `location_features` is a `choose_bool` because
`build_feature_table` takes it as an argument.

**Verified.**

---

## 5. Optuna's `best_params` are indices, not values

**Pain.** A finished study reports:

```python
{'0:session_gap': '2:60', '1:location_features': '0:True'}
```

The key is `<choice id>:<name>` and the value is `<outcome index>:<label>`. That
is not something `make_data_op` can consume, and the index is only meaningful
against the exact plan that produced it.

**Data Ops answer.** Rebuild the plan for the same stage, read each choice's
`outcomes` list, and map the index back. `main.best_params_from_study` does this
in about 20 lines.

**Why it matters for the deck.** It is the seam between "the search found
something" and "the pipeline uses it", and it is not automatic.

**Verified.**

---

## 6. A search is only resumable if you name the study

**Pain.** Without `study_name`, skrub generates a fresh UUID per call. Every run
starts a new study, so an interrupted search cannot be resumed and the results
database fills with orphans. This repo's database accumulated six studies before
the name was set.

**Data Ops answer.** `make_randomized_search(study_name=..., storage=...)` with
`load_if_exists=True` resumes where it stopped.

**Verified** — the `gaps` and `structural` studies were resumed this way.

---

## 7. Extracting a sub-pipeline from a fitted plan

**Pain.** A model search re-fits the expensive prefix — here, the feature table
and the vectorizer — on every trial, even though it is identical across all of
them. In a plain script you would cache the intermediate frame by hand.

**Data Ops answer.** Name the node, then truncate the fitted learner at it:

```python
all_features = X.skb.apply_func(add_features, feature_table).skb.set_name("features")

learner = op.skb.make_learner().fit(env)
prefix = learner.truncated_after("features")
prefix.transform(env)          # 0.9 s, 82 columns, no model refit
```

`op.skb.find("features")` locates the node in the unfitted plan.

**Verified.**

---

## 8. Choice names must be unique across the whole graph

**Pain.** Two *different* choice objects sharing a name, both reachable, raise:

```
ValueError: Choice and node names must be unique.
```

This bites when a choice is constructed inside a branch — the natural way to
write "this knob only exists in that branch" produces two objects with one name.

**Data Ops answer.** Build the choice once and reuse the object in the `match`.
An unused choice does not leak into the plan, so eager construction is safe;
duplicate *names* are the only constraint.

**Verified.**

---

## 9. The cache key does not follow dependencies

**Pain.** joblib keys a cached call on the cached function's own source plus its
arguments. It does not look at the functions that function calls. Editing
`adding_features.py` leaves `build_feature_table`'s key unchanged, so the cache
keeps serving the old feature table — silently, with plausible-looking results.

**Data Ops answer.** Pass a source fingerprint as an extra argument of the
cached function, which puts the dependencies back into the key:

```python
FEATURES_FINGERPRINT = _source_fingerprint(
    sys.modules[__name__], sys.modules["add_churn"], sys.modules["adding_features"]
)
```

**Why it matters for the deck.** It is the sharp edge of pain point 5: caching
makes iteration fast, and a stale cache is worse than no cache because it fails
quietly.

**Verified** — this is the mechanism in `main.py` today.

---

## 10. Parallel trials are not safe with a process-pool node

**Pain.** Parsing is a node of the plan and it spawns a `ProcessPoolExecutor`
sized to the core count. N parallel search workers on a cold cache means N ×
cores processes. Measured on a 24 GB, 8-core machine at a 2% sample:

| workers | outcome |
|---|---|
| 1 | fine |
| 2 | 1.4× speedup |
| 4 | 2.6× speedup |
| 4, cold parse | slower than sequential |
| 8 | 40 GB of swap |

There is a second trap: skrub switches to Optuna's **threaded** parallelisation
whenever `timeout` is set *and* `n_jobs > 1`, and threads deadlock against the
parser's process pool — workers sit at 0% CPU and never finish.

**Data Ops answer.** None, and that is the honest slide. The search runs
sequentially here. The mitigation is the cache (pain point 5), not parallelism.

**Verified** — both failure modes were hit during this session.

---

## 11. `n_steps` bounds a numeric choice

**Pain.** A continuous `choose_float` gives the sampler and the cache an
unbounded number of distinct values, which is wasteful when the response is flat
over most of the range.

**Data Ops answer.** `choose_float(0.02, 0.3, log=True, n_steps=8)` discretises
the range into 8 values. Used here for `learning_rate`.

**Verified** — appears as a `DiscretizedNumericChoice` in the grid.

---

## 12. `error_score` decides whether a bug looks like a bad trial

**Pain.** `make_randomized_search` defaults `error_score` to `np.nan`, so a
trial that raises is recorded as a NaN score and the search moves on. A genuine
bug is indistinguishable from an unlucky hyperparameter combination.

**Data Ops answer.** `error_score="raise"` while developing. This is how the
single-class-fold bug surfaced: HGB silently fitted a constant, and logistic
regression raised.

**Verified.**

---

## Suggested grouping for the deck

If any of these go in, they cluster naturally:

- **Extend pain point 3** (tuning architecture): entries 1, 3, 4, 11 — the
  mechanics of writing a search space that means what you think.
- **Extend pain point 4** (debugging): entry 7 — pulling a sub-pipeline out of a
  fitted plan.
- **Extend pain point 5** (caching): entries 9, 10 — the two ways caching and
  parallelism bite.
- **New pain point** (from search to production): entries 2, 5, 6 — the seam
  between a finished study and a pipeline that uses its result.

Entry 8 is a one-line footnote at most.
