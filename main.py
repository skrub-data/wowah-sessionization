"""
Predict, for each character and month, whether the character will churn in the
next month.

Features for a row at month M come from month M-1 only, and the splitter
iterates by month, so nothing from the future leaks into training.
"""

# %%
import hashlib
import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import skrub
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from skrub import SessionEncoder

from add_churn import build_churn_dataset
from src.parse_to_single_table import list_raw_files, parse_files
from src.utils import sample_by_user

from adding_features import (
    add_general_features,
    get_session_duration,
    add_location_features,
    add_lagged_features,
)

MIN_DATE = datetime.strptime("2006-02-01", "%Y-%m-%d")
# The last test month is November: its target needs December, and the January
# 2009 data stops on the 10th.
MAX_DATE = datetime.strptime("2008-10-01", "%Y-%m-%d")

# Raw WoWAH logs: one ".txt" heartbeat file every ~10 minutes, under
# "YYYY-MM-DD" day folders.
RAW_DATA_ROOT = "data/raw/WoWAH"

# skrub's DataOp cache. Parsing is a node of the plan, so without a cache it is
# re-run on every fit and every score. Git-ignored; delete it to force a
# recompute.
CACHE_DIR = Path(__file__).parent / ".skrub_cache"
skrub.set_config(cache=str(CACHE_DIR))


def _source_fingerprint(*modules):
    """
    Hash the source of the modules a cached node depends on.

    joblib keys a cached call on the cached function's own source plus its
    arguments, not on the functions it calls. Passing this fingerprint as an
    argument puts those dependencies back into the key.
    """
    digest = hashlib.sha256()
    for module in modules:
        digest.update(Path(module.__file__).read_bytes())
    return digest.hexdigest()[:16]


# `main.py` is included because `build_feature_table` calls helpers defined
# here. The parse node is not fingerprinted: the parser is frozen, so editing
# `src/parse_to_single_table.py` is not detected -- delete `.skrub_cache/` by
# hand if you ever do.
FEATURES_FINGERPRINT = _source_fingerprint(
    sys.modules[__name__],
    sys.modules["add_churn"],
    sys.modules["adding_features"],
)


# "historical" is used to fit the pipeline. October 2008 is the last month
# because `churn_next` needs two months to measure.
HISTORICAL_END = datetime.strptime("2008-10-31", "%Y-%m-%d")

# "unseen" is never used while fitting. It overlaps October so that November's
# features, built from October's logs, are not degenerate.
UNSEEN_START = datetime.strptime("2008-10-01", "%Y-%m-%d")
UNSEEN_END = datetime.strptime("2009-01-10", "%Y-%m-%d")

ENV_KINDS = ("train", "unseen")
DEFAULT_ENV_KIND = "train"


class Splitter:
    """
    Expanding-window time split: for each split point, train on everything up
    to the previous month and test on the following one.

    The target of a row is the churn status of the month after it, so the
    training set stops one month before the split point.
    """

    def split(self, user_month, has_played=None):
        """
        Yield (train_idx, test_idx) per split point.

        `has_played` is the target, passed positionally by scikit-learn. It is
        used to skip folds whose training set has a single class: the first
        split point has almost no history under `only_prev_active`, and such a
        fold cannot train a classifier.
        """
        time_range = pl.date_range(MIN_DATE, MAX_DATE, "1mo", eager=True)
        for split_point in time_range:
            test_month = pl.Series([split_point]).dt.offset_by("1mo").first()
            last_train_month = pl.Series([split_point]).dt.offset_by("-1mo").first()
            train_idx = (
                user_month.with_row_index("idx")
                .filter(pl.col("month") <= last_train_month)["idx"]
                .to_list()
            )
            test_idx = (
                user_month.with_row_index("idx")
                .filter(pl.col("month") == test_month)["idx"]
                .to_list()
            )
            if not (train_idx and test_idx):
                continue
            if has_played is not None:
                train_target = has_played[train_idx]
                if train_target.n_unique() < 2:
                    print(f"Skipping month {split_point}: single-class training set")
                    continue
            print(f"Working on month {split_point}")
            yield train_idx, test_idx

    def get_n_splits(self, X, y):
        return sum(1 for _ in self.split(X, y))


def filter_df_by_month(df, month):
    return df.filter(pl.col("month") == month)


# %%
def build_feature_table(
    historical_data,
    session_gap=30,
    zone_session_gap=None,
    use_location=True,
    add_gini=False,
    lags=(1, 2),
    diff_lags=(1,),
    use_monthly_features=True,
    use_class_features=True,
    _fingerprint=None,
):
    """
    Build the feature table for every (char, month) pair.

    Features for a row at month M come from month M-1 only, to avoid leakage.

    The table covers every (char, month) pair, not just the rows of a CV fold,
    because lagged features need each character's full monthly timeline.

    `_fingerprint` is unused; it only exists so the cache key changes when the
    code this function calls into changes (see `FEATURES_FINGERPRINT`).

    `zone_session_gap=None` reuses `session_gap`.
    """
    del _fingerprint
    if zone_session_gap is None:
        zone_session_gap = session_gap
    features_by_month = []

    # Stateless, so it is refitted for every month.
    session_encoder = SessionEncoder(
        split_by="char", timestamp_col="timestamp", session_gap=session_gap * 60
    )
    historical_data = historical_data.with_columns(
        month=pl.col("timestamp").dt.truncate("1mo")
    ).lazy()

    # Every (char, month) pair, whether or not the character played that month.
    user_month = build_churn_dataset(historical_data).select("char", "month").lazy()
    # Fixed per-character attributes.
    user_month = user_month.join(
        historical_data.select("char", "race", "charclass").unique("char"),
        on="char",
        how="left",
        maintain_order="left",
    )
    # Tenure: months since the character was first seen. `first_month` is
    # otherwise dropped, and without it the model has no explicit account-age
    # signal -- it reads age off `char`, which is an id assigned at creation.
    user_month = user_month.join(
        historical_data.group_by("char")
        .agg(pl.col("month").min().alias("first_month"))
        .with_columns(
            first_month=pl.col("first_month").dt.year() * 12
            + pl.col("first_month").dt.month()
        ),
        on="char",
        how="left",
        maintain_order="left",
    ).with_columns(
        tenure=(
            pl.col("month").dt.year() * 12
            + pl.col("month").dt.month()
            - pl.col("first_month")
        ).clip(0, 36)
    ).drop("first_month").collect()

    for month in user_month["month"].unique().sort():
        this_month_X = filter_df_by_month(user_month, month)

        # Shift forward by one month and filter on the current month, which is
        # equivalent to keeping the previous month's rows.
        kept_historical_data = historical_data.with_columns(
            pl.col("month").dt.offset_by("1mo")
        ).filter(pl.col("month") == month)

        historical_data_sessions = session_encoder.fit_transform(
            kept_historical_data.collect()
        ).lazy()
        historical_data_sessions = get_session_duration(historical_data_sessions)

        df_with_features = add_general_features(
            this_month_X.lazy(),
            historical_data_sessions,
            use_monthly_features=use_monthly_features,
            use_class_features=use_class_features,
        )

        # The location block is the slowest part of the table.
        if use_location:
            session_encoder_zone = SessionEncoder(
                split_by=["char", "zone"],
                timestamp_col="timestamp",
                session_gap=zone_session_gap * 60,
                suffix="session_id_zone",
            )
            historical_data_zone_sessions = session_encoder_zone.fit_transform(
                kept_historical_data.collect()
            ).lazy()
            historical_data_zone_sessions = get_session_duration(
                historical_data_zone_sessions,
                session_column="timestamp_session_id_zone",
            )
            df_with_features = add_location_features(
                df_with_features,
                historical_data_zone_sessions,
                add_gini=add_gini,
            )
        else:
            df_with_features = df_with_features.collect()
        features_by_month.append(df_with_features)
        assert len(df_with_features) == len(this_month_X)

    feature_table = pl.concat(features_by_month, how="vertical")
    to_fill = pl.col("monthly_total_session_duration", "monthly_avg_session_duration")
    feature_table = feature_table.with_columns(
        to_fill.fill_null(pl.duration(seconds=0))
    )
    if lags is not None:
        feature_table = add_lagged_features(
            feature_table, lags=lags, diff_lags=diff_lags
        )
    return feature_table


def add_features(X, feature_table):
    """Attach this CV fold's (char, month) rows to the precomputed features."""
    return X.join(
        feature_table, on=["char", "month"], how="left", maintain_order="left"
    )


def load(file_paths, fraction=0.1):
    """Parse the given raw ".txt" log files, then subsample by character."""
    df = parse_files(file_paths)
    if fraction == 1:
        return df.lazy()

    df = sample_by_user(df, fraction=fraction)
    return df.lazy()


# The full corpus, written by `build_parquet.py` from the same `parse_files`
# the plan uses. It is a strict superset of `list_raw_files`: a heartbeat at
# 23:57 lives in the next day's folder, so filtering by folder name drops it.
PARSED_PARQUET = "data/wowah_data_all.parquet"


def load_parquet(start_date=None, end_date=None, fraction=0.1):
    """
    Read the pre-parsed corpus instead of parsing the raw files.

    Same data as `load`, so it can replace the parse node without changing any
    downstream result. The date bounds apply to `timestamp` rather than to
    folder names. Both bounds are inclusive, matching `list_raw_files`.
    """
    lf = pl.scan_parquet(PARSED_PARQUET)
    if start_date is not None:
        lf = lf.filter(pl.col("timestamp") >= start_date)
    if end_date is not None:
        # `end_date` is a day, so include the whole day.
        lf = lf.filter(pl.col("timestamp") < end_date + timedelta(days=1))
    if fraction == 1:
        return lf
    # `sample_by_user` needs a materialised frame to draw the character sample.
    return sample_by_user(lf.collect(), fraction=fraction).lazy()


# %%
# Choices are grouped by what they cost to evaluate, which decides how they can
# be searched:
#
# - FEATURE_CHOICES change `build_feature_table`'s arguments, so each
#   combination is a separate cache entry and a full rebuild.
# - STRUCTURAL_CHOICES only affect what happens after the feature table, so a
#   trial costs one refit per fold.
# - CAPACITY_CHOICES are the model's own hyperparameters.
FEATURE_CHOICES = (
    "session_gap",
    "zone_session_gap",
    "lags",
    "diff_lags",
    "location_features",
    "add_gini",
    "use_monthly_features",
    "use_class_features",
)
STRUCTURAL_CHOICES = (
    "drop_char",
    "drop_month",
    "cardinality_threshold",
    "high_cardinality",
    "drop_null_fraction",
    "logreg_C",
)
CAPACITY_CHOICES = (
    "learning_rate",
    "max_leaf_nodes",
    "min_samples_leaf",
    "l2_regularization",
)

# Which choices each stage declares. An undeclared choice takes its default.
STAGE_CHOICES = {
    "features": FEATURE_CHOICES,
    "structural": STRUCTURAL_CHOICES,
    "capacity": CAPACITY_CHOICES,
    # The feature-table knobs whose ranking was inside the noise of a 2% sample,
    # re-tested at 10% with the rest of the feature group frozen.
    "gaps": ("session_gap", "zone_session_gap", "use_class_features"),
    "full": FEATURE_CHOICES + STRUCTURAL_CHOICES + CAPACITY_CHOICES,
    # No choice at all: every knob takes its default. Used for the final fit.
    "none": (),
}


def make_data_op(
    load_fraction=0.1,
    return_x=False,
    stage="full",
    frozen=None,
    model_kind=None,
    source="parquet",
):
    """
    Build the churn-prediction plan.

    `source` is "parquet" (read the pre-parsed corpus) or "raw" (parse the
    `.txt` files). Both go through the same `parse_files` code, so the choice
    does not change any result -- see `build_parquet.py`.

    `stage` selects which group of choices is declared (see `STAGE_CHOICES`).

    `frozen` maps a choice name to a value, pinning it regardless of `stage`.
    This is how one stage's winner is carried into the next.

    `model_kind` is "logreg" or "hgb"; `None` derives it from `stage`.
    """
    frozen = {} if frozen is None else dict(frozen)
    active = set(STAGE_CHOICES[stage])
    if model_kind is None:
        model_kind = "logreg" if stage in ("structural", "gaps") else "hgb"

    def chosen(name, value, default):
        """
        Use `value` if this stage searches `name`, else `default`.

        An unused choice does not leak into the plan, so `value` is built
        eagerly. Two *different* choice objects sharing a name, both reachable,
        raise "Choice and node names must be unique" -- which is why the nested
        choices below are built once and reused in their `match`.
        """
        if name in frozen:
            return frozen[name]
        if name in active:
            return value
        return default

    # Loading is a node of the plan, so the same graph runs on whichever data
    # the environment points to.
    raw_files = skrub.var("raw_files")
    if source == "parquet":
        historical_data = skrub.var("date_window").skb.apply_func(
            lambda window: load_parquet(
                start_date=window["start"],
                end_date=window["end"],
                fraction=load_fraction,
            )
        )
    elif source == "raw":
        historical_data = raw_files.skb.apply_func(load, fraction=load_fraction)
    else:
        raise ValueError(f"source must be 'raw' or 'parquet', got {source!r}")
    # "guild == -1" means "no guild".
    historical_data = historical_data.with_columns(
        pl.when(pl.col("guild") == -1)
        .then(None)
        .otherwise(pl.col("guild"))
        .alias("guild")
    )

    # Only score characters that played in the previous month: the others have
    # no features.
    user_month_has_played = historical_data.skb.apply_func(
        build_churn_dataset, only_prev_active=True
    ).skb.set_name("query")
    X = user_month_has_played["char", "month"].skb.mark_as_X(cv=Splitter())
    y = user_month_has_played["churn_next"].skb.mark_as_y()

    # --- Feature-table choices -------------------------------------------
    # 15 minutes is not offered: heartbeats arrive every ~10 minutes, so it
    # would split a session whenever a single poll is missed.
    session_gap = chosen(
        "session_gap",
        skrub.choose_from([30, 45, 60], name="session_gap"),
        30,
    )
    # Zone dwell is a different process from play sessions, so it gets its own
    # gap. "same" reuses `session_gap`.
    zone_gap_choice = skrub.choose_from(
        {"same": None, "30": 30, "60": 60}, name="zone_session_gap"
    )
    lags = chosen(
        "lags",
        skrub.choose_from(
            {"no": None, "1": (1,), "1_2": (1, 2), "1_2_3": (1, 2, 3)}, name="lags"
        ),
        (1, 2),
    )
    # Nested under `lags`: the lag block is skipped entirely when `lags` is
    # None, so `diff_lags` would be ignored.
    diff_lags_choice = skrub.choose_from({"1": (1,), "1_2": (1, 2)}, name="diff_lags")
    if hasattr(lags, "match"):
        diff_lags_choice_obj = chosen("diff_lags", diff_lags_choice, (1,))
        diff_lags = lags.match(
            {
                None: (1,),
                (1,): diff_lags_choice_obj,
                (1, 2): diff_lags_choice_obj,
                (1, 2, 3): diff_lags_choice_obj,
            }
        )
    else:
        diff_lags = (
            chosen("diff_lags", diff_lags_choice, (1,)) if lags is not None else (1,)
        )
    use_location = chosen(
        "location_features", skrub.choose_bool(name="location_features"), True
    )
    # Nested under `location_features`: the zone encoder only exists when the
    # location block is on, and the Gini coefficient is computed from the zone
    # sessions. `match` keeps those invalid cells out of the grid.
    add_gini_choice = skrub.choose_bool(default=False, name="add_gini")
    if hasattr(use_location, "match"):
        zone_session_gap = use_location.match(
            {
                True: chosen("zone_session_gap", zone_gap_choice, None),
                False: None,
            }
        )
        add_gini = use_location.match(
            {
                True: chosen("add_gini", add_gini_choice, False),
                False: False,
            }
        )
    else:
        zone_session_gap = (
            chosen("zone_session_gap", zone_gap_choice, None) if use_location else None
        )
        add_gini = (
            chosen("add_gini", add_gini_choice, False) if use_location else False
        )
    use_monthly_features = chosen(
        "use_monthly_features",
        skrub.choose_bool(name="use_monthly_features"),
        True,
    )
    use_class_features = chosen(
        "use_class_features",
        skrub.choose_bool(name="use_class_features"),
        True,
    )

    # A plain, non-X node, so lagged features see each character's complete
    # monthly timeline regardless of which fold is being evaluated.
    feature_table = historical_data.collect().skb.apply_func(
        build_feature_table,
        session_gap=session_gap,
        zone_session_gap=zone_session_gap,
        use_location=use_location,
        add_gini=add_gini,
        lags=lags,
        diff_lags=diff_lags,
        use_monthly_features=use_monthly_features,
        use_class_features=use_class_features,
        _fingerprint=FEATURES_FINGERPRINT,
    )
    # Named so `learner.truncated_after("features")` returns the feature table
    # without refitting the vectorizer or the model.
    all_features = X.skb.apply_func(add_features, feature_table).skb.set_name(
        "features"
    )

    # --- Structural choices ----------------------------------------------
    # `char` is an identifier, not a feature. Exposed as a choice so the search
    # can show whether it helps; `default=None` keeps the column.
    all_features = all_features.skb.apply(
        chosen(
            "drop_char",
            skrub.optional(
                skrub.DropCols(["char"]), name="drop_char", default=None
            ),
            None,
        )
    )
    # `month` expands to year/month/day/total_seconds, which mostly encodes
    # "which fold is this".
    all_features = all_features.skb.apply(
        chosen(
            "drop_month",
            skrub.optional(
                skrub.DropCols(["month"]), name="drop_month", default=None
            ),
            None,
        )
    )

    cardinality_threshold = chosen(
        "cardinality_threshold",
        skrub.choose_int(10, 100, log=True, name="cardinality_threshold"),
        40,
    )
    high_cardinality = chosen(
        "high_cardinality",
        skrub.choose_from(
            {
                "string": skrub.StringEncoder(n_components=30),
                "minhash": skrub.MinHashEncoder(),
            },
            name="high_cardinality",
        ),
        skrub.StringEncoder(n_components=30),
    )
    drop_null_fraction = chosen(
        "drop_null_fraction",
        skrub.choose_float(0.5, 1.0, name="drop_null_fraction"),
        1.0,
    )
    encoded = all_features.skb.apply(
        skrub.TableVectorizer(
            cardinality_threshold=cardinality_threshold,
            high_cardinality=high_cardinality,
            drop_null_fraction=drop_null_fraction,
        )
    )

    # --- Model ------------------------------------------------------------
    logreg_C = chosen(
        "logreg_C",
        skrub.choose_float(1e-3, 10, log=True, name="logreg_C"),
        1.0,
    )
    learning_rate = chosen(
        "learning_rate",
        skrub.choose_float(0.02, 0.3, log=True, n_steps=8, name="learning_rate"),
        0.1,
    )
    max_leaf_nodes = chosen(
        "max_leaf_nodes",
        skrub.choose_int(15, 63, log=True, name="max_leaf_nodes"),
        31,
    )
    min_samples_leaf = chosen(
        "min_samples_leaf",
        skrub.choose_int(20, 200, log=True, name="min_samples_leaf"),
        20,
    )
    l2_regularization = chosen(
        "l2_regularization",
        skrub.choose_float(1e-3, 10, log=True, name="l2_regularization"),
        0.0,
    )

    if model_kind == "logreg":
        # The lag columns are null for a character's first months and a linear
        # model cannot consume nulls. `SquashingScaler` is what skrub's
        # `tabular_pipeline` uses for linear models.
        model = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", skrub.SquashingScaler()),
                ("model", LogisticRegression(C=logreg_C, max_iter=1000)),
            ]
        )
    elif model_kind == "hgb":
        model = HGB(
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            # The internal validation split is random, so it would leak across
            # the time folds.
            early_stopping=False,
        )
    else:
        raise ValueError(f"model_kind must be 'logreg' or 'hgb', got {model_kind!r}")

    data_op = encoded.skb.apply(model, y=y).skb.with_scoring("roc_auc")
    if return_x:
        return data_op, X, y
    return data_op


# %%


def get_env(env_kind=DEFAULT_ENV_KIND, source="parquet"):
    """
    Build the environment for one of the two data selections.

    "train" is everything up to October 2008; "unseen" is never used while
    fitting. `source` decides which key the environment carries: "raw" gets the
    list of `.txt` files, "parquet" gets the date window.
    """
    if env_kind == "train":
        start, end = None, HISTORICAL_END
    elif env_kind == "unseen":
        start, end = UNSEEN_START, UNSEEN_END
    else:
        raise ValueError(f"env_kind must be one of {ENV_KINDS}, got {env_kind!r}")

    if source == "parquet":
        return {"date_window": {"start": start, "end": end}}
    if source == "raw":
        return {"raw_files": list_raw_files(RAW_DATA_ROOT, start_date=start, end_date=end)}
    raise ValueError(f"source must be 'raw' or 'parquet', got {source!r}")


def _ensure_study(study_name, storage):
    """Create the study up front so the search only ever loads it."""

    if study_name is None:
        return
    import optuna

    optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )


def random_search(
    env_kind=DEFAULT_ENV_KIND,
    load_fraction=0.1,
    stage="full",
    n_iter=16,
    study_name=None,
    timeout=None,
    frozen=None,
    model_kind=None,
    source="parquet",
):
    """
    Wrap the plan in an Optuna study.

    The search runs sequentially. Parallel trials are not offered -- see
    AGENTS.md, "Do not parallelise the search".

    `study_name` makes a run resumable; without it skrub generates a fresh UUID
    per call.
    """
    storage = "sqlite:///wowah_churn_study.db"
    _ensure_study(study_name, storage)
    data_op = make_data_op(
        load_fraction=load_fraction,
        stage=stage,
        frozen=frozen,
        model_kind=model_kind,
        source=source,
    )
    search = data_op.skb.make_randomized_search(
        backend="optuna",
        n_jobs=1,
        n_iter=n_iter,
        study_name=study_name,
        timeout=timeout,
        storage=storage,
        # A trial that raises is a bug, not a bad hyperparameter combination:
        # let it surface instead of recording it as a NaN score.
        error_score="raise",
    )
    return search, get_env(env_kind, source=source)


def best_params_from_study(study_name, stage, frozen=None, storage=None):
    """
    Turn a finished study's best trial into a `frozen` mapping.

    Optuna stores a categorical choice as an index into its outcome list, so
    the raw `best_params` are strings like `'2:60'`. This rebuilds the plan for
    the same stage and maps each index back to the value.

    Returns `(frozen, score)`.
    """
    import optuna

    storage = storage or "sqlite:///wowah_churn_study.db"
    study = optuna.load_study(study_name=study_name, storage=storage)
    best = study.best_trial

    # Rebuild the plan with the same stage and the same frozen inputs, so the
    # choice ids and outcome lists match the ones the search used.
    data_op = make_data_op(load_fraction=0.1, stage=stage, frozen=frozen)
    choices = data_op.skb.get_choices(named_only=True)

    out = {} if frozen is None else dict(frozen)
    for key, value in best.params.items():
        # Keys look like "0:session_gap"; the part after the colon is the name.
        name = key.split(":", 1)[1]
        if name not in choices:
            continue
        choice = choices[name]
        outcomes = getattr(choice, "outcomes", None)
        if outcomes is None:
            # A numeric choice: Optuna already stored the value itself.
            out[name] = value
            continue
        # A categorical choice: the value is "<index>:<label>".
        idx = int(str(value).split(":", 1)[0])
        out[name] = outcomes[idx]
    return out, best.value


def evaluate(
    env_kind=DEFAULT_ENV_KIND,
    load_fraction=0.1,
    stage="full",
    frozen=None,
    source="parquet",
):
    data_op = make_data_op(
        load_fraction=load_fraction, stage=stage, frozen=frozen, source=source
    )
    results = data_op.skb.eval(get_env(env_kind, source=source))
    return results, data_op


MODEL_PATH = Path(__file__).parent / "reports" / "final_learner.joblib"


def fit_final(frozen, load_fraction=1.0, source="parquet", path=MODEL_PATH):
    """
    Fit the plan on the whole historical window and save the learner.

    `load_fraction=1.0` keeps every character, which is the point of the final
    fit: the searches ran on a sample, this does not.

    The learner is saved with joblib so that scoring does not have to refit.
    """
    import joblib

    data_op = make_data_op(
        load_fraction=load_fraction, stage="none", frozen=frozen, source=source
    )
    learner = data_op.skb.make_learner()
    learner.fit(get_env("train", source=source))
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(learner, path)
    print(f"saved learner to {path}")
    return learner


def score_unseen(path=MODEL_PATH, source="parquet"):
    """
    Load the saved learner and score November 2008 from the unseen window.

    Only November is scored: December's target is `churn` in January 2009, and
    the dataset stops on January 10th, so a character who played later that
    month is recorded as absent and the label is wrong for them.

    Returns a (char, month, churn_next_pred, churn_next) table, so the
    predictions can be scored against the labels.
    """
    import joblib

    learner = joblib.load(path)
    unseen_env = get_env("unseen", source=source)
    predictions = learner.predict(unseen_env)

    # Rebuild the (char, month, churn_next) rows for the same window. The plan
    # is deterministic, so the row order matches the predictions.
    data_op, X, y = make_data_op(
        load_fraction=1.0, return_x=True, stage="none", source=source
    )
    x_unseen = X.skb.eval(unseen_env)
    y_unseen = y.skb.eval(unseen_env)
    result = x_unseen.with_columns(
        pl.Series("churn_next_pred", predictions),
        pl.Series("churn_next", y_unseen),
    ).filter(pl.col("month") == datetime(2008, 11, 1))
    return result.sort(["month", "char"])


def predict_unseen(load_fraction=0.1, frozen=None, source="parquet"):
    """
    Fit on the historical files, then score November 2008 from the unseen ones.

    Kept for the sampled, in-memory path. `fit_final` + `score_unseen` is the
    full-data equivalent that saves the learner.
    """
    data_op, X, _y = make_data_op(
        load_fraction=load_fraction, return_x=True, frozen=frozen, source=source
    )
    learner = data_op.skb.make_learner()
    learner.fit(get_env("train", source=source))

    unseen_env = get_env("unseen", source=source)
    predictions = learner.predict(unseen_env)
    # Both calls are deterministic, so the row order matches.
    x_unseen = X.skb.eval(unseen_env)
    result = x_unseen.with_columns(pl.Series("churn_next_pred", predictions)).filter(
        pl.col("month") == datetime(2008, 11, 1)
    )
    return result.sort(["month", "char"])


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Evaluate the churn prediction DataOps plan."
    )
    parser.add_argument(
        "--env",
        choices=[*ENV_KINDS, "predict-unseen", "fit-final", "score-unseen"],
        default=DEFAULT_ENV_KIND,
        help=(
            "'train' evaluates the plan on the historical files, 'unseen' "
            "evaluates it on the held-out files, 'predict-unseen' fits on "
            "'train' and predicts November 2008, 'fit-final' fits on every "
            "character and saves the learner, 'score-unseen' loads that "
            "learner and scores November 2008"
        ),
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.1,
        help="fraction of characters to sample (1 keeps all of them)",
    )
    parser.add_argument(
        "--stage",
        choices=[*STAGE_CHOICES, "none"],
        default="full",
        help=(
            "which group of choices to declare; 'none' declares no choice at "
            "all, which is what the final full-data fit uses"
        ),
    )
    parser.add_argument(
        "--frozen",
        default=None,
        help=(
            "JSON object mapping choice names to values, pinning them "
            "regardless of --stage, e.g. '{\"session_gap\": 30}'"
        ),
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="run an Optuna search instead of a single evaluation",
    )
    parser.add_argument("--n-iter", type=int, default=16)
    parser.add_argument(
        "--study-name",
        default=None,
        help="Optuna study name; without it every run creates a new study",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=(
            "hard wall-clock bound on the search, in seconds. The search is "
            "sequential, so this is safe; note it counts wall-clock time, so a "
            "sleeping machine burns it"
        ),
    )
    parser.add_argument(
        "--source",
        choices=["parquet", "raw"],
        default="parquet",
        help=(
            "'parquet' reads the pre-parsed corpus (~1 s), 'raw' parses the "
            ".txt files (~75 s). Same data: build_parquet.py writes the "
            "parquet with parse_files"
        ),
    )
    args = parser.parse_args()

    frozen = json.loads(args.frozen) if args.frozen else None
    # JSON has no tuples, but `lags` and `diff_lags` are tuples of ints.
    if frozen:
        for key in ("lags", "diff_lags"):
            if isinstance(frozen.get(key), list):
                frozen[key] = tuple(frozen[key])

    if args.env == "fit-final":
        fit_final(frozen, load_fraction=args.fraction, source=args.source)
    elif args.env == "score-unseen":
        result = score_unseen(source=args.source)
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(result["churn_next"].to_numpy(), result["churn_next_pred"])
        print(f"November 2008: {len(result)} rows, ROC AUC {auc:.4f}")
        print(result.head(10))
    elif args.env == "predict-unseen":
        predictions = predict_unseen(load_fraction=args.fraction, frozen=frozen)
        print(predictions)
    elif args.search:
        search, env = random_search(
            args.env,
            load_fraction=args.fraction,
            stage=args.stage,
            n_iter=args.n_iter,
            study_name=args.study_name,
            timeout=args.timeout,
            frozen=frozen,
        )
        search.fit(env)
        print(f"best score: {search.best_score_}")
        print(f"best params: {search.best_params_}")
    else:
        results, data_op = evaluate(
            args.env,
            load_fraction=args.fraction,
            stage=args.stage,
            frozen=frozen,
            source=args.source,
        )
        report = data_op.skb.report(
            environment=get_env(args.env, source=args.source),
            output_dir="reports/execution_report",
            overwrite=True,
        )
