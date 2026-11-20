"""
Exploratory analysis of the WoWAH heartbeat table.

The raw dataset is one row per heartbeat: a snapshot, roughly every 10 minutes,
of a character that was online. This script profiles the table, the characters
and their activity over time, and looks at the gaps between heartbeats, which is
what sessionization (cutting a character's heartbeat stream into play sessions)
builds on.

Most of the work is done lazily with polars so the full file (35M rows) can be
scanned without loading it into memory. A sample of characters is drawn for the
model-facing part of the exploration, where `skrub.TableReport` summarizes the
columns and `skrub.column_associations` / `skrub.TableVectorizer` show how the
table would be prepared for a model.

Usage (from the repository root):

    /Users/rcap/venvs/dev/bin/python explore_data.py
    /Users/rcap/venvs/dev/bin/python explore_data.py --data data/wowah_data_2008.parquet
    /Users/rcap/venvs/dev/bin/python explore_data.py --gap-minutes 60 --sample-chars 500

Iterating on the figures does not need the full scan: the monthly table and the
gap histogram are cached under `<output-dir>/cache/`, so

    /Users/rcap/venvs/dev/bin/python explore_data.py --only-plots --plots characters --show

redraws a single figure from the cache and opens it in a window.

Outputs are printed to the console; the skrub HTML report and the figures are
written to `--output-dir` (plus a JSON report when the installed skrub version
supports it).
"""

# %%
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import polars as pl
import skrub

DEFAULT_DATA = "data/wowah_data_all.parquet"
# Thresholds (in minutes) at which we report the share of heartbeat gaps that
# would be considered "inside a session".
GAP_THRESHOLDS_MIN = (1, 5, 10, 30, 60, 180, 1440)

# A character is "regular" in a month when it is seen on at least this many
# distinct days and with at least this many heartbeats. This separates sustained
# activity from one-off observations (for example, the mass of level-1
# characters that appears in October 2008).
REGULAR_MIN_ACTIVE_DAYS = 2
REGULAR_MIN_HEARTBEATS = 5

# Expansion launches on the realm the dataset was collected from. WoWAH is from
# the "Light's Hope" realm in Taiwan, so it follows the Taiwan/Hong Kong release
# schedule, not the North American/European one:
#   The Burning Crusade: 2007-04-03 (NA/EU: 2007-01-16)
#   Wrath of the Lich King: 2008-11-18 (NA/EU: 2008-11-13)
# The third element is the side the label extends to: "right" means the text
# ends at the line and grows leftwards, which keeps it inside the figure for
# releases close to the right edge.
EXPANSIONS = (
    ("The Burning Crusade", datetime(2007, 4, 3), "right"),
    ("Wrath of the Lich King", datetime(2008, 11, 18), "right"),
)

# Vertical room (in points) left above the axes for the two-line labels.
EXPANSION_LABEL_PAD = 34


# %%
def heading(title):
    print(f"\n{title}\n{'-' * len(title)}")


def bucket_expr(column, bins):
    """
    Map `column` to a label per (low, high] interval.

    `bins` is a list of `(low, high, label)`; `high=None` means no upper bound.
    Written as a `when` chain because `Expr.cut` is deprecated in recent polars
    versions and its replacement is still experimental.
    """
    expression = None
    for low, high, label in reversed(bins):
        condition = pl.col(column) > low
        if high is not None:
            condition = condition & (pl.col(column) <= high)
        branch = pl.when(condition).then(pl.lit(label))
        expression = (
            branch
            if expression is None
            else pl.when(condition).then(pl.lit(label)).otherwise(expression)
        )
    return expression


def bucket_labels(bins):
    return [label for _, _, label in bins]


def quantile_table(df, columns, quantiles=(0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)):
    """One row per column, one column per quantile, plus the number of nulls."""
    rows = []
    for col in columns:
        row = {"metric": col, "nulls": df[col].null_count()}
        for q in quantiles:
            row[f"p{int(q * 100):02d}"] = df[col].quantile(q)
        rows.append(row)
    return pl.DataFrame(rows)


# %%
def overview(lf):
    """Shape of the table, coverage in time, cardinalities and missing values."""
    heading("1. Overview")
    n_rows = lf.select(pl.len()).collect().item()
    stats = lf.select(
        pl.col("timestamp").min().alias("first heartbeat"),
        pl.col("timestamp").max().alias("last heartbeat"),
        pl.col("char").n_unique().alias("characters"),
        pl.col("zone").n_unique().alias("zones"),
        pl.col("guild").n_unique().alias("guild ids"),
    ).collect().row(0, named=True)

    print(f"rows: {n_rows:,}")
    for key, value in stats.items():
        print(f"{key:<16}: {value}")
    print(f"\nschema: {dict(lf.collect_schema())}")

    nulls = lf.select(pl.all().null_count()).collect()
    shares = (
        nulls.transpose(include_header=True, column_names=["null_count"])
        .with_columns(
            pl.col("null_count").cast(pl.Float64).truediv(n_rows).alias("share")
        )
        .sort("null_count", descending=True)
    )
    print("\nmissing values:")
    print(shares)

    duplicates = (
        lf.select(pl.struct("char", "timestamp").is_duplicated().sum())
        .collect()
        .item()
    )
    print(f"\nrows sharing a (char, timestamp) with another row: {duplicates:,}")


# %%
def categoricals(lf):
    """Distribution of the fixed attributes, levels and zones."""
    heading("2. Attributes")

    for col in ("race", "charclass"):
        counts = lf.group_by(col).len().sort("len", descending=True).collect()
        counts = counts.with_columns(
            (pl.col("len") / pl.col("len").sum() * 100).round(2).alias("share_%")
        )
        print(f"\n{col}:")
        print(counts)

    levels = lf.select(
        pl.col("level").min().alias("min"),
        pl.col("level").max().alias("max"),
        pl.col("level").mean().round(2).alias("mean"),
        pl.col("level").median().alias("median"),
        pl.col("level").std().round(2).alias("std"),
    ).collect()
    print(f"\nlevel:\n{levels}")

    level_hist = (
        lf.with_columns(bucket=(pl.col("level") // 10) * 10)
        .group_by("bucket")
        .len()
        .sort("bucket")
        .collect()
    )
    print(f"\nlevel histogram (10-level buckets):\n{level_hist}")

    zones = lf.group_by("zone").len().sort("len", descending=True).head(15).collect()
    zones = zones.with_columns(
        (pl.col("len") / pl.col("len").sum() * 100).round(2).alias("share_of_top_%")
    )
    print(f"\ntop 15 zones:\n{zones}")

    guild = lf.select(
        pl.col("guild").is_not_null().mean().alias("share_with_guild"),
        pl.col("level")
        .filter(pl.col("guild").is_not_null())
        .mean()
        .alias("mean_level_with_guild"),
        pl.col("level")
        .filter(pl.col("guild").is_null())
        .mean()
        .alias("mean_level_without_guild"),
    ).collect()
    print(f"\nguild:\n{guild}")


# %%
def per_character(lf):
    """How many heartbeats, active days and zones does a character have?"""
    heading("3. Characters")
    per_char = (
        lf.group_by("char")
        .agg(
            n_events=pl.len(),
            active_days=pl.col("timestamp").dt.date().n_unique(),
            first_seen=pl.col("timestamp").min(),
            last_seen=pl.col("timestamp").max(),
            n_zones=pl.col("zone").n_unique(),
            level_min=pl.col("level").min(),
            level_max=pl.col("level").max(),
            race=pl.col("race").first(),
            charclass=pl.col("charclass").first(),
            has_guild=pl.col("guild").is_not_null().any(),
        )
        .with_columns(
            span_days=(pl.col("last_seen") - pl.col("first_seen")).dt.total_days(),
            events_per_active_day=pl.col("n_events") / pl.col("active_days"),
        )
        .collect(engine="streaming")
    )

    print(f"characters: {per_char.height:,}")
    print(
        quantile_table(
            per_char,
            [
                "n_events",
                "active_days",
                "span_days",
                "n_zones",
                "level_max",
                "events_per_active_day",
            ],
        )
    )
    print(
        "\ncharacters observed only once:",
        per_char.filter(pl.col("n_events") == 1).height,
    )
    print(
        "characters with a single active day:",
        per_char.filter(pl.col("active_days") == 1).height,
    )
    print("\nlongest-lived characters:")
    print(
        per_char.sort("span_days", descending=True)
        .select("char", "race", "charclass", "level_max", "active_days", "span_days")
        .head(10)
    )


# %%
def gap_analysis(lf):
    """
    Distribution of the time between two consecutive heartbeats of a character.

    The dataset's heartbeat files are produced every ~10 minutes, so most gaps
    sit around 600s (two files are not exactly 600s apart). Gaps of 0 come from
    a character appearing twice in the same snapshot. The head of the
    distribution (gaps of a few seconds) is what makes a naive "one heartbeat =
    one observation" sessionization brittle.
    """
    heading("4. Heartbeat gaps")
    gaps = (
        lf.select("char", "timestamp")
        .sort("char", "timestamp")
        .with_columns(gap_s=pl.col("timestamp").diff().over("char").dt.total_seconds())
        .select(pl.col("gap_s").drop_nulls())
    )

    summary = gaps.select(
        pl.col("gap_s").quantile(q).alias(f"p{int(q * 100):02d}")
        for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
    ).collect()
    print("gap between consecutive heartbeats (seconds):")
    print(summary.transpose(include_header=True, column_names=["value"]))

    thresholds = gaps.select(
        (pl.col("gap_s") <= minutes * 60).mean().alias(f"<= {minutes} min")
        for minutes in GAP_THRESHOLDS_MIN
    ).collect()
    print("\nshare of gaps below a threshold:")
    print(thresholds.transpose(include_header=True, column_names=["share"]))

    poll = gaps.select(
        (pl.col("gap_s") == 600).mean().alias("exactly 600s"),
        (pl.col("gap_s") == 0).mean().alias("exactly 0s"),
    ).collect()
    print("\nheartbeat regularity:")
    print(poll.transpose(include_header=True, column_names=["share"]))

    # Buckets chosen to separate the phenomena we care about: duplicate rows,
    # a character that appears a few times, the normal ~10-minute heartbeat
    # interval, and absences of days or weeks.
    bins = [
        (-1, 0, "0s (duplicate)"),
        (0, 5, "0-5s"),
        (5, 60, "5-60s"),
        (60, 300, "1-5m"),
        (300, 600, "5-10m"),
        (600, 660, "10-11m"),
        (660, 1200, "11-20m"),
        (1200, 3600, "20-60m"),
        (3600, 6 * 3600, "1-6h"),
        (6 * 3600, 86400, "6-24h"),
        (86400, 7 * 86400, "1-7d"),
        (7 * 86400, 30 * 86400, "7-30d"),
        (30 * 86400, None, ">30d"),
    ]
    gap_hist = (
        gaps.with_columns(bin=bucket_expr("gap_s", bins))
        .group_by("bin")
        .len()
        .with_columns(
            order=pl.col("bin").replace_strict(
                {label: i for i, label in enumerate(bucket_labels(bins))},
                return_dtype=pl.Int32,
            )
        )
        .sort("order")
        .drop("order")
        .collect(engine="streaming")
    )
    print(f"\ngap histogram:\n{gap_hist}")
    return gap_hist


# %%
def session_analysis(lf, gap_minutes):
    """
    Cut characters' heartbeats into sessions and summarize them.

    A session starts when the gap with the previous heartbeat exceeds
    `gap_minutes` (or on the character's first heartbeat). Because we only see
    the character at heartbeat time, the measured duration is a lower bound:
    the real play session can start up to one heartbeat interval earlier and end
    one later.
    """
    heading(f"5. Sessions (gap > {gap_minutes} min starts a new session)")
    threshold = gap_minutes * 60

    beats = (
        lf.select("char", "timestamp", "zone")
        .sort("char", "timestamp")
        .with_columns(gap_s=pl.col("timestamp").diff().over("char").dt.total_seconds())
        .with_columns(
            is_new_session=(pl.col("gap_s").is_null() | (pl.col("gap_s") > threshold))
        )
        .with_columns(
            session=pl.col("is_new_session").cast(pl.UInt32).cum_sum().over("char")
        )
    )

    sessions = (
        beats.group_by("char", "session")
        .agg(
            start=pl.col("timestamp").min(),
            end=pl.col("timestamp").max(),
            n_beats=pl.len(),
            n_zones=pl.col("zone").n_unique(),
        )
        .with_columns(
            duration_min=(pl.col("end") - pl.col("start")).dt.total_seconds() / 60
        )
        .collect(engine="streaming")
    )

    print(f"sessions: {sessions.height:,}")
    print(
        quantile_table(
            sessions,
            ["duration_min", "n_beats", "n_zones"],
            quantiles=(0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0),
        )
    )

    per_char = sessions.group_by("char").agg(
        n_sessions=pl.len(),
        mean_session_min=pl.col("duration_min").mean(),
    )
    print("\nsessions per character:")
    print(quantile_table(per_char, ["n_sessions", "mean_session_min"]))

    print("\nmost common session start hours:")
    print(
        sessions.group_by(hour=pl.col("start").dt.hour())
        .len()
        .sort("len", descending=True)
        .head(8)
    )


# %%
def monthly_activity(lf):
    """Heartbeats and distinct characters per month, all vs regular."""
    heading("6. Activity per month")
    month = pl.col("timestamp").dt.truncate("1mo")
    monthly = (
        lf.group_by(month=month)
        .agg(
            events=pl.len(),
            characters=pl.col("char").n_unique(),
            zones=pl.col("zone").n_unique(),
            median_level=pl.col("level").median(),
            max_level=pl.col("level").max(),
        )
        .sort("month")
    )
    # Count characters with enough activity to be considered regular players.
    regular = (
        lf.group_by("char", month=month)
        .agg(
            active_days=pl.col("timestamp").dt.date().n_unique(),
            heartbeats=pl.len(),
        )
        .filter(
            (pl.col("active_days") >= REGULAR_MIN_ACTIVE_DAYS)
            & (pl.col("heartbeats") >= REGULAR_MIN_HEARTBEATS)
        )
        .group_by("month")
        .len()
        .rename({"len": "regular_characters"})
    )
    monthly = (
        monthly.join(regular, on="month", how="left")
        .with_columns(pl.col("regular_characters").fill_null(0))
        .sort("month")
        .collect(engine="streaming")
    )
    monthly = monthly.with_columns(
        pct_of_peak=(
            pl.col("characters") / pl.col("characters").max() * 100
        ).round(1)
    )
    print(monthly)
    return monthly


# %%
def skrub_reports(lf, n_chars, gap_minutes, output_dir, seed):
    """
    Build the skrub-facing view: a character sample with a few derived columns.

    `TableReport` gives a browsable HTML/JSON summary of every column, while
    `column_associations` and `TableVectorizer` show, respectively, which
    attributes are related and how skrub would encode the table for a model.
    """
    heading(f"7. skrub reports ({n_chars} sampled characters)")
    sample_chars = (
        lf.select("char")
        .unique()
        .collect()["char"]
        .sort()
        .sample(n=n_chars, seed=seed)
        .to_list()
    )
    sample = (
        lf.filter(pl.col("char").is_in(sample_chars))
        .collect(engine="streaming")
        .sort("char", "timestamp")
        .with_columns(
            gap_min=pl.col("timestamp").diff().over("char").dt.total_seconds() / 60,
            hour=pl.col("timestamp").dt.hour(),
            weekday=pl.col("timestamp").dt.weekday(),
        )
    )
    print(f"sample: {sample.height:,} heartbeats, {sample['char'].n_unique():,} characters")

    report = skrub.TableReport(
        sample,
        title=f"WoWAH sample: {n_chars} characters",
        n_rows=20,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "table_report.html"
    report.write_html(html_path)
    written = [html_path]
    if hasattr(report, "write_json"):  # added after skrub 0.11
        json_path = output_dir / "table_report.json"
        report.write_json(json_path)
        written.append(json_path)
    print(f"\nwrote {' and '.join(str(path) for path in written)}")

    columns = [
        {
            "column": column["name"],
            "dtype": column["dtype"],
            "n_unique": column["n_unique"],
            "null_share": column["null_proportion"],
        }
        for column in report.dict()["columns"]
    ]
    print(f"\ncolumn summary:\n{pl.DataFrame(columns)}")

    # Character-level table with coarse buckets: Cramer's V on raw numeric
    # columns mostly measures how unique their values are, not real
    # associations, so continuous features are binned first.
    top_zones = (
        sample["zone"]
        .value_counts()
        .sort("count", descending=True)
        .head(10)["zone"]
        .to_list()
    )
    char_table = (
        sample.with_columns(
            is_new_session=(pl.col("gap_min").is_null() | (pl.col("gap_min") > gap_minutes))
        )
        .with_columns(
            session=pl.col("is_new_session").cast(pl.UInt32).cum_sum().over("char")
        )
        .group_by("char")
        .agg(
            race=pl.col("race").first(),
            charclass=pl.col("charclass").first(),
            has_guild=pl.col("guild").is_not_null().any(),
            level=pl.col("level").max(),
            active_days=pl.col("timestamp").dt.date().n_unique(),
            n_sessions=pl.col("session").n_unique(),
            main_zone=pl.col("zone").mode().first(),
        )
        .with_columns(
            level_band=pl.format(
                "{}-{}", pl.col("level") // 20 * 20, pl.col("level") // 20 * 20 + 19
            ),
            active_band=bucket_expr(
                "active_days",
                [
                    (0, 1, "1d"),
                    (1, 3, "2-3d"),
                    (3, 7, "4-7d"),
                    (7, 30, "8-30d"),
                    (30, 90, "1-3mo"),
                    (90, 365, "3-12mo"),
                    (365, None, ">1y"),
                ],
            ),
            session_band=bucket_expr(
                "n_sessions",
                [
                    (0, 1, "1"),
                    (1, 5, "2-5"),
                    (5, 20, "6-20"),
                    (20, 100, "21-100"),
                    (100, None, ">100"),
                ],
            ),
            zone_band=pl.when(pl.col("main_zone").is_in(top_zones))
            .then(pl.col("main_zone"))
            .otherwise(pl.lit("other")),
        )
        .select(
            "race",
            "charclass",
            "has_guild",
            "level_band",
            "active_band",
            "session_band",
            "zone_band",
        )
    )

    associations = skrub.column_associations(char_table, compute_pearson=False)
    print("\nstrongest column associations in the character table (Cramer's V):")
    print(associations.head(15))

    vectorizer = skrub.TableVectorizer()
    encoded = vectorizer.fit_transform(char_table)
    print(f"\nTableVectorizer output: {encoded.shape}")
    print(encoded.head(3))
    kinds = pl.DataFrame(
        {
            "column": list(vectorizer.column_to_kind_.keys()),
            "kind": [
                getattr(kind, "__name__", str(kind))
                for kind in vectorizer.column_to_kind_.values()
            ],
        }
    )
    print("\nskrub's inferred column types:")
    print(kinds)


# %%
def import_pyplot(interactive=False):
    """Import matplotlib lazily so the rest of the script runs without it."""
    try:
        import matplotlib

        if not interactive:
            matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        return None, None
    plt.style.use("seaborn-v0_8-talk")
    return plt, mdates


def annotate_expansions(ax, start, end, fontsize=11, color="0.35"):
    """
    Mark expansion releases that fall inside the plotted time range.

    The labels sit in the margin above the axes, so they never cover the data
    or the legend. Callers leave room for them by passing
    `pad=EXPANSION_LABEL_PAD` to `set_title`.
    """
    for name, date, side in EXPANSIONS:
        if not start <= date <= end:
            continue
        ax.axvline(date, color=color, linestyle="--", linewidth=1.2, alpha=0.8)
        ax.annotate(
            f"{name}\n{date:%b %Y}",
            xy=(date, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(-4 if side == "right" else 4, 4),
            textcoords="offset points",
            ha=side,
            va="bottom",
            fontsize=fontsize,
            color=color,
            annotation_clip=False,
        )


def finish_figure(fig, plt, path, dpi, show):
    """Lay out, write and optionally display a figure, then release it."""
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    print(f"wrote {path}")
    if show:
        plt.show()
    plt.close(fig)


def format_month_axis(ax, mdates, interval=4):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=45)


def plot_gap_histogram(plt, gap_hist, output_dir, dpi=150, show=False):
    """How often does a character disappear for a given amount of time?"""
    fig, ax = plt.subplots(figsize=(15, 6.5))
    ax.bar(
        range(gap_hist.height),
        gap_hist["len"].to_numpy(),
        color="steelblue",
    )
    ax.set_xticks(range(gap_hist.height))
    # 13 bucket labels along the axis: the talk preset's 13 pt ticks crowd
    # them, so step down to 11 pt.
    ax.set_xticklabels(
        [str(b) for b in gap_hist["bin"]], rotation=60, ha="right", fontsize=11
    )
    ax.set_yscale("log")
    ax.set_xlabel("gap between heartbeats")
    ax.set_ylabel("heartbeats (log scale)")
    ax.set_title("Heartbeat gaps")
    ax.grid(True, axis="y", alpha=0.3)

    finish_figure(fig, plt, output_dir / "heartbeat_gaps.png", dpi, show)


def plot_characters_by_month(plt, mdates, monthly, output_dir, dpi=150, show=False):
    """Distinct characters seen each month, all vs regular."""
    months = monthly["month"].to_list()
    characters = monthly["characters"].to_list()
    regular = monthly["regular_characters"].to_list()

    fig, ax = plt.subplots(figsize=(15, 6.5))
    ax.plot(
        months,
        characters,
        marker="o",
        ms=5,
        color="tab:blue",
        label="seen at least once",
    )
    ax.fill_between(months, characters, color="tab:blue", alpha=0.12)
    ax.plot(
        months,
        regular,
        marker="o",
        ms=5,
        color="tab:orange",
        label=(
            f"regular (≥{REGULAR_MIN_ACTIVE_DAYS} active days, "
            f"≥{REGULAR_MIN_HEARTBEATS} heartbeats)"
        ),
    )
    annotate_expansions(ax, months[0], months[-1])
    format_month_axis(ax, mdates)
    ax.set_xlabel("month")
    ax.set_ylabel("distinct characters")
    ax.set_title("Characters seen per month", pad=EXPANSION_LABEL_PAD)
    ax.set_ylim(0, max(characters) * 1.18)
    ax.set_xlim(months[1], months[-1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    finish_figure(fig, plt, output_dir / "characters_by_month.png", dpi, show)


def plot_levels_over_time(plt, mdates, monthly, output_dir, dpi=150, show=False):
    """Max and median character level per month, with expansion releases."""
    months = monthly["month"].to_list()
    median_level = monthly["median_level"].to_list()
    max_level = monthly["max_level"].to_list()

    fig, ax = plt.subplots(figsize=(15, 6.5))
    ax.step(
        months,
        max_level,
        where="post",
        color="tab:orange",
        label="max level",
    )
    ax.plot(
        months,
        median_level,
        marker="o",
        ms=5,
        color="tab:blue",
        label="median level",
    )
    annotate_expansions(ax, months[0], months[-1])
    format_month_axis(ax, mdates)
    ax.set_xlabel("month")
    ax.set_ylabel("character level")
    ax.set_title("Max and median character level by month", pad=EXPANSION_LABEL_PAD)
    ax.set_ylim(0, 100)
    ax.set_xlim(months[1], months[-1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    finish_figure(fig, plt, output_dir / "levels_over_time.png", dpi, show)

    for cap, name, date in (
        (70, "The Burning Crusade", datetime(2007, 4, 3)),
        (80, "Wrath of the Lich King", datetime(2008, 11, 18)),
    ):
        first = monthly.filter(pl.col("max_level") >= cap)["month"].min()
        if first is None:
            print(
                f"  {name} (released {date:%Y-%m}): max level {cap} "
                "not observed in this data"
            )
            continue
        print(
            f"  {name} (released {date:%Y-%m}): first max level >= {cap} "
            f"observed in {first:%Y-%m}"
        )


# %%
PLOT_NAMES = ("gaps", "characters", "levels")


def plots(gap_hist, monthly, output_dir, names=PLOT_NAMES, dpi=150, show=False):
    """Save the exploration figures."""
    plt, mdates = import_pyplot(interactive=show)
    if plt is None:
        print("\nmatplotlib is not installed, skipping figures")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if "gaps" in names:
        if gap_hist is None:
            print("no gap histogram available, skipping heartbeat_gaps.png")
        else:
            plot_gap_histogram(plt, gap_hist, output_dir, dpi, show)
    if {"characters", "levels"} & set(names):
        if monthly is None:
            print("no monthly table available, skipping the monthly figures")
        else:
            if "characters" in names:
                plot_characters_by_month(plt, mdates, monthly, output_dir, dpi, show)
            if "levels" in names:
                plot_levels_over_time(plt, mdates, monthly, output_dir, dpi, show)


def cache_dir(output_dir):
    """Caches live next to the figures but are not tracked by git."""
    return output_dir / "cache"


def monthly_cache_path(output_dir, data_path):
    """Cache name includes the data file so different inputs never collide."""
    return cache_dir(output_dir) / f"monthly_activity_{Path(data_path).stem}.parquet"


def gap_hist_cache_path(output_dir, data_path):
    return cache_dir(output_dir) / f"gap_histogram_{Path(data_path).stem}.parquet"


def load_or_compute_monthly(lf, output_dir, data_path, refresh=False):
    """
    Reuse the cached monthly table when it exists.

    Iterating on the figures should not require rescanning the 35M-row parquet,
    so `--only-plots` reads this cache and only falls back to the full scan when
    it is missing.
    """
    path = monthly_cache_path(output_dir, data_path)
    if path.exists() and not refresh:
        print(f"\nusing cached monthly activity from {path}")
        return pl.read_parquet(path)
    monthly = monthly_activity(lf)
    cache_dir(output_dir).mkdir(parents=True, exist_ok=True)
    monthly.write_parquet(path)
    print(f"cached monthly activity to {path}")
    return monthly


def load_or_compute_gap_hist(lf, output_dir, data_path, refresh=False):
    """Same idea as `load_or_compute_monthly`, for the heartbeat gap histogram."""
    path = gap_hist_cache_path(output_dir, data_path)
    if path.exists() and not refresh:
        print(f"\nusing cached gap histogram from {path}")
        return pl.read_parquet(path)
    gap_hist = gap_analysis(lf)
    cache_dir(output_dir).mkdir(parents=True, exist_ok=True)
    gap_hist.write_parquet(path)
    print(f"cached gap histogram to {path}")
    return gap_hist


def cache_gap_hist(gap_hist, output_dir, data_path):
    cache_dir(output_dir).mkdir(parents=True, exist_ok=True)
    path = gap_hist_cache_path(output_dir, data_path)
    gap_hist.write_parquet(path)
    print(f"cached gap histogram to {path}")


# %%
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--data", default=DEFAULT_DATA, help="heartbeat parquet file")
    parser.add_argument(
        "--gap-minutes",
        type=float,
        default=30,
        help="inactivity (in minutes) that starts a new session",
    )
    parser.add_argument(
        "--sample-chars",
        type=int,
        default=300,
        help="number of characters in the skrub report sample",
    )
    parser.add_argument("--output-dir", default="reports/exploration")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--only-plots",
        action="store_true",
        help="skip the analysis and redraw the figures from the cached tables",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="recompute the cached tables instead of reusing them",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=("all", *PLOT_NAMES),
        default=["all"],
        help="which figures to draw (default: all)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="open the figures in a window instead of only writing them",
    )
    parser.add_argument(
        "--dpi", type=int, default=150, help="resolution of saved figures"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    names = PLOT_NAMES if "all" in args.plots else tuple(args.plots)

    if args.only_plots:
        if args.no_plots:
            print("--only-plots and --no-plots are mutually exclusive, nothing to do")
            return
        lf = pl.scan_parquet(args.data)
        gap_hist = (
            load_or_compute_gap_hist(lf, output_dir, args.data, args.refresh_cache)
            if "gaps" in names
            else None
        )
        monthly = (
            load_or_compute_monthly(lf, output_dir, args.data, args.refresh_cache)
            if {"characters", "levels"} & set(names)
            else None
        )
        plots(gap_hist, monthly, output_dir, names, args.dpi, args.show)
        return

    lf = pl.scan_parquet(args.data)

    print(f"Exploring {args.data}")
    overview(lf)
    categoricals(lf)
    per_character(lf)
    gap_hist = gap_analysis(lf)
    cache_gap_hist(gap_hist, output_dir, args.data)
    session_analysis(lf, args.gap_minutes)
    monthly = load_or_compute_monthly(lf, output_dir, args.data, args.refresh_cache)
    skrub_reports(lf, args.sample_chars, args.gap_minutes, output_dir, args.seed)
    if not args.no_plots:
        plots(gap_hist, monthly, output_dir, names, args.dpi, args.show)


if __name__ == "__main__":
    main()
