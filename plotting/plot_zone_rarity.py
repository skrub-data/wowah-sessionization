"""
Two views of the location features, on March 2008.

Left: every zone placed by how many characters visited it and by its rarity
log(N / n_v), with the 10th-percentile threshold that turns the lowest-rarity
zones into "hubs" drawn as a horizontal line.

Right: the distribution of the per-character Gini coefficient over time spent
in non-hub zones, which is what `get_location_gini` computes.

Writes images/zone_rarity.png.
"""
# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from skrub import SessionEncoder

from adding_features import get_location_gini, get_zone_rarity
from src.utils import get_session_duration

plt.style.use("seaborn-v0_8-talk")

DATA_PATH = "data/wowah_data_2008.parquet"
OUTPUT_PATH = Path("images/zone_rarity.png")
MONTH = 3
GAP_MINUTES = 30
HUB_QUANTILE = 0.1

# Zones worth naming on the scatter, with the offset each label needs so that
# the cluster of busy zones stays readable.
LABELLED = {
    "Orgrimmar": (-4, -14),
    "Shattrath City": (10, -3),
    "The Barrens": (14, 22),
    "Durotar": (-64, -10),
    "Wailing Caverns": (8, 2),
    "Warsong Gulch": (-78, 14),
    "GM Island": (6, 4),
    "Ironforge": (6, 4),
}


# %%
def load_zone_sessions(data_path=DATA_PATH, month=MONTH, gap_minutes=GAP_MINUTES):
    df = pl.read_parquet(data_path)
    month_df = df.filter(pl.col("timestamp").dt.month() == month)
    encoder = SessionEncoder(
        split_by=["char", "zone"],
        timestamp_col="timestamp",
        session_gap=gap_minutes * 60,
        suffix="session_id_zone",
    )
    zone_sessions = encoder.fit_transform(month_df)
    return get_session_duration(zone_sessions, session_column="timestamp_session_id_zone")


def gini_by_char(zone_sessions, zones):
    joined = zone_sessions.join(zones, on="zone", how="left")
    return get_location_gini(joined.lazy(), with_hub=False).collect()


# %%
def plot_rarity(ax, zones):
    threshold = zones["rarity"].quantile(HUB_QUANTILE)
    hubs = zones.filter(pl.col("is_hub"))
    rest = zones.filter(~pl.col("is_hub"))

    for frame, color, label in (
        (rest, "#8a8a8a", f"regular zones ({rest.height})"),
        (hubs, "#c04a10", f"hubs ({hubs.height})"),
    ):
        ax.scatter(
            frame["unique_visitors"],
            frame["rarity"],
            s=34,
            color=color,
            alpha=0.8,
            label=label,
            zorder=3,
        )

    ax.axhline(threshold, color="#2a52a8", lw=1.2, ls="--", zorder=2)
    ax.annotate(
        f"10th percentile of rarity = {threshold:.2f}\nbelow it, a zone is a hub",
        xy=(zones["unique_visitors"].min(), threshold),
        xytext=(4, 6),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=11,
        color="#2a52a8",
    )

    for row in zones.filter(pl.col("zone").is_in(LABELLED)).iter_rows(named=True):
        ax.annotate(
            row["zone"],
            (row["unique_visitors"], row["rarity"]),
            xytext=LABELLED[row["zone"]],
            textcoords="offset points",
            fontsize=10,
            color="#333333",
        )

    ax.set_xscale("log")
    ax.margins(y=0.14)
    ax.set_xlabel("characters that visited the zone (log scale)")
    ax.set_ylabel(r"rarity  $\log(N / n_\mathrm{v})$")
    ax.set_title("Zone rarity in March 2008")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)


def plot_gini(ax, gini):
    values = gini["gini"].to_numpy()
    ax.hist(values, bins=np.linspace(0, 1, 41), color="#1a7a5a", alpha=0.75)

    median = float(np.median(values))
    zero = float((values == 0).mean())
    ax.axvline(median, color="#333333", lw=1.2, ls="--")
    ax.annotate(
        f"median {median:.2f}",
        xy=(median, ax.get_ylim()[1] * 0.92),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=11,
        color="#333333",
    )
    ax.annotate(
        f"{zero:.0%} of characters\nvisit a single\nnon-hub zone",
        xy=(0.06, ax.get_ylim()[1] * 0.62),
        fontsize=11,
        color="#8a4a10",
    )

    ax.set_xlabel("Gini coefficient over time spent per non-hub zone")
    ax.set_ylabel("characters")
    ax.set_title("Was the month concentrated or spread out?")
    ax.spines[["top", "right"]].set_visible(False)


def plot(zone_sessions, zones, gini, output_path=OUTPUT_PATH):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4))
    plot_rarity(axes[0], zones)
    plot_gini(axes[1], gini)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig


# %%
if __name__ == "__main__":
    zone_sessions = load_zone_sessions()
    zones = get_zone_rarity(zone_sessions)
    gini = gini_by_char(zone_sessions, zones)
    plot(zone_sessions, zones, gini)
