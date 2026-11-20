"""
Draw one character's night of heartbeats and the sessions that
skrub.SessionEncoder finds at different session_gap values.

The night is chosen because it contains both a short gap (~10 min, one missed
poll) and a long one (~50 min), so it shows why the threshold is a real choice.
Writes images/session_gap_example.png.
"""
# %%
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from skrub import SessionEncoder

plt.style.use("seaborn-v0_8-talk")

DATA_PATH = "data/wowah_data_2008.parquet"
OUTPUT_PATH = Path("images/session_gap_example.png")

CHAR = 66560
DAY = date(2008, 3, 14)
GAPS = (15, 30, 60)
COLORS = ("#2a52a8", "#c04a10", "#1a7a5a")


def load_night(data_path=DATA_PATH, char=CHAR, day=DAY):
    df = pl.read_parquet(data_path)
    return df.filter(
        (pl.col("char") == char) & (pl.col("timestamp").dt.date() == pl.lit(day))
    ).sort("timestamp")


def session_ids(night, gap_minutes):
    encoder = SessionEncoder(
        split_by="char", timestamp_col="timestamp", session_gap=gap_minutes * 60
    )
    return encoder.fit_transform(night)["timestamp_session_id"]


def plot(night, output_path=OUTPUT_PATH, gaps=GAPS):
    hours = (night["timestamp"].dt.hour() + night["timestamp"].dt.minute() / 60).to_list()

    fig, axes = plt.subplots(
        len(gaps), 1, figsize=(13, 1.5 * len(gaps) + 1.6), sharex=True
    )
    for ax, gap, color in zip(axes, gaps, COLORS):
        ids = session_ids(night, gap)
        # Draw one contiguous bar per session, from its first to its last beat
        for session in ids.unique(maintain_order=True).to_list():
            mask = [i == session for i in ids]
            starts = [h for h, m in zip(hours, mask) if m]
            ax.plot(
                [min(starts), max(starts)],
                [0, 0],
                lw=12,
                color=color,
                alpha=0.35,
                solid_capstyle="round",
            )
        ax.scatter(hours, [0] * len(hours), s=60, color=color, zorder=3)
        ax.set_yticks([])
        ax.set_ylabel(f"gap = {gap} min", rotation=0, ha="right", va="center")
        ax.set_title(
            f"{ids.n_unique()} session{'s' if ids.n_unique() > 1 else ''}",
            loc="right",
        )
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

    axes[-1].set_xlabel(f"time of day — character {CHAR}, {DAY.isoformat()}")
    axes[-1].set_xticks([21, 21.5, 22, 22.5, 23, 23.5])
    axes[-1].set_xticklabels(["21:00", "", "22:00", "", "23:00", ""])
    axes[-1].set_xlim(min(hours) - 0.15, max(hours) + 0.15)
    fig.suptitle("Same heartbeats, different session counts", y=0.995)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"wrote {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot(load_night())
