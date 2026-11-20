"""
Why `load` samples characters and not rows.

A 10% sample of the *rows* of the 2008 heartbeat table, for 40 characters drawn
at random from the 20,472 the sample kept. Grey cells are months in which the
character genuinely did not play; orange cells are months the character did play
but the row sample dropped.

Writes images/sampling.png.
"""
# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import ListedColormap

DATA_PATH = "data/wowah_data_2008.parquet"
OUTPUT_PATH = Path("images/sampling.png")
FRACTION = 0.1
SEED = 42
N_CHARS = 40

PRESENT = "#2a52a8"
SPURIOUS = "#c04a10"
ABSENT = "#e6e6e6"


# %%
def load_beats(data_path=DATA_PATH):
    df = pl.read_parquet(data_path, columns=["char", "timestamp"])
    return df.with_columns(pl.col("timestamp").dt.truncate("1mo").alias("month"))


def cells(df):
    return df.select(["char", "month"]).unique()


def sample_rows(df, fraction=FRACTION, seed=SEED):
    return df.sample(fraction=fraction, seed=seed)


# %%
def grid(sample_cells, truth_cells, chars, months):
    """0 = present, 1 = missing but truly present, 2 = truly missing."""
    present = set(zip(sample_cells["char"].to_list(), sample_cells["month"].to_list()))
    truth = set(zip(truth_cells["char"].to_list(), truth_cells["month"].to_list()))
    out = np.zeros((len(chars), len(months)), dtype=int)
    for i, char in enumerate(chars):
        for j, month in enumerate(months):
            if (char, month) in present:
                out[i, j] = 0
            elif (char, month) in truth:
                out[i, j] = 1
            else:
                out[i, j] = 2
    return out


def draw(ax, mask, months, n_kept):
    ax.imshow(
        mask,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap([PRESENT, SPURIOUS, ABSENT]),
        vmin=0,
        vmax=2,
    )
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([m.strftime("%b") for m in months], fontsize=9)
    ax.set_yticks([])
    ax.set_title(
        f"{mask.shape[0]} characters, a snippet of the {n_kept:,} the sample kept",
        fontsize=10,
        loc="left",
    )
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


# %%
def plot(beats, output_path=OUTPUT_PATH):
    truth = cells(beats)
    row_sample = sample_rows(beats)
    row_cells = cells(row_sample)

    # A snippet of the characters the row sample kept, so the panel stays
    # readable: the sample kept far more characters than can be drawn.
    kept = row_cells["char"].unique().to_list()
    shared = pl.Series(sorted(kept)).sample(n=N_CHARS, seed=SEED).to_list()
    months = sorted(truth["month"].unique().to_list())

    mask = grid(row_cells, truth, shared, months)

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    draw(ax, mask, months, len(kept))

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PRESENT),
        plt.Rectangle((0, 0), 1, 1, color=SPURIOUS),
        plt.Rectangle((0, 0), 1, 1, color=ABSENT),
    ]
    fig.legend(
        handles,
        ["played, kept by the sample", "played, dropped by the sample", "did not play"],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"wrote {output_path}")
    return fig


# %%
if __name__ == "__main__":
    plot(load_beats())
