# %%
import polars as pl
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-talk")

from add_churn import build_churn_dataset

# The rows the model sees: one per (character, month) in which the character
# played in the previous month, with a known target.
df = build_churn_dataset(
    pl.scan_parquet("../data/wowah_data_2008.parquet"), only_prev_active=True
)
# %%
df.group_by("month").agg(pl.len()).sort("month")
# %%
df_grouped = df.group_by("month").agg(pl.len()).sort("month")

fig, ax = plt.subplots()
ax.barh(df_grouped["month"].cast(str), df_grouped["len"])
plt.show()
# %%
df_grouped = df_grouped.with_columns(cumsum=pl.col("len").cum_sum())
# %%
months = df_grouped["month"].dt.strftime("%Y-%m")
lens = df_grouped["len"].to_list()
lefts = [0] + df_grouped["cumsum"].to_list()#[:-1]  # left offset for each month's segment
n = len(months)

split_labels = [f"Split {i+1}" for i in range(n)]
cmap = plt.cm.plasma
colors = [cmap(j / max(n - 1, 1)) for j in range(n)]

fig, ax = plt.subplots(figsize=(13, 7.5))

for j in range(n):
    ax.barh(
        months[j:],
        [lens[j]] * (n - j),
        left=lefts[j],
        color=colors[j],
        label=split_labels[j],
    )

cumsums = df_grouped["cumsum"].to_list()
for i, total in enumerate(cumsums):
    diff = lens[i] if i == 0 else lefts[i+1] - lefts[i]
    sign = "+" if diff >= 0 else ""
    ax.text(total, months[i], f"  {total:,}\n ({sign}{diff:,})", va="center")

ax.invert_yaxis()  # Split 1 at top for natural reading order
ax.set_xlabel("Number of (avatar, month) rows in the training set")
ax.set_ylabel("CV Split (Month)")
ax.set_title("Number of rows in each CV split")
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles, labels,
    title="CV Split", loc="upper left", bbox_to_anchor=(1.01, 1.0),
    frameon=False,
)

desc = (
    "The training set grows with each CV split, but not by a constant amount:\n"
    f"a month adds between {min(lens):,} and {max(lens):,} rows\n"
    "(avatars active in the previous month)."
)


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# text box at top-centre; one arrow to first (smallest) split, one to last (largest)
ax.annotate(
    desc.strip(),
    xy=(0.1, 0.92),                 # arrow tip: near the short first-split bar
    xytext=(0.58, 0.94),             # text box: top-middle of the axes
    xycoords="axes fraction",
    textcoords="axes fraction",
    ha="center", va="top",
    bbox=dict(boxstyle="round,pad=0.4", fc="lightgrey", ec="grey", alpha=0.8),
    arrowprops=dict(arrowstyle="->", color="grey"),
)
ax.annotate(
    "",
    xy=(0.85, 0.2),                 # arrow tip: toward the long last-split bar
    xytext=(0.62, 0.82),             # bottom edge of the text box
    xycoords="axes fraction",
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", color="grey"),
)

plt.tight_layout()
plt.show()
# %%
fig.savefig("../images/user_months_plot.png")
# %%
