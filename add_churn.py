"""
This script builds the (char, month) table with the prediction target.

The idea is that, given a month M, we use all the data up to the last day of the
previous month (M - 1) to predict whether a character will be absent in both
M and M + 1 ("churn_next").

So, if we are in the month of April, we use all the data up to March 31st to
predict whether a character will be absent in both April and May. We look at
May because we might want to take action in April to prevent the character from
churning.

To do this, we need one row per character per month, and a "has_played" column
that is True if the character played in that month, and False otherwise:

- we create all the months in the range we have data for, and take the cross
  product with the unique characters;
- we find all the unique (char, month) combinations in the original dataset,
  and left join them to the cross product. Null values in "has_played" are
  filled with False;
- months prior to the first month a character appears in are filtered out.

From "has_played" we derive:

- "prev_played": "has_played" in the previous month;
- "churn": the character is absent this month AND the previous month;
- "churn_next": "churn" in the following month, i.e. the target.

Since the features of month M are built from M - 1, a character that was already
absent in M - 1 has no features. `build_churn_dataset(only_prev_active=True)`
keeps only the rows where the character played in M - 1.
"""

# %%
import polars as pl
# %%


def make_data(df, interval="1mo"):
    """
    Prepare a dataframe that contains the months in which each player has
    actually played. The column "has_played" is then set to True for those months.
    Any month in which the player did not play is missing from this dataset,
    because the dataset is built starting from player activity, so any "empty
    month" simply doesn't exist in this dataset.
    """

    data = (
        df.with_columns(
            pl.col("timestamp").dt.truncate(interval).alias("month"),
            first_month=pl.col("timestamp").dt.truncate(interval).min().over("char"),
        )
        .unique(subset=["char", "month"])
        .with_columns(pl.lit(True).alias("has_played"))
    )
    return data


def make_user_month(df, interval="1mo"):
    """
    Create a DataFrame with all unique combinations of users and months.
    This will be used to ensure that we have a row for each user for each month
    in the range of the dataset, even if the user did not play in that month.

    This is done with a cross-product between the unique users and the range of months.

    This function is artificially generating 3 months past the last month in the
    dataset. Their "churn_next" is null, so `build_churn_dataset` drops them: they
    are only visible when calling `add_churn` directly.

    Note that since the months are generated from the range of the dataset, they
    may include months prior to a user's first activity. These will need to be
    filtered out later.
    """

    months = pl.datetime_range(
        start=df.select(pl.col("timestamp").dt.truncate(interval).min())
        .collect()
        .item(),
        end=df.select(
            # Adding 3 months to the end of the range because we are adding
            # lagged features up to 3 months.
            pl.col("timestamp").dt.truncate(interval).max().dt.offset_by("3mo")
        )
        .collect()
        .item(),
        interval=interval,
        closed="both",
        eager=True,
    )

    char_month = (
        df.with_columns(month=pl.col("timestamp").dt.truncate(interval))
        .select("char")
        .unique()
        .join(months.to_frame(name="month").lazy(), how="cross")
    )
    return char_month


def add_churn(user_month, data):
    """
    Add churn information to the user-month DataFrame by merging it with the actual
    play data. The resulting DataFrame will have a "has_played" column indicating
    whether the user played in that month. Any missing values in "has_played" are
    filled with False: missing values mean that the player did not play in that
    particular month.

    It also adds the "churn" column (churned as of this month: absent this month
    and the previous one) and the "churn_next" column, the prediction target:
    the value of "churn" in the *following* month. Features for a given month are
    built from the previous month's data, so the target cannot be the churn
    status of the same month: that status includes "absent last month", which
    is exactly what the features reveal. "churn_next" is null for the last
    observed month and for the padding months added by `make_user_month`,
    because the following month has not been observed.
    """

    last_month = data.lazy().select(pl.col("month").max()).collect().item()

    df_with_user_month = (
        user_month.join(
            data,
            on=["char", "month"],
            how="left",
        )
        .with_columns(pl.col("has_played").fill_null(False))
        .with_columns(first_month=pl.col("first_month").min().over("char"))
        .filter(pl.col("first_month") <= pl.col("month"))
        .select(
            "char",
            "month",
            "has_played",
            "first_month",
        )
        .sort(["char", "month"])
        .with_columns(
            # has_played in the previous month for this character
            prev_played=pl.col("has_played").shift(1).over("char").fill_null(False)
        )
        .with_columns(
            # churn = absent this month AND absent last month
            # prev_played is False on a character's first month
            churn=~pl.col("has_played") & ~pl.col("prev_played")
        )
        .with_columns(
            churn_next=pl.when(pl.col("month").dt.offset_by("1mo") <= last_month)
            .then(pl.col("churn").shift(-1).over("char"))
            .otherwise(None)
        )
        .select(
            "char",
            "month",
            "has_played",
            "prev_played",
            "churn",
            "churn_next",
            "first_month",
        )
    )
    return df_with_user_month


def build_churn_dataset(historical_data, only_prev_active=False):
    """
    Build the (char, month) table with `has_played`, `prev_played`, `churn` and
    the target `churn_next`.

    Rows whose target is unknown (the last observed month and the padding months
    after it) cannot be used to train or evaluate a model, so they are dropped.

    With `only_prev_active=True`, only the rows where the character played in the
    previous month are kept. Features for a row at month M are built from month
    M - 1, so a character that was already absent in M - 1 has no features and
    is not a prediction we could act on: the target then only contains
    characters that were active in M - 1 and are absent in both M and M + 1.
    Keep the default (all rows) when the full (char, month) grid is needed, for
    example to build lagged features, which need each character's full timeline.
    """
    # Load the dataset
    data = make_data(historical_data)
    # Add the combinations of user months
    user_month = make_user_month(historical_data)
    churn_data = add_churn(user_month, data).filter(
        pl.col("churn_next").is_not_null()
    )
    if only_prev_active:
        churn_data = churn_data.filter(pl.col("prev_played"))
    return churn_data.collect()


if __name__ == "__main__":
    churn_data = build_churn_dataset(pl.scan_parquet("data/wowah_data_2008.parquet"))
    print(churn_data.head())
    # churn_data.write_parquet("data/wowah_churn_data.parquet")
