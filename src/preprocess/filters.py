from typing import Optional

import pandas as pd

from ..stats.base import base_stats

"""
Filter interactions.
"""


def min_count_filter(data, min_count, col_name, verbose=False):
    """Filter by occurrence threshold.

    :param data: interactions log
    :param min_count: minimal number of interactions required
    :param col_name: column name, e.g. user or item id
    """
    counts = data[col_name].value_counts()
    data = data[data[col_name].isin(counts[counts >= min_count].index)]

    if verbose:
        print(base_stats(data, extended=False))
    return data


def drop_consecutive_repeats(
    data: pd.DataFrame, user_id="user_id", item_id="item_id", timestamp="timestamp"
):
    """Remove repeated items like i-i-j -> i-j. Keep the first consecutive interaction.

    :param data: interactions log
    :param user_id: user col name, defaults to 'user_id'
    :param item_id: item col name, defaults to 'item_id'
    :param timestamp: timestamp col name, defaults to 'timestamp'
    """

    data_sorted = data.sort_values([user_id, timestamp], kind="stable")
    data_sorted["shifted"] = data_sorted.groupby(user_id)[item_id].shift(periods=1)
    return (
        data_sorted[data_sorted[item_id] != data_sorted["shifted"]]
        .drop("shifted", axis=1)
        .reset_index(drop=True)
    )


def core_filter(
    data,
    item_min_count=5,
    seq_min_len=5,
    drop_consec_repeats=False,
    user_id="user_id",
    item_id="item_id",
    timestamp="timestamp",
    verbose=True,
):
    """N-core filter

    :param data: _description_
    :param item_min_count: _description_, defaults to 5
    :param seq_min_len: _description_, defaults to 5
    :param drop_consec_repeats: if remove consecutive repeated items, defaults to False
    """
    step = 1

    data = data.copy()
    if drop_consec_repeats:
        data = drop_consecutive_repeats(data, user_id, item_id, timestamp)

        if verbose:
            print("After consecutive repeats filtering")
            print(base_stats(data, extended=False))

    while len(data) > 0 and (
        data[user_id].value_counts().min() < seq_min_len
        or data[item_id].value_counts().min() < item_min_count
    ):
        data = min_count_filter(
            data, min_count=seq_min_len, col_name=user_id, verbose=verbose
        )
        data = min_count_filter(
            data, min_count=item_min_count, col_name=item_id, verbose=verbose
        )

        if drop_consec_repeats:
            data = drop_consecutive_repeats(
                data, user_id=user_id, item_id=item_id, timestamp=timestamp
            )
        if verbose:
            print(f"After n-core filtering on step {step}")
            print(base_stats(data, extended=False))
        step += 1
    return data


def filter_by_date(
    data: pd.DataFrame, start_date: Optional[str] = None, end_date: Optional[str] = None
) -> pd.DataFrame:
    """Filter DataFrame rows by date range.

    Converts timestamp column to datetime index and filters between start_date (inclusive)
    and end_date (inclusive). For end_date, time is set to 23:59:59.
    Dates should be in DD/MM/YYYY format. Timestamps should be in Unix seconds.

    Args:
        data: DataFrame containing a 'timestamp' column with Unix timestamps
        start_date: Optional start date in DD/MM/YYYY format. If None, uses earliest date.
        end_date: Optional end date in DD/MM/YYYY format. If None, uses latest date.

    Returns:
        Filtered DataFrame with datetime index

    Example:
    >>> filtered = filter_by_date(df, start_date='01/01/2021', end_date='01/05/2021')
    """
    data = data.copy(deep=True)

    data["timestamp"] = pd.to_datetime(data["timestamp"], dayfirst="True", unit="s")
    data = data.set_index("timestamp")

    start_datetime = (
        pd.to_datetime(start_date, dayfirst="True") if start_date else data.index.min()
    )
    end_datetime = (
        pd.to_datetime(end_date, dayfirst="True") if end_date else data.index.max()
    )

    end_datetime = end_datetime.replace(hour=23, minute=59, second=59)

    filtered_data = data[(data.index >= start_datetime) & (data.index <= end_datetime)]

    return filtered_data
