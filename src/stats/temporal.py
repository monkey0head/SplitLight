from typing import Optional

import pandas as pd

from ..preprocess.filters import filter_by_date


def deltas_between_subsets(
    input_data: pd.DataFrame, target_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculates the time difference between the last timestamp in the input_data and
    the first timestamp in the target_data for each user.

    Args:
        input_data (DataFrame): Input interaction data with 'user_id' and 'timestamp' columns.
        target_data (DataFrame): Target interaction data with 'user_id' and 'timestamp' columns.

    Returns:
        DataFrame: A DataFrame with columns ['user_id', 'timestamp', 'delta'], where
                   'delta' is the time difference between the first target interaction
                   and the last input interaction for each user.
    """
    first_target = (
        target_data.sort_values("timestamp", kind="stable")
        .groupby("user_id")["timestamp"]
        .apply(lambda x: x.iloc[0])
        .to_frame()
    )
    last_input = (
        input_data.sort_values("timestamp", kind="stable")
        .groupby("user_id")["timestamp"]
        .apply(lambda x: x.iloc[-1])
    )

    first_target = first_target.reset_index()
    first_target["delta"] = first_target["timestamp"] - first_target["user_id"].map(
        last_input
    )

    return first_target


def inters_per_period(
    data: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    granularity: str = "D",
) -> pd.DataFrame:
    """
    Counts number of interactions per time interval (e.g., day, hour, etc.).

    Args:
        data (DataFrame): DataFrame containing 'timestamp' and 'item_id'.
        start_date (str, optional): Start date for filtering (inclusive) in DD/MM/YYYY format.
        end_date (str, optional): End date for filtering (inclusive) in DD/MM/YYYY format.
        granularity (str): Time-based resampling granularity (e.g., 'D', 'W') from pandas.

    Returns:
        DataFrame: A DataFrame with time periods and corresponding interaction counts.
    """
    filtered = filter_by_date(data, start_date, end_date)
    result = (
        filtered.resample(granularity)["item_id"].count().reset_index(name="n_inters")
    )

    return result


def time_counts(
    data: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    granularity: str = "h",
    normalize: bool = False,
) -> pd.DataFrame:
    """
    Calculates the number of interactions occurred in a specified time unit (hour, day of week, etc.).

    Args:
        data (DataFrame): DataFrame containing 'timestamp' and 'item_id'.
        start_date (str, optional): Start date for filtering (inclusive) in DD/MM/YYYY format.
        end_date (str, optional): End date for filtering (inclusive) in DD/MM/YYYY format.
        granularity (str): Time unit for aggregation ('h', 'd', 'm', 'y').
        normalize (bool): Whether to normalize the interaction counts.

    Returns:
        DataFrame: A DataFrame with interaction counts per time unit.
    """

    data = data.copy()
    data = inters_per_period(data, start_date, end_date, granularity=granularity)

    time_unit_map = {"h": "hour", "d": "day_of_week", "m": "month", "y": "year"}
    time_unit = time_unit_map[granularity.lower()]

    # Extract time component from timestamps
    if granularity.lower() == "h":
        data[time_unit] = data["timestamp"].dt.hour
    elif granularity.lower() == "d":
        data[time_unit] = data["timestamp"].dt.day_of_week
    elif granularity.lower() == "m":
        data[time_unit] = data["timestamp"].dt.month
    elif granularity.lower() == "y":
        data[time_unit] = data["timestamp"].dt.year

    # Group by time unit and sum interactions
    result = data.groupby(time_unit)["n_inters"].sum().reset_index()

    if normalize:
        result["n_inters"] /= result["n_inters"].sum()

    return result
