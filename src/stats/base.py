from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .utils import get_deltas

def get_ts_collisions(data: pd.DataFrame, user_id: str = "user_id", timestamp: str = "timestamp") -> Optional[pd.DataFrame]:
    """
    Adds a flag column indicating duplicate timestamp interactions.
    
    Args:
        data: DataFrame containing user interactions with columns: user_id, timestamp
        user_id (str, optional): Name of the user ID column.
        timestamp (str, optional): Name of the timestamp column.
        
    Returns:
        DataFrame with 'timestamp_collisions' column added
    """
    data = data.copy()
    data['timestamp_collisions'] = data.duplicated(subset=[user_id, timestamp], keep='first')
    
    return data

def calc_lifetime(
    data: pd.DataFrame, timestamp: str = "timestamp", col: str = "user_id"
) -> pd.DataFrame:
    """
    Calculate the lifetime (in days) of each entity (e.g., user or session) in a DataFrame,
    based on the minimum and maximum timestamps.

    Args:
        data (pd.DataFrame): Interaction DataFrame.
        timestamp (str, optional): Name of the timestamp column.
        col (str, optional): Name of the column used to group entities (e.g., user ID). Defaults to "user_id".

    Returns:
        pd.DataFrame: A DataFrame with the entity column, minimum timestamp, maximum timestamp,
                      and calculated lifetime in days.
    """
    duration = data.groupby(col)[timestamp].agg(min_ts="min", max_ts="max")
    duration["lifetime"] = duration["max_ts"] - duration["min_ts"]
    return duration.reset_index()


def get_mean_median(series: pd.Series, prefix: str = "") -> Dict[str, float]:
    """
    Compute mean and median of a series.

    Args:
        series (pd.Series): Series to compute mean and median of.
        prefix (str, optional): Prefix to prepend to each statistic name.

    Returns:
        Dictionary of mean and median.
    """
    return {
        f"mean_{prefix}": series.mean(),
        f"median_{prefix}": series.median(),
    }


def count_delta_stats(data: pd.DataFrame, col="user_id", timestamp="timestamp") -> Dict[str, float]:
    """
    Compute time between interactions statistics (mean and median) from timestamped data.

    The delta is calculated as the time difference (in seconds) between
    each interaction and the previous interaction of a user.

    Args:
        data (pd.DataFrame): DataFrame containing timestamped records.

    Returns:
        Dictionary with mean and median time between interactions.
    """
    deltas = get_deltas(data, col=col, timestamp=timestamp)
    return get_mean_median(deltas["delta"], prefix="time_between_interactions")


def get_time_period_days(max_timestamp: float, min_timestamp: float) -> float:
    """
    Convert time difference in seconds to days.

    Args:
        max_timestamp (float): Latest timestamp in seconds.
        min_timestamp (float): Earliest timestamp in seconds.

    Returns:
        Duration in days.
    """
    return (max_timestamp - min_timestamp) / (60 * 60 * 24)


def gini(array: np.ndarray) -> float:
    """
    Calculate the Gini coefficient of a numpy array.
    Source: https://github.com/oliviaguest/gini

    Args:
        array (np.array): 1D numpy array of non-negative values.

    Returns:
        Gini coefficient (float).
    """
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array = array + 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1, array.shape[0] + 1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))


def calculate_sequence_stats(lengths: pd.Series, prefix: str = "") -> Dict[str, float]:
    """
    Compute descriptive statistics of a sequence of lengths.

    Args:
        lengths (pd.Series): Pandas Series of sequence lengths.
        prefix (str, optional): Prefix to prepend to each statistic name.

    Returns:
        Dictionary of statistical measures with prefixed keys.
    """

    desc = lengths.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    stats = {
        f"{prefix}count": desc["count"],
        f"{prefix}mean": desc["mean"],
        f"{prefix}std": lengths.std(),
        f"{prefix}min": desc["min"],
        f"{prefix}q01": desc["1%"],
        f"{prefix}q05": desc["5%"],
        f"{prefix}q25": desc["25%"],
        f"{prefix}median": desc["50%"],
        f"{prefix}q75": desc["75%"],
        f"{prefix}q95": desc["95%"],
        f"{prefix}q99": desc["99%"],
        f"{prefix}max": desc["max"],
    }

    return stats


def additional_stats(data: pd.DataFrame,
    user_id: str = "user_id",
    item_id: str = "item_id") -> Dict[str, Union[int, float]]:
    """
    Compute additional statistics for a dataset.

    Args:
        data (pd.DataFrame): Interaction DataFrame.

    Returns:
        Dictionary of additional statistics.
    """
    stats = {}        
    n_users = data[user_id].nunique()
    n_items = data[item_id].nunique()
    stats["space_size"] = n_users * n_items / 1000
    stats["space_size_log"] = np.log10(n_users * n_items / 1000)
    stats["shape"] = n_users / n_items
    stats["shape_log"] = np.log10(stats["shape"])

    # Inequality in user/item distributions
    stats["Gini_users"] = gini(data[user_id].values)
    stats["Gini_items"] = gini(data[item_id].values)
    return stats


def temporal_stats(data: pd.DataFrame,
    user_id: str = "user_id",
    item_id: str = "item_id",
    timestamp: str = "timestamp") -> Dict[str, Union[int, float]]:
    """
    Compute temporal statistics for a dataset.

    Args:
        data (pd.DataFrame): Interaction DataFrame.
        timestamp (str, optional): Name of the timestamp column.
        user_id (str, optional): Name of the user ID column.
        item_id (str, optional): Name of the item ID column.

    Returns:
        Dictionary of temporal statistics.
    """
    stats = {}
    # Temporal statistics
    stats.update(
        {
            "max_timestamp": data[timestamp].max(),
            "min_timestamp": data[timestamp].min(),
        }
    )
    stats["timeframe"] = data[timestamp].max() - data[timestamp].min()
    stats.update(count_delta_stats(data, col=user_id, timestamp=timestamp))
    # User lifetime stats (in days)
    user_lifetimes = calc_lifetime(data, timestamp, user_id)
    stats.update(get_mean_median(user_lifetimes["lifetime"], prefix="user_lifetime"))
    stats["mean_user_lifetime, %"] = (
        stats["mean_user_lifetime"] * 100 / stats["timeframe"]
    )
    # Item lifetime stats (in days)
    item_lifetimes = calc_lifetime(data, timestamp, item_id)
    stats.update(get_mean_median(item_lifetimes["lifetime"], prefix="item_lifetime"))
 
    stats["mean_item_lifetime, %"] = (
        stats["mean_item_lifetime"] * 100 / stats["timeframe"]
    )
    
    ts_collisions = get_ts_collisions(data, user_id, timestamp)['timestamp_collisions']
    stats["timestamp_collisions"] = ts_collisions.sum()
    stats["timestamp_collisions, %"] = ts_collisions.mean() * 100
    
    return stats

def base_stats(
    data: pd.DataFrame,
    extended: bool = False,
    user_id: str = "user_id",
    item_id: str = "item_id",
    timestamp: str = "timestamp",
    return_df=True,
) -> Dict[str, Union[int, float]]:
    """
    Compute dataset-level statistics for user-item interaction data.

    Args:
        data (pd.DataFrame): Interaction DataFrame.
        extended (bool): Whether to compute advanced statistics (default: False).
        user_id (str, optional): Name of the user ID column.
        item_id (str, optional): Name of the item ID column.
        timestamp (str, optional): Name of the timestamp column.

    Returns:
        Dictionary of dataset-level statistics.
    """

    n_users = data[user_id].nunique()
    n_items = data[item_id].nunique()
    n_interactions = len(data)
    seq_lengths = data.groupby(user_id).size()

    duration_days = data[timestamp].max() - data[timestamp].min()

    stats = {
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_interactions,
        "avg_seq_length": seq_lengths.mean(),
        "density": n_interactions / (n_users * n_items),
        "timeframe": duration_days,
    }
    if extended:
        # stats.update(additional_stats(data, user_id, item_id))
        stats.update(temporal_stats(data, user_id, item_id, timestamp))
        # Item occurrence and user activity distributions
        item_counts = data[item_id].value_counts()
        stats.update(get_mean_median(item_counts, prefix="item_occurrence"))

        user_counts = data[user_id].value_counts()
        stats.update(get_mean_median(user_counts, prefix="user_activity"))

    stats = pd.DataFrame([stats]) if return_df else pd.Series(stats)

    return stats


def group_subsets(
    subsets: Union[List[pd.DataFrame], Dict[str, pd.DataFrame]],
    subset_names: Optional[List[str]] = None,
    extended: bool = False,
) -> pd.DataFrame:
    """
    Compute statistics for multiple dataset subsets.

    Args:
        subsets (Union[List[pd.DataFrame], Dict[str, pd.DataFrame]]): List or dictionary of DataFrames.
        subset_names (List[str], optional): Optional list of names for subsets if input is a list.
        extended (bool): Whether to compute extended statistics.

    Returns:
        DataFrame with subset statistics.
    """
    if not isinstance(subsets, (list, dict)):
        raise TypeError("Input must be a list or dict of dataframes")

    # Convert list of DataFrames into a dictionary
    if isinstance(subsets, list):
        if subset_names is not None:
            # Add extra default names if not enough names are provided
            keys = [
                subset_names[i] if i < len(subset_names) else f"Subset {i + 1}"
                for i in range(len(subsets))
            ]
        else:
            # Use default names if no names provided
            keys = [f"Subset {i + 1}" for i in range(len(subsets))]

        subsets = {key: df for key, df in zip(keys, subsets)}

    # Compute base statistics for each subset and return as DataFrame
    result = {
        key: base_stats(df, extended, return_df=False) for key, df in subsets.items()
    }

    return pd.DataFrame(result).T


def compare_subsets(
    subsets: Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame]],
    reference_data: Optional[pd.DataFrame] = None,
    subset_names: Optional[List[str]] = None,
    extended: bool = False,
    return_ref_stats: bool = False,
) -> Union[pd.DataFrame, tuple[pd.Series, pd.DataFrame]]:
    """
    Compare statistics of multiple dataset subsets, optionally against a reference dataset.

    Args:
        subsets (Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame]]): A single DataFrame, list of DataFrames, or dictionary of DataFrames.
        reference_data (pd.DataFrame, optional): Optional reference dataset to compare all subsets against.
        subset_names (List[str], optional): Optional list of names for subsets (used if subsets is a list).
        extended (bool, optional): Whether to compute extended statistics (True = more detailed).
        return_ref_stats (bool, optional): If True, also return the reference statistics alongside comparison.

    Returns:
        A DataFrame comparing the statistics (absolute + percentage diff),
        or a tuple (reference_stats, comparison_df) if return_ref_stats is True.
    """
    # If only one dataset is passed, wrap it in a list
    if isinstance(subsets, pd.DataFrame):
        subsets = [subsets]

    subset_stats = group_subsets(subsets, subset_names, extended)

    if reference_data is not None:
        # Compute reference statistics
        reference_stats = base_stats(reference_data, extended, return_df=False)

        # Compute % share of each subset relative to the reference
        pct_share = (subset_stats / reference_stats * 100).round(2)

        combined_df = pd.concat(
            [subset_stats, pct_share], axis=1, keys=["Abs. value", "%"]
        )

        combined_df = combined_df.reorder_levels([1, 0], axis=1)

        # Reorder to match reference stats order
        result = combined_df[reference_stats.index]
    else:
        result = subset_stats

    return (reference_stats.to_frame().T, result) if return_ref_stats else result
