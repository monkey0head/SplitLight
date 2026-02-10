from typing import Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp


def combine_and_cumcount(
    df_base: pd.DataFrame, df_new: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine two DataFrames, compute cumulative interaction count per user, and separate again.

    Args:
        df_base (pd.DataFrame): The original DataFrame.
        df_new (pd.DataFrame): The new DataFrame to combine.

    Returns:
        Tuple of (updated df_base, updated df_new) with cumulative counts.
    """
    df_combined = pd.concat(
        [df_base.assign(source="base"), df_new.assign(source="new")]
    )

    df_combined = cumcount_by_user(df_combined)

    base_result = df_combined[df_combined["source"] == "base"].drop("source", axis=1)
    new_result = df_combined[df_combined["source"] == "new"].drop("source", axis=1)

    return base_result, new_result


def cumcount_by_user(df):

    df = df.sort_values(["user_id", "timestamp"], kind="stable")
    df["cumcount"] = df.groupby("user_id").cumcount() + 1

    return df


def compare_distributions(
    samples1: Union[pd.Series, np.ndarray], samples2: Union[pd.Series, np.ndarray]
) -> pd.Series:
    """
    Compare two distributions using statistical distance metrics.

    Args:
        samples1: First sample array or Series.
        samples2: Second sample array or Series.

    Returns:
        A Series with test statistics (KS test, Wasserstein, Energy distance).
    """

    stat_dict = {}

    stat_dict["Kolmogorov-Smirnov test"] = stats.kstest(samples1, samples2).statistic
    stat_dict["Wasserstein distance"] = stats.wasserstein_distance(samples1, samples2)
    stat_dict["Energy distance"] = stats.energy_distance(samples1, samples2)

    return pd.Series(stat_dict)


def get_deltas(data: pd.DataFrame, col="user_id", timestamp="timestamp") -> pd.DataFrame:
    """
    Computes the time difference (delta) between successive interactions for each enity (user or item).

    The delta is calculated as the time difference (in seconds) between
    each interaction and the previous interaction for the same entity.

    Args:
        data (pd.DataFrame): DataFrame with 'user_id', 'item_id' and 'timestamp' columns.
        col (str, optional): Name of the column used to group entities (e.g., user ID). Defaults to "user_id".
        timestamp (str, optional): Name of the timestamp column. Defaults to "timestamp".

    Returns:
        DataFrame: The original DataFrame with an added 'delta' column.
    """
    data = data.copy().reset_index(drop=True)

    # Calculate time difference between consecutive interactions per user
    data["delta"] = (
        data.sort_values([col, timestamp], kind="stable")
        .groupby(col)[timestamp]
        .diff()
    )
    return data


def get_conseq_duplicates(data: pd.DataFrame, user_id: str = "user_id", item_id: str = "item_id", timestamp: str = "timestamp") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Identifies consecutive duplicate interactions in the dataset.

    Args:
        data (pd.DataFrame): A DataFrame containing user interactions.

    Returns:
        DataFrame: The original DataFrame with an added 'conseq_duplicate' column marking consecutive duplicates
    """
    data_sorted = data.copy()
    data_sorted.sort_values([user_id, timestamp], kind="stable", inplace=True)
    data_sorted["shifted"] = data_sorted.groupby(user_id)[item_id].shift(periods=1)

    data_sorted["conseq_duplicate"] = (
        data_sorted[item_id] == data_sorted["shifted"]
    ).fillna(False)

    return data_sorted.drop(columns="shifted")

def resample_by_time(data: pd.DataFrame, granularity: str):
    """
    Resamples a pandas DataFrame by time based on specified granularity.

    Converts a timestamp column from seconds to datetime, sets it as index,
    and resamples the data according to the given time granularity.

    Args:
        data (pd.DataFrame): Input DataFrame containing a 'timestamp' column in seconds.
        granularity (str): Time frequency string for resampling (e.g., 's' for seconds,
                    'min' for minutes, 'H' for hours). Defaults to 's' (seconds).

    Returns:
        A pandas Resampler object that can be used with aggregation functions.
    """
    data["timestamp"] = pd.to_datetime(data["timestamp"], unit="s")
    data = data.set_index("timestamp")

    return data.resample(granularity)


def distribution_distances(distr1, distr2, log=True, n_bins=100):
    """
    Compute several distance measures between distributions:
    Kolmogorov-Smirnov test, Jensen-Shannon divergence, Hellinger distance,
    histogram intersection and total variation distance.

    Args:
        distr1: Array with observations of the first variable
        distr2:  Array with observations of the second variable
        log (bool): Whether to log transform before computing distances.
        n_bins (int): Number of bins for computing histograms.

    Returns:
        dict with different distances.
    """

    if log:
        distr1 = np.log(1 + distr1)
        distr2 = np.log(1 + distr2)

    ks_stat, ks_pval = ks_2samp(distr1, distr2)

    min_value = min(distr1.min(), distr2.min())
    max_value = max(distr1.max(), distr2.max())
    bins = np.linspace(min_value, max_value, n_bins)
    bin_width = (max_value - min_value) / n_bins

    p, _ = np.histogram(distr1, bins=bins, density=True)
    q, _ = np.histogram(distr2, bins=bins, density=True)

    jsd = jensenshannon(p, q)

    hist_intersection = np.sum(np.minimum(p, q)) * bin_width
    total_variation = 1/2 * np.sum(np.abs(p-q)) * bin_width

    sqrt_diff = np.sqrt(p) - np.sqrt(q)
    h2 = 0.5 * np.sum(sqrt_diff**2) * bin_width
    hellinger = np.sqrt(max(h2, 0))

    return {'KS': ks_stat, 'jensen-shannon': jsd, 'hellinger': hellinger,
            'hist_intersection': hist_intersection, 'total_variation': total_variation} 
