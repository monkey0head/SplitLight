"""
Preprocessing utils.
"""

import numpy as np

def encode(data, col_name, shift):
    """Encode items/users to consecutive ids.

    :param col_name: column to do label encoding, e.g. 'item_id'
    :param shift: shift encoded values to start from shift
    """
    data[col_name] = data[col_name].astype("category").cat.codes + shift
    return data


def rename_cols(data, user_id="user_id", item_id="item_id", timestamp="timestamp"):
    "Rename columns of dataframe"

    data = data.rename(
        columns={user_id: "user_id", item_id: "item_id", timestamp: "timestamp"}
    )

    return data

def sample_users(data, user_id="user_id", users_sample=None, random_state=42):
    """
    users_sample:
        - int   -> number of users
        - float -> fraction of users (0, 1]
    """
    data = data.copy()
    
    if users_sample is None:
        return data

    users = data[user_id].unique()
    n_users = len(users)

    if isinstance(users_sample, float):
        if not (0 < users_sample <= 1):
            raise ValueError("users_sample as float must be in (0, 1]")
        k = int(n_users * users_sample)

    elif isinstance(users_sample, int):
        k = min(users_sample, n_users)

    else:
        raise TypeError("users_sample must be int or float")

    rng = np.random.default_rng(random_state)
    sampled_users = rng.choice(users, size=k, replace=False)

    return data[data[user_id].isin(sampled_users)]

def shuffle_timestamp_collisions(data, user_id="user_id", timestamp="timestamp", random_state=42):
    data = data.copy()
    
    rng = np.random.default_rng(random_state)
    data["_rand"] = rng.random(len(data))

    data = data.sort_values([user_id, timestamp, "_rand"]).drop(columns="_rand")
    data = data.reset_index(drop=True)
    return data
