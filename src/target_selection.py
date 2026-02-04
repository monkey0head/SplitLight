from typing import Tuple, Union

import pandas as pd


def leave_first(
    holdout_data: pd.DataFrame, input_data: pd.DataFrame = None
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Split data into input and target interactions.

    For warm users (in input_data), use first interaction as target.
    For cold users, use first interaction as input and second as target.
    If no input_data provided, use first interaction as target for all users.

    Args:
        holdout_data: DataFrame containing user-item interactions
        input_data: Optional existing interactions to combine with inputs

    Returns:
        If input_data is None: DataFrame of target interactions
        Else: tuple of (input_data, target_interactions)
    """
    data_sorted = holdout_data.sort_values(["user_id", "timestamp"], kind="stable")

    if input_data is not None:
        warm_users = input_data["user_id"].unique()
        warm_user_mask = data_sorted["user_id"].isin(warm_users)

        # For warm users: take first interaction as target
        warm_target = (
            data_sorted[warm_user_mask].groupby("user_id").apply(lambda x: x.iloc[0]).reset_index()
        )

        # For cold users:
        cold_full = data_sorted[~warm_user_mask].groupby("user_id")

        cold_input = cold_full.apply(lambda x: x.iloc[0]).reset_index()  # first interaction becomes input
        cold_target = cold_full.apply(lambda x: x.iloc[1] if len(x) > 1 else None).dropna().reset_index()

        targets = pd.concat([warm_target, cold_target], ignore_index=True)

        final_input = pd.concat([input_data, cold_input]).sort_values(
            ["user_id", "item_id"], kind="stable"
        )
    else:
        # If no input data is specified, take first interaction as input, second as target
        final_input = data_sorted.groupby("user_id").apply(lambda x: x.iloc[0]).reset_index()
        targets = data_sorted.groupby("user_id").apply(lambda x: x.iloc[1] if len(x) > 1 else None).dropna().reset_index()

    final_input['timestamp'] = final_input['timestamp'].astype(int)
    targets['timestamp'] = targets['timestamp'].astype(int)
    return final_input, targets


def leave_last(
    holdout_data: pd.DataFrame,
    input_data: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into input and target interactions using leave-last-out strategy.

    For each user:
    - Last interaction becomes target
    - All previous interactions become input

    Args:
        holdout_data: DataFrame containing user-item interactions
        input_data: Optional existing interactions to combine with inputs

    Returns:
        tuple of (input_interactions, target_interactions) if input_data is provided
        or just target_interactions if input_data is None
    """
    data_sorted = holdout_data.sort_values(["user_id", "timestamp"], kind="stable")

    # Get last interaction per user as target
    targets = data_sorted.groupby("user_id").apply(lambda x: x.iloc[-1]).reset_index(drop=True)

    # Select all interactions except last as inputs
    final_input = data_sorted.groupby("user_id").head(-1).reset_index(drop=True)

    if input_data is not None:
        final_input = pd.concat([input_data, final_input]).sort_values(
            ["user_id", "item_id"], kind="stable"
        )
    return final_input, targets


def leave_random(
    holdout_data: pd.DataFrame, input_data: pd.DataFrame = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits data into input and target, keeping a random interaction as target.

    For warm users: selects random interaction as target
    For cold users: selects random interaction (excluding first) as target
    All previous interactions become input.

    Args:
        holdout_data: DataFrame containing user-item interactions
        input_data: Optional existing interactions to combine with inputs

    Returns:
        Tuple of (input_data, target_data)
    """
    data_sorted = holdout_data.sort_values(["user_id", "timestamp"], kind="stable")
    data_sorted["time_id"] = data_sorted.groupby("user_id").cumcount()

    if input_data is not None:
        warm_users = input_data["user_id"].unique()
        warm_user_mask = data_sorted["user_id"].isin(warm_users)

        # For warm users: take any random interaction as target
        warm_full = data_sorted[warm_user_mask]
        warm_target = warm_full.groupby("user_id").sample(n=1)

        # For cold users: take random interaction except the first one
        cold_full = data_sorted[~warm_user_mask]
        cold_target = (
            cold_full[cold_full["time_id"] > 0].groupby("user_id").sample(n=1)
            if not cold_full.empty
            else cold_full
        )  # exclude first interaction

        targets = pd.concat([warm_target, cold_target])
    else:
        # If no warm users specified, select random non-first interaction as target
        targets = data_sorted[data_sorted["time_id"] > 0].groupby("user_id").sample(n=1)

    max_time_id_map = targets.set_index("user_id")["time_id"]
    data_sorted["sampled_time_id"] = data_sorted["user_id"].map(max_time_id_map)

    # Select all interactions before the target as input
    final_input = data_sorted[
        data_sorted["time_id"] < data_sorted["sampled_time_id"]
    ].drop(columns=["time_id", "sampled_time_id"])

    if input_data is not None:
        final_input = pd.concat([input_data, final_input]).sort_values(
            ["user_id", "item_id"], kind="stable"
        )

    return final_input, targets.drop(columns=["time_id"])
