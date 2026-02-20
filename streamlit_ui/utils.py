import os

import pandas as pd
import streamlit as st
import yaml

stats_map = {
    'n_users': '#Users',
    'n_items': '#Items',
    'n_interactions': '#Interactions',
    'avg_seq_length': 'Avg. Seq. Length',
    'density': 'Density',
    'timeframe': 'Data Timeframe',
    'max_timestamp': 'Max Timestamp',
    'min_timestamp': 'Min Timestamp',
    'mean_time_between_interactions': 'Avg. Δt between Interactions',
    'median_time_between_interactions': 'Median Δt between Interactions',
    'mean_user_lifetime': 'Avg. User Lifetime',
    'median_user_lifetime': 'Median User Lifetime',
    'mean_user_lifetime, %': 'Avg. User Lifetime (%)',
    'mean_item_lifetime': 'Avg. Item Lifetime',
    'median_item_lifetime': 'Median Item Lifetime',
    'mean_item_lifetime, %': 'Avg. Item Lifetime (%)',
    'timestamp_collisions':'Timestamp Collisions',
    'timestamp_collisions, %':'Timestamp Collisions (%)',
    'mean_item_occurrence': 'Avg. Item Occurrence',
    'median_item_occurrence': 'Median Item Occurrence',
    'mean_user_activity': 'Avg. User Activity',
    'median_user_activity': 'Median User Activity',
    'consec_duplicate':'Consec. Duplicates',
    'item_duplicate':'Repetitive Interact.',

}


def filter_by_date(df, start_date, end_date):
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.set_index('timestamp')

    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date).replace(hour=23, minute=59, second=59)

    filtered_df = df[(df.index >= start_datetime) & (df.index <= end_datetime)]

    return filtered_df.reset_index()


def encode(data, col_name, shift):
    """Encode items/users to consecutive ids.

    :param col_name: column to do label encoding, e.g. 'item_id'
    :param shift: shift encoded values to start from shift
    """
    data[col_name] = data[col_name].astype("category").cat.codes + shift
    return data


@st.cache_data
def load_data(subset):
    subset_path = st.session_state.df_paths[subset]

    df = pd.read_csv(subset_path) if subset_path.endswith('.csv') else pd.read_parquet(subset_path)

    if subset == 'raw':
        with open(f'runs/configs/dataset/{st.session_state.selected_dataset}.yaml', 'r') as file:
            col_map = yaml.safe_load(file)['column_name']

        col_map = {val:i for i, val in col_map.items()}
        df = df.rename(columns=col_map)

        df = encode(df, 'item_id', 0)
        df = encode(df, 'user_id', 0)

    return df


@st.cache_data
def convert_cols(df, final_cols_ints, final_cols_floats, round_2):
    for col in final_cols_ints:
        if col in df.columns:
            df[col] = df[col].astype(int)
    for col in final_cols_floats:
        if col in df.columns:
            df[col] = df[col].astype(float).round(6)
    for col in round_2:
        if col in df.columns:
            df[col] = df[col].round(2)
    return df


@st.cache_data
def process_one_name(name, final_col_names):
    if name in final_col_names:
        return final_col_names[name]
    elif name.split('_')[0] in ['last', 'first', 'random', 'successive', 'LOO']:
        return name.split('_')[0].capitalize() if not name.split('_')[0] == 'LOO' else 'LOO'
    else:
        return name


def convert_time(df, time_unit):
    df = df.copy()
    unit_map = {
        'minutes': 60,
        'hours': 3600,
        'days': 3600 * 24,
        'years': 3600 * 24 * 365,
    }
    time_factor = unit_map.get(time_unit, 1)
    cols_to_convert = [
        'timeframe', 
        'mean_time_between_interactions', 
        'mean_user_lifetime', 
        'median_user_lifetime', 
        'mean_item_lifetime', 
        'median_item_lifetime'
    ]

    has_multiindex = isinstance(df.columns, pd.MultiIndex)

    for col in cols_to_convert:
        col = stats_map.get(col, col)
        if has_multiindex:
            df.loc[:, (col,'Abs. value')] = df[col]['Abs. value'] / time_factor
        else:
            df[col] = df[col] / time_factor

    return df


@st.cache_data
def prepare_to_print(df):
    final_cols_ints = ['n_users', 'n_items', 'n_interactions', 'timestamp_range_in_days']
    final_cols_floats = ['density', 'avg_seq_length']
    round_2 = ['avg_seq_length']

    df_to_print = convert_cols(df, final_cols_ints, final_cols_floats, round_2)
    df_to_print = df_to_print.rename(columns=stats_map)

    has_multiindex = isinstance(df.columns, pd.MultiIndex)
    for col in ['Min Timestamp', 'Max Timestamp']:
        if has_multiindex:
            df_to_print.loc[:, (col,'Abs. value')] =  pd.to_datetime(df_to_print[col]['Abs. value'], unit='s').values
            df_to_print = df_to_print.drop(columns=(col,'%'))
        else:
            df_to_print[col] = pd.to_datetime(df_to_print[col], unit='s')

    return df_to_print


def subset_selector(default_subsets=None):
    available_subsets = list(st.session_state.df_paths.keys())
    col1, col2 = st.columns(2)

    default_subsets = default_subsets or [st.session_state.default_subset]*2

    with col1:
        target_subset = st.selectbox(
            "Select analyzed subset:", 
            available_subsets, 
            index=available_subsets.index(default_subsets[1]), 
            key="target"
            )

    with col2:
        reference_subset = st.selectbox(
            "Select reference subset:", 
            available_subsets, 
            index=available_subsets.index(default_subsets[0]), 
            key="reference"
            )
    return  reference_subset, target_subset


@st.cache_data
def load_data_split(split, subset):
    if subset == 'raw' or subset == 'preprocessed':
        subset_path = os.path.join(st.session_state.data_path, st.session_state.selected_dataset, subset + '.csv')
    else:
        subset_path = os.path.join(st.session_state.data_path, st.session_state.selected_dataset, split, subset + '.csv')

    df = pd.read_csv(subset_path)

    if subset == 'raw':
        with open(f'runs/configs/dataset/{st.session_state.selected_dataset}.yaml', 'r') as file:
            col_map = yaml.safe_load(file)['column_name']

        col_map = {val:i for i, val in col_map.items()}
        df = df.rename(columns=col_map)

        df = encode(df, 'item_id', 1)
        df = encode(df, 'user_id', 0)

    return df

def page_config(layout="wide"):
    st.set_page_config(layout=layout, page_icon=":streamlit:")

    if not st.session_state.data_loaded:
        st.warning("Please load data to access this page")
        st.stop()


def select_test_and_reference_subsets():

    col1, col2 = st.columns(2)
    with col1:
        target_subset = st.selectbox("Select **test or validation** data:",
                                     options=['test', 'validation'])
    with col2:
        base_subset = st.selectbox("Select **reference** subset:",
                                   options=['preprocessed', 'raw', 'train'])

    dataset_path = os.path.join(st.session_state.data_path, st.session_state.selected_dataset)
    split_options = [f for f in os.listdir(dataset_path)
                     if os.path.isdir(os.path.join(dataset_path, f))]
    st.markdown("Select splits to compare")
    selected_splits = st.multiselect(
        "Select splits to compare",
        split_options,
        default=st.session_state.selected_split,
        label_visibility="collapsed",
    )

    return target_subset, base_subset, selected_splits
