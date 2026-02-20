"""
Repeated consumption page.
"""

import warnings

import pandas as pd
import streamlit as st

from src.stats.duplicates import _duplicate_counts, get_all_duplicates
from src.stats.plots import plot_hist_px
from streamlit_ui.utils import load_data, page_config


warnings.filterwarnings('ignore')
st.set_page_config(layout="wide", page_title="Repeat Consumption Analysis")
pd.set_option('display.max_columns', None)

page_config()

type_map = {
    'Repetitive Interact.':'item_duplicate',
    'Consecutive Repeats':'consec_duplicate',
}


def main():

    st.title(f'Repeat Consumption Patterns')
    st.markdown(f'**Dataset:** {st.session_state.selected_dataset}' + (f' | **Split:** {st.session_state.selected_split}' if st.session_state.selected_split else ''))


    st.write("This page contains statistics for the repeat consumption patterns analysis.")
    st.write("The first group of statistics is related to general **repeat consumption patterns**.")
    with st.expander("More details on Repeated Interactions"):
        st.write('''
        Repeated interactions are interactions of the same user with an item multiple times, like buying the same good again after some time. \
        For example, if a user has the following interactions: [1, 2, 3, **2**, 5, **5**, **5**], \
            then the repeated interactions are all interactions with an item except the first one, i.e. [2] and [5, 5] here. \n
        ''')
        col, _ = st.columns([1, 1.2])
        with col:
            st.image("streamlit_ui/pictures/repeated_consumption.png", width='stretch')

        st.write('''
        Presence of repeated interactions in the dataset could lead to overestimation of the model performance. 
        Repeat consumption patterns could influence some recommendation pipeline steps, like filtering seen items from the recommendations, \
        or, on the contrary, filtering them out evaluation subset. \
        Such decisions should be motivated and reported to ensure the fairness of the evaluation.
        ''')
    st.markdown("The second group of statistics is related more specifically to the **consecutive user interactions with the same item**.")
    with st.expander("More details on Consecutive Repeated Interactions"):
        st.write('''
        Consecutive interactions are interactions of the same user with an item multiple times consecutively, like listening to the same song on repeat. \
        For example, if a user has the following interactions: [1, 2, 3, 2, 5, **5**, **5**], \
            then the consecutive repeats are all consecutive interactions with an item except the first one, i.e. [5, 5] here. \n
        ''')
        col, _ = st.columns([1, 1.2])
        with col:
            st.image("streamlit_ui/pictures/consecutive_repeats.png", width='stretch')

        st.write('''
        The consecutive interactions could make the model learn to recommend the same item repeatedly, decreasing recommendations utility. \
        Presence of consecutive repeats in evaluation subset could lead to overestimation of the model performance. \
        Thus it is important to be aware of presence of consecutive interactions in the dataset to aggregate or remove them if needed. \
        Such decisions should be motivated and reported to ensure the fairness of the evaluation.
        ''')

    st.markdown("### Select Subset")
    available_subsets = list(st.session_state.df_paths.keys())

    subset = st.selectbox(
            "Select subset",
            available_subsets,
            index=available_subsets.index(st.session_state.default_subset),
            key="base",
            label_visibility="collapsed"
        )

    col1, col2 = st.columns([0.2, 0.8])
    
    with col1:
        st.subheader("Repetition type")
        type_chosen = st.radio(
            "Select subset",
            type_map.keys(),
            index=0,
            key="duplicate_type",
            label_visibility="collapsed"
        )
        duplicate_type = type_map.get(type_chosen)
        consider_no_repeats = st.checkbox('Consider Users with no Repetitions')
    
    with st.spinner(text="Calculating statistics...", show_time=False, width="content"):
        df = load_data(subset)
        df_flags = get_all_duplicates(df)
        df_stats = _duplicate_counts(df_flags, duplicate_type, consider_no_repeats)

    with col2:
        st.subheader(f" {type_chosen}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Repeated Interactions", f"{df_flags[duplicate_type].sum():,}")
        col5.metric("All interactions", f"{len(df_flags):,}")
        col6.metric("Share of Repeats", f"{df_flags[duplicate_type].mean():.4f}")

        if df_stats['Number of Users'] != 0:
            col1, col2, col3 = st.columns(3)
            col1.metric("Users with Repeats", f"{df_stats['Number of Users']:,}")
            col2.metric("Total Users", f"{df_flags['user_id'].nunique():,}")
            col3.metric("Share of Users", f"{df_stats['Share of Users']:.4f}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Avg. Number per User", f"{df_stats['Avg. Number per user']:.4f}")
            col2.metric("Avg. Sequence Length", f"{df_flags.groupby('user_id')['item_id'].count().mean():.4f}")
            col3.metric("Avg. Share per User", f"{df_stats['Avg. Share per user']:.4f}")

    if df_stats['Number of Users'] != 0:
        # st.divider()
        col1, col2 = st.columns([0.2, 0.8])

        with col1:
            st.markdown("#### Plot settings")
            hist_type = st.radio(
                    "Select data type",
                    ['Number of Interactions per User', 'Share of Interactions per User'],
                    key="hist_type"
                )

            num_bins = st.slider(
                    "Number of bins for histogram", min_value=5, max_value=100, value=50, step=1, key='bins'
                )

        with col2:
            agg_df = df_flags.groupby('user_id')
            agg_df = agg_df.mean() if "Share" in hist_type else agg_df.sum()

            if consider_no_repeats:
                relevant_users = agg_df.index
            else:
                relevant_users = agg_df[agg_df[duplicate_type] > 0].index

            filtered_series = agg_df.loc[relevant_users, :]

            duplicate_hist, stats_to_display = plot_hist_px(
                filtered_series,
                col=duplicate_type,
                nbins=st.session_state.bins,
                labels=[type_chosen],
                opacity=0.7,
                color_discrete_sequence=['lightblue'],
            )

            st.plotly_chart(duplicate_hist, config={"width":"stretch"})

        st.dataframe(stats_to_display)


if __name__ == "__main__":
    main()
