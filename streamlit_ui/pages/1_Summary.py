"""
Summary page.
"""

import streamlit as st

import pandas as pd
import numpy as np
from src.stats.base import compare_subsets
from src.stats.cold import cold_stats
from src.stats.duplicates import get_all_duplicates
from src.stats.leaks import leak_counts, temporal_overlap, find_shared_interactions
from streamlit_ui.utils import load_data, load_data_split, page_config, stats_map
from streamlit_ui.summary_utils import (
    # core helpers / styling
    render_card, color_for_category, color_map,
    # thresholds + config
    load_summary_config, flatten_thresholds,
    build_thresholds_for_subset, get_quality_with_subsets,
    # top-row prepass
    precompute_category_summary, display_category_cards, build_reference_only_summary,
    # runtime helpers
    categorize, metrics_cards
)
from scipy.stats import ks_2samp
from src.stats.utils import combine_and_cumcount, cumcount_by_user


# ─── PAGE CONFIG ─────────────────────────────────────────────
page_config()

# ─── PAGE HEADER ─────────────────────────────────────────────
st.title("Quality Summary")
st.markdown(f'**Dataset:** {st.session_state.selected_dataset}' + (f' | **Split:** {st.session_state.selected_split}' if st.session_state.selected_split else ''))
st.markdown(
    """
    This page summarizes **reference** data quality and **split** consistency at a glance.
    Colors are driven by thresholds in your `streamlit_ui/config/summary.yml` and help you spot issues early.
    """,
)


# ─── SUBSET SELECTION ────────────────────────────────────────
available_subsets = list(st.session_state.df_paths.keys())

col1, col2 = st.columns(2)
with col1:
    target_subset = st.selectbox("Select **test or validation** split data:", options=['test', 'validation'])
with col2:
    base_subset = st.selectbox(
        "Select **reference** subset:",
        options=[s for s in available_subsets if s in ['train', 'raw', 'preprocessed']]
    )
st.caption(
    """
    **Note:** For selected **test or validation** data, we show splitting statistics (absolute values, and their % where applicable) vs. the **reference** data.
    **Reference** is the anchor subset (often `raw`, `preprocessed` or `train` data) used for comparisons. 
    """
)


# Helper: do we have the split files?
has_target = f"{target_subset}_target" in st.session_state.df_paths
has_input  = f"{target_subset}_input"  in st.session_state.df_paths
no_splits  = not (has_target and has_input)

st.text("")
st.text("")
st.markdown(
    """
    The view below reports main metrics for **reference** subset and **split-related** metrics.
    Use `More details` on the right of each section to open the dedicated deep-dive. 

    **Summary Status:** After computation, each metric is assigned a color-coded *status* (*OK*, *Need Attention*, *Warning*, and *Info*).
    The global status counter at the top aggregates all **non-Info** cards.
    **Click any status in the counter** to see exactly which metrics contribute, with brief explanations and suggestions.
    """
)
st.text("")

with st.spinner(text="Calculating statistics...", show_time=False):
    # ─── DATA LOADING ────────────────────────────────────────────
    base_df = load_data(base_subset)
    # train is always expected
    train_df = load_data("train") if not no_splits else None

    # Build all_df depending on availability
    if not no_splits:
        all_df = {
            "train": train_df,
            "test_target": load_data(f"{target_subset}_target"),
            "test_input":  load_data(f"{target_subset}_input")
        }

    # ─── LOAD YAML CONFIG & BUILD THRESHOLDS ─────────────────────
    summary_cfg = load_summary_config("streamlit_ui/config/summary.yml")

    def _cfg_section(cfg, section_key):
        return {"sections": {section_key: cfg["sections"][section_key]}}

    ref_cfg  = _cfg_section(summary_cfg, "reference")
    main_cfg = _cfg_section(summary_cfg, "main_split")

    # Flatten separately so the same metric id can have different thresholds per section
    base_thresholds_ref,  meta_ref,  subset_overrides_ref  = flatten_thresholds(ref_cfg)
    base_thresholds_main, meta_main, subset_overrides_main = flatten_thresholds(main_cfg)

    # ─── REFERENCE stats & quality (always used) ─────────────────
    ref_stats   = compare_subsets({base_subset: base_df}, None, extended=True)
    ref_qual_df = get_quality_with_subsets(ref_stats, base_thresholds_ref, subset_overrides_ref)

    # ─── MAIN stats & quality (only if splits exist) ────────────
    if not no_splits:
        stats = compare_subsets(all_df, base_df, extended=True)
        qual_df = get_quality_with_subsets(stats, base_thresholds_main, subset_overrides_main)


# Decide which summary to show
if no_splits:
    st.warning(
        f"No {target_subset} split files found (missing 'train', '{target_subset}_target' and/or '{target_subset}_input'). "
        "Showing Reference Dataset only."
    )
    counts, popovers = build_reference_only_summary(
        base_subset=base_subset,
        ref_stats=ref_stats,
        ref_qual_df=ref_qual_df,
        base_thresholds_ref=base_thresholds_ref,
        subset_overrides_ref=subset_overrides_ref,
        meta_ref=meta_ref,
        base_df=base_df,
    )
else:
    counts, popovers = precompute_category_summary(
        base_subset=base_subset,
        stats=stats, qual_df=qual_df,
        ref_stats=ref_stats, ref_qual_df=ref_qual_df,
        base_thresholds_ref=base_thresholds_ref,   subset_overrides_ref=subset_overrides_ref,   meta_ref=meta_ref,
        base_thresholds_main=base_thresholds_main, subset_overrides_main=subset_overrides_main, meta_main=meta_main,
        base_df=base_df, all_df=all_df
    )

display_category_cards(counts, popovers)
st.text("")
st.text("")

# ─── HELPER RENDERERS ────────────────────────────────────────
def render_metrics_section(title, subset_name, metrics, stats_df, qual_df,
                           color_map=color_map, link_page=None,
                           meta_for_section=None, base_th=None, subset_overrides=None):
    cols = st.columns([len(metrics)-1, 1], vertical_alignment="bottom")
    with cols[0]:
        st.markdown(f"#### {title}")
    if link_page:
        with cols[-1]:
            st.page_link(page=link_page, label="More details", icon="🔄")

    subset_thresholds = build_thresholds_for_subset(subset_name, base_th, subset_overrides)

    metrics_cards(
        subset_name, stats_df, qual_df, metrics, color_map,
        section_key=title,
        config_meta=meta_for_section,
        thresholds_cfg=subset_thresholds,
        aggregator=None
    )
    st.text("")

def render_duplicate_cards(df):
    duplicates = get_all_duplicates(df).mean(axis=0)
    metrics = ["consec_duplicate", "item_duplicate"]
    cols = st.columns(len(metrics))
    for i, metric in enumerate(metrics):
        value = duplicates[metric]
        cfg = base_thresholds_ref.get(metric)   # reference thresholds
        is_pct = (meta_ref.get(metric, {}).get("type", "val") == "pct")
        cat = categorize(value * 100 if is_pct else value, cfg) if cfg else "info"
        color = color_map.get(cat, "#6c8ebf")
        title = meta_ref.get(metric, {}).get("label", stats_map.get(metric, metric))
        cols[i].markdown(render_card(title, f"{value:.2%}", color=color), unsafe_allow_html=True)
    st.text("")

def render_cold_start(df_test, df_train):
    subset_thresholds = build_thresholds_for_subset("test_target", base_thresholds_main, subset_overrides_main)
    cold_df = cold_stats(df_test, df_train)
    cols = st.columns(2)
    for i, (idx, row) in enumerate(cold_df.iterrows()):
        entity = "user" if idx.lower().startswith("cold users") else "item"
        cfg_key = f"cold_{entity}_share"
        cfg = subset_thresholds.get(cfg_key)
        value = row["Share (by count)"] * 100.0
        cat = categorize(value, cfg) if cfg else "info"
        color = color_for_category(cat)
        val_str = f"{row['Share (by count)']:.2%}"
        title = meta_main.get(cfg_key, {}).get("label", stats_map.get(idx, idx))
        cols[i].markdown(render_card(title, val_str, color=color, tooltip=f"Category: {cat}"), unsafe_allow_html=True)
    st.text("")

def render_generic_cards(metrics):
    cols = st.columns(len(metrics))
    for i, metric in enumerate(metrics):
        value = "???"
        cols[i].markdown(
            render_card(stats_map.get(metric, metric), value, color="#6c8ebf"),
            unsafe_allow_html=True
        )
    st.text("")

def _deltas_between_subsets(df_input, df_target):
    """time gap between last INPUT and first TARGET per user"""
    last_input = df_input.sort_values(['user_id', 'timestamp']).groupby('user_id').last()['timestamp']
    first_target = df_target.sort_values(['user_id', 'timestamp']).groupby('user_id').first()['timestamp']
    deltas = pd.merge(last_input, first_target, on='user_id', suffixes=('_last_input', '_first_target'))
    deltas['delta'] = deltas['timestamp_first_target'] - deltas['timestamp_last_input']
    return deltas

def _ks_stat(a, b):
    """robust KS on numeric arrays (handles Timedelta)"""
    if len(a) == 0 or len(b) == 0:
        return np.nan
    # convert timedeltas to seconds if needed
    if np.issubdtype(a.dtype, np.timedelta64):
        a = a.astype('timedelta64[s]').astype(float)
    if np.issubdtype(b.dtype, np.timedelta64):
        b = b.astype('timedelta64[s]').astype(float)
    stat, _ = ks_2samp(a, b)
    return float(stat)

# ─── REFERENCE DATASET SECTION (always shown) ─────────────────
with st.expander("REFERENCE DATASET SUMMARY", expanded=True):
    st.subheader(f"{base_subset.title()} Dataset Summary")


    base_col1, base_col2 = st.columns([5, 2.2], vertical_alignment="bottom")
    with base_col1:
        render_metrics_section(
            "Core Statistics:", base_subset,
            ['n_interactions', 'n_users', 'n_items', 'avg_seq_length', 'density'],
            ref_stats, ref_qual_df,
            link_page="streamlit_ui/pages/2_Core_and_Temporal_Statistics.py",
            meta_for_section=meta_ref, base_th=base_thresholds_ref, subset_overrides=subset_overrides_ref
        )

    with base_col2:
        cols = st.columns(2, vertical_alignment="bottom")
        with cols[-1]:
            st.page_link(page="streamlit_ui/pages/4_Repeated_Consumption.py", label="More details", icon="🔄")
        render_duplicate_cards(base_df)

    render_metrics_section(
        "Temporal Statistics:", base_subset,
        ['timeframe', 'mean_time_between_interactions','mean_user_lifetime','mean_item_lifetime','timestamp_collisions, %'],
        ref_stats, ref_qual_df,
        link_page="streamlit_ui/pages/2_Core_and_Temporal_Statistics.py",
        meta_for_section=meta_ref, base_th=base_thresholds_ref, subset_overrides=subset_overrides_ref
    )

# ─── MAIN SPLIT STATISTICS (only if splits exist) ────────────
if not no_splits:
    with st.expander("MAIN SPLIT STATISTICS", expanded=True):
        st.subheader("Main Split Statistics")


        if base_subset != 'train':
            render_metrics_section(
                f"Training vs. {base_subset.title()} Data:", "train",
                ['n_interactions','n_users','n_items','avg_seq_length','timeframe'],
                stats, qual_df,
                link_page="streamlit_ui/pages/2_Core_and_Temporal_Statistics.py",
                meta_for_section=meta_main, base_th=base_thresholds_main, subset_overrides=subset_overrides_main
            )

        render_metrics_section(
            f"{target_subset.title()} Targets vs. {base_subset.title()} Data:", "test_target",
            ['n_interactions','n_users','n_items','avg_seq_length','timeframe'],
            stats, qual_df,
            meta_for_section=meta_main, base_th=base_thresholds_main, subset_overrides=subset_overrides_main
        )

        render_metrics_section(
            f"{target_subset.title()} Inputs vs. {base_subset.title()} Data:", "test_input",
            ['n_interactions','n_users','n_items','avg_seq_length','timeframe'],
            stats, qual_df,
            meta_for_section=meta_main, base_th=base_thresholds_main, subset_overrides=subset_overrides_main
        )

        with st.expander("What are targets/inputs? More details on splitted subsets"):
            st.markdown("""
            *SplitLight* works with the following data subsets: raw dataset, preprocessed dataset, and splitting results, including training, test, and validation parts.
            Test and validation parts are divided into **input** and **target** subsets.
            The **input** subset includes the user sequence passed to the model during the inference stage, while the **target** subset contains ground truth items.

            The illustrated examples of the splitted subsets for the global temporal split and the leave-one-out split are shown below:
            """)

            col1, col2 = st.columns([1, 1], gap="large", vertical_alignment="center")
            with col1:
                st.markdown("**Global temporal split (GTS)** with last-item targets:")
                st.image("streamlit_ui/pictures/gts-last_split.png", width='stretch')
            with col2:
                st.markdown("**Leave-one-out (LOO)** split:")
                st.image("streamlit_ui/pictures/loo_split.png", width='stretch')


        # ─── Data Leakage (actual metrics) ───────────────────────
        st.text("")
        st.text("")

        cols = st.columns([2, 1], vertical_alignment="bottom")
        with cols[0]:
            st.markdown("#### Data Leakage:")
        with cols[-1]:
            st.page_link(page="streamlit_ui/pages/5_Split:_Temporal_Leakage.py", label="More details", icon="💧")

        df_base   = all_df['train']
        df_target = all_df["test_target"]

        # Overlapping Interactions (Yes/No)
        has_overlap_inter = int(not find_shared_interactions(df_target, df_base).empty)
        cat_overlap_inter = "warning" if has_overlap_inter else "ok"
        color_overlap_inter = color_map.get(cat_overlap_inter, "#6c8ebf")
        title_overlap_inter = (meta_main.get("has_overlapping_interactions", {}).get("label", "Overlapping Interactions"))
        val_overlap_inter = "Yes" if has_overlap_inter else "No"

        # Temporal Overlap (No/Yes)
        overlap_row = temporal_overlap(df_target, df_base).iloc[0]
        overlap_yes = 1 if overlap_row["overlap_duration_sec"] > 0 else 0
        cat_overlap = "warning" if overlap_yes else "ok"
        color_overlap = color_map.get(cat_overlap, "#6c8ebf")
        title_overlap = (meta_main.get("temporal_overlap", {}).get("label", "Temporal Overlap"))
        val_overlap = "Yes" if overlap_yes else "No"

        # Leaked Interactions (abs + %)
        summary_stats = leak_counts(df_target, df_base)
        leak_abs = int(summary_stats["leak_interactions"])
        leak_pct = float(summary_stats["leak_share"]) * 100.0

        cfg_leak  = base_thresholds_main.get("leak_interactions_share")
        meta_leak = meta_main.get("leak_interactions_share", {})
        cat_leak  = categorize(leak_pct, cfg_leak) if cfg_leak else "info"
        color_leak = color_map.get(cat_leak, "#6c8ebf")
        title_leak = meta_leak.get("label", "Leaked Interactions")

        leak_cols = st.columns(3)
        leak_cols[0].markdown(
            render_card(title_overlap_inter, val_overlap_inter, color=color_overlap_inter),
            unsafe_allow_html=True
        )
        leak_cols[1].markdown(
            render_card(title_overlap, val_overlap, color=color_overlap),
            unsafe_allow_html=True
        )
        leak_cols[2].markdown(
            render_card(title_leak, [f"{leak_abs:,}", f"{leak_pct:.2f}%"], color=color_leak),
            unsafe_allow_html=True
        )

        st.text("")
        cols2 = st.columns(2)

        # Cold Start
        with cols2[0]:
            cols_cold_start = st.columns([1, 1], vertical_alignment="bottom")
            with cols_cold_start[0]:
                st.markdown("#### Cold Start:")
            with cols_cold_start[-1]:
                st.page_link(page="streamlit_ui/pages/6_Split:_Cold_Start.py", label="More details", icon="❄️")
            render_cold_start(all_df["test_target"], all_df["train"])

        with cols2[1]:
            # Data Shift (KS between selected split and reference)
            cols_shift_hdr = st.columns(4, vertical_alignment="bottom")
            with cols_shift_hdr[0]:
                st.markdown("#### Data Shift:")
            with cols_shift_hdr[1]:
                st.page_link(page="streamlit_ui/pages/8_Compare_Splits:_Time_Deltas.py", label="More details", icon="🔄")
            with cols_shift_hdr[2]:
                st.page_link(page="streamlit_ui/pages/9_Compare_Splits:_item_positions.py", label="More details", icon="🔄")

            # Load current split’s input/target for the selected test/validation
            split_name = st.session_state.selected_split
            df_input  = load_data_split(split_name, f"{target_subset}_input")
            df_target = load_data_split(split_name, f"{target_subset}_target")

            # --- Temporal Shift (KS): base = intra-user deltas on reference; split = input→target gap ---
            base_sorted = base_df.sort_values(['user_id', 'timestamp']).copy()
            base_sorted['delta'] = base_sorted.groupby('user_id')['timestamp'].diff()
            split_deltas = _deltas_between_subsets(df_input, df_target)

            ks_temporal = _ks_stat(
                base_sorted['delta'].dropna().values,
                split_deltas['delta'].dropna().values
            )

            # --- Positional Shift (KS): base = within-user positions on reference; split = positions of targets ---
            positions_base = cumcount_by_user(base_df)                         # -> DataFrame with 'cumcount'
            _, positions_target = combine_and_cumcount(df_input, df_target)    # second return has 'cumcount'

            ks_positional = _ks_stat(
                positions_base['cumcount'].dropna().values,
                positions_target['cumcount'].dropna().values
            )

            ks_cols = st.columns(2)
            ks_cols[0].markdown(
                render_card("Temporal Shift (KS)", f"{ks_temporal:.3f}" if np.isfinite(ks_temporal) else "—", color=color_map["info"]),
                unsafe_allow_html=True
            )
            ks_cols[1].markdown(
                render_card("Positional Shift (KS)", f"{ks_positional:.3f}" if np.isfinite(ks_positional) else "—", color=color_map["info"]),
                unsafe_allow_html=True
            )



        st.caption(
            """**Note: Data Leakage**, **Cold Start** and **Data Shift** only make sense when computed **against training data**. Therefore, these metrics are always computed against `train`.
            """)