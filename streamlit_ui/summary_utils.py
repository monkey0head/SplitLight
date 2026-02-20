import numpy as np
import pandas as pd
import streamlit as st
import yaml
import uuid

from src.stats.cold import cold_stats
from src.stats.duplicates import get_all_duplicates
from src.stats.leaks import (find_shared_interactions, leak_counts,
                             temporal_overlap)
from streamlit_ui.utils import stats_map


# ---------------- color map ----------------
color_map = {
    "ok": "#22c55e",            # green-500
    "need attention": "#f59e0b",# amber-500
    "warning": "#ef4444",       # red-500
    "info": "#3b82f6",          # blue-500
}


# ============== Config loader & flatteners =================
@st.cache_data
def load_summary_config(path: str = "config/summary.yml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


@st.cache_data
def flatten_thresholds(cfg: dict):
    """
    Returns:
      thresholds_base: {metric_id: {"type","direction","ok","caution"}}
      meta:            {metric_id: {"info":bool,"messages":{...},"label":str,"type":..., "direction":..., "info_per_subset":{...}}}
      overrides:       {metric_id: {subset: {"type","direction","ok","caution"}}}
    """
    thresholds_base, meta, overrides = {}, {}, {}
    sections = cfg.get("sections", {})
    for section in sections.values():
        for group in (section.get("groups") or {}).values():
            for metric_id, m in (group.get("metrics") or {}).items():
                meta.setdefault(metric_id, {})
                meta[metric_id]["info"] = bool(m.get("info", False))
                meta[metric_id]["messages"] = m.get("messages", {}) or {}
                meta[metric_id]["label"] = m.get("label", stats_map.get(metric_id, metric_id))
                meta[metric_id]["type"] = m.get("type", "val")
                meta[metric_id]["direction"] = m.get("direction", "high")
                meta[metric_id]["info_per_subset"] = m.get("info_per_subset", {}) or {}

                if not meta[metric_id]["info"]:
                    th = m.get("thresholds") or {}
                    thresholds_base[metric_id] = {
                        "type": m.get("type", "val"),
                        "direction": m.get("direction", "high"),
                        "ok": th.get("ok", 0),
                        "caution": th.get("caution", 0),
                    }

                    # per-subset overrides
                    per = m.get("per_subset") or {}
                    if per:
                        overrides.setdefault(metric_id, {})
                        for subset, sub_cfg in per.items():
                            overrides[metric_id][subset] = {
                                "type": sub_cfg.get("type", m.get("type", "val")),
                                "direction": sub_cfg.get("direction", m.get("direction", "high")),
                                "ok": sub_cfg.get("ok", th.get("ok", 0)),
                                "caution": sub_cfg.get("caution", th.get("caution", 0)),
                            }
    return thresholds_base, meta, overrides


@st.cache_data
def get_quality_with_subsets(stats: pd.DataFrame, base_cfg: dict, overrides: dict):
    """Same shape as stats; uses per-subset overrides when present."""
    qual_df = pd.DataFrame(np.nan, index=stats.index, columns=stats.columns)
    has_multiindex = isinstance(stats.columns, pd.MultiIndex)

    all_metrics = list(base_cfg.keys() | overrides.keys())

    for metric in all_metrics:
        cfg0 = overrides.get(metric, {}).get(stats.index[0], base_cfg.get(metric, {"type": "val"}))
        col_type = "Abs. value" if cfg0["type"] == "val" else "%"

        for subset_name in stats.index:
            cfg = overrides.get(metric, {}).get(subset_name, base_cfg.get(metric))
            if cfg is None:
                continue
            if has_multiindex and (metric, col_type) in stats.columns:
                v = stats.loc[subset_name, (metric, col_type)]
                qual_df.loc[subset_name, (metric, col_type)] = categorize(v, cfg)
            elif (not has_multiindex) and metric in stats.columns:
                v = stats.loc[subset_name, metric]
                qual_df.loc[subset_name, metric] = categorize(v, cfg)

    return qual_df


@st.cache_data
def build_thresholds_for_subset(subset_name: str, base: dict, overrides: dict) -> dict:
    resolved = {}
    keys = set(base.keys()) | set(overrides.keys())
    for k in keys:
        resolved[k] = overrides.get(k, {}).get(subset_name, base.get(k))
    return resolved


# ================== categorize / get_quality ==================
@st.cache_data
def categorize(value, cfg):
    ok = cfg["ok"]; caution = cfg["caution"]; direction = cfg["direction"]
    if pd.isna(value):
        return np.nan
    if direction == "high":
        if value >= ok: return "ok"
        elif value >= caution: return "need attention"
        else: return "warning"
    elif direction == "low":
        if value <= ok: return "ok"
        elif value <= caution: return "need attention"
        else: return "warning"
    else:
        raise ValueError(f"Unknown direction: {direction}")

@st.cache_data
def get_quality(stats, config):
    qual_df = pd.DataFrame(np.nan, index=stats.index, columns=stats.columns)
    has_multiindex = isinstance(stats.columns, pd.MultiIndex)
    for metric, cfg in config.items():
        if has_multiindex:
            col_type = "Abs. value" if cfg["type"] == "val" else "%"
            if (metric, col_type) in stats.columns:
                qual_df[(metric, col_type)] = stats[(metric, col_type)].apply(lambda v: categorize(v, cfg))
        else:
            if metric in stats.columns:
                qual_df[metric] = stats[metric].apply(lambda v: categorize(v, cfg))
    return qual_df


def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _rgb(rgb, a=None):
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {1 if a is None else a})"


def _mix(rgb, t=0.70):
    # mix with white; t controls tint strength (lower = stronger color)
    return tuple(round(255 * t + c * (1 - t)) for c in rgb)


def render_card(
    title: str,
    value,
    color: str = "#3b82f6",
    border_radius: str = "14px",
    text_color: str  = None,
    padding: str = "14px 16px",
    title_font_size: int = 14,
    value_font_size: int = 23,
    tooltip: str = ""
):
    """Self-contained KPI card with CSS :hover (no JS), safe across page navigation."""
    uid = f"kpi-{uuid.uuid4().hex[:8]}"

    base = _hex_to_rgb(color)
    bg_soft = _mix(base, t=0.70)                          # stronger tint
    grad    = f"linear-gradient(180deg, {_rgb(base, 0.10)} 0%, {_rgb(base, 0.06)} 100%), {_rgb(bg_soft)}"
    bd_soft = _mix(base, t=0.72)
    halo    = _rgb(base, 0.16)
    halo_hover = _rgb(base, 0.24)
    bd_hover   = _rgb(base, 0.25)

    badge_bg = _rgb(base, 0.12)
    badge_bd = _rgb(base, 0.25)
    badge_fg = "#0b1220"
    fg_color = "#0b1220" if text_color is None else text_color

    if isinstance(value, list):
        value_ab, value_pct = value[0], value[1]
        value_html = f"""
          <div style="display:flex; align-items:baseline; gap:10px;">
            <div style="font-weight:700; line-height:1.05; font-size:{value_font_size}px">{value_ab}</div>
            <div style="font-size:12px; font-weight:700; padding:3px 8px; border-radius:999px;
                        background:{badge_bg}; border:1px solid {badge_bd}; color:{badge_fg};
                        letter-spacing:.2px;">{value_pct}</div>
          </div>
        """
    else:
        value_html = f'<div style="font-weight:700; line-height:1.05; font-size:{value_font_size}px">{value}</div>'

    css = f"""
    <style>
      /* scoped to this single card via unique id */
      #{uid} {{
        position: relative;
        background: {grad};
        color: {fg_color};
        border: 1px solid {_rgb(bd_soft, 0.5)};
        border-radius: {border_radius};
        padding: {padding};
        box-shadow: 0 8px 26px {halo};
        font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
        transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease;
        overflow: hidden;
      }}
      #{uid}:hover {{
        transform: translateY(-2px);
        box-shadow: 0 18px 40px {halo_hover};
        border-color: {bd_hover};
      }}
      #{uid} .kpi-title {{
        font-weight: 600;
        letter-spacing: .2px;
        color: #111827;
        margin-bottom: 10px;
        font-size: {title_font_size}px;
      }}
      #{uid} .kpi-stripe {{
        position: absolute; inset: 0 auto 0 0; width: 6px; background: {color};
        border-top-left-radius: {border_radius}; border-bottom-left-radius: {border_radius}; opacity: .95;
      }}
    </style>
    """

    html = f"""
      {css}
      <div id="{uid}" title="{tooltip}">
        <div class="kpi-stripe"></div>
        <div class="kpi-title">{title}</div>
        {value_html}
      </div>
    """
    return "\n".join(line.lstrip() for line in html.splitlines()).strip()


@st.cache_data
def color_for_category(cat: str):
    return color_map.get(cat, "#95a5a6")


# ================== CategoryAggregator & popovers ======================
class CategoryAggregator:
    """Collects every box rendered on the page for proper counting and popover text."""
    def __init__(self, ignore_info: bool = True):
        self.items = []      # list of dicts: section, title, metric_id, category, message
        self.ignore_info = ignore_info

    def add(self, *, section: str, title: str, metric_id: str, category: str, message: str  = None):
        if self.ignore_info and category == "info":
            return
        self.items.append({
            "section": section,
            "title": title,
            "metric_id": metric_id,
            "category": category,
            "message": message or "",
        })

    def counts(self):
        out = {"ok": 0, "need attention": 0, "warning": 0}
        for it in self.items:
            if it["category"] in out:
                out[it["category"]] += 1
        return out

    def popovers(self):
        """Compose HTML text for the popovers."""
        by_cat = {"ok": [], "need attention": [], "warning": []}
        for it in self.items:
            if it["category"] in by_cat:
                by_cat[it["category"]].append(it)

        def lead(cat):
            return {
                "ok": "These metrics are <b>OK</b>.",
                "need attention": "Worth checking — maybe <b>Need Attention</b>.",
                "warning": "Likely action needed — <b>Warning</b>."
            }[cat]

        def split_ref(items):
            ref, splits = [], []
            for it in items:
                (ref if str(it["section"]).lower().startswith("reference") else splits).append(it)
            keyf = lambda x: (x["section"], x["title"])
            return sorted(ref, key=keyf), sorted(splits, key=keyf)

        def bullet_line(it):
            if it["message"]:
                return it["message"]
            alias = "Train" if "Train" in it["section"] else ("Targets" if "Targets" in it["section"] else ("Input" if "Input" in it["section"] else ""))
            combined = f"{alias} {it['title']}".strip()
            bold = f"<b>{combined}</b>"
            if it["category"] == "ok":
                return f"{bold} is adequate."
            verb = "Review" if it["category"] == "need attention" else "Fix"
            return f"{verb} {bold}."

        def ul(items):
            if not items:
                return '<ul style="margin:4px 0 8px 16px; padding:0;"><li><i>None</i></li></ul>'
            lis = [f'<li style="margin:2px 0">{bullet_line(it)}</li>' for it in items]
            return '<ul style="margin:4px 0 8px 16px; padding:0;">' + "".join(lis) + '</ul>'

        def h5(txt): return f'<h5 style="margin:12px 0 -17px">{txt}</h5>'

        out = {}
        for cat in ("ok", "need attention", "warning"):
            ref_items, split_items = split_ref(by_cat[cat])
            html = [
                f'<p style="margin:0 0 6px">{lead(cat)}</p>',
                '<hr style="border:none;border-top:1px solid #eee;margin:8px 0"/>',
                h5("Reference Dataset:"),
                ul(ref_items),
                h5("Split:"),
                ul(split_items),
            ]
            out[cat] = "".join(html)
        return out


# ================== formatting helpers ======================
@st.cache_data
def format_time(amount: float) -> str:
    conversion_rules = {'seconds': (60, 'minutes'), 'minutes': (60, 'hours'),
                        'hours': (24, 'days'), 'days': (365, 'years')}
    unit_lower = 'seconds'
    while unit_lower in conversion_rules:
        divisor, next_unit = conversion_rules[unit_lower]
        if amount >= divisor:
            amount /= divisor
            unit_lower = next_unit
        else:
            break
    return f"{round(amount, 1)} {unit_lower}"


def _format_value(metric, value, pct=None):
    """Shared formatter used for card and message variables."""
    if metric == "density":
        value *= 100
        abs_str = f"{value:.3f}%"
    elif metric == "timestamp_collisions, %":
        abs_str = f"{value:.2f}%"
    elif metric in ['timeframe', 'mean_time_between_interactions', 'mean_user_lifetime', 'mean_item_lifetime']:
        abs_str = format_time(value)
    else:
        abs_str = f"{int(value):,}" if value % 1 == 0 else f"{value:,.2f}"
    if pct is not None:
        return [abs_str, f"{pct:.2f}%"]
    return abs_str

def _message_from_template(tpl: str , **kwargs) -> str :
    if not tpl:
        return None
    try:
        return tpl.format(**kwargs)
    except Exception:
        return tpl


# ================== metrics card renderer ====================
def metrics_cards(
    subset_name, df, qual_df, metrics, color_map,
    section_key: str  = None,
    config_meta: dict  = None,
    thresholds_cfg: dict  = None,
    aggregator: CategoryAggregator  = None,
):
    """Render metric cards AND (optionally) register them with the aggregator."""
    has_multiindex = isinstance(df.columns, pd.MultiIndex)
    df_subset = df.loc[subset_name]
    qual_subset = qual_df.loc[subset_name] if qual_df is not None else None
    cols = st.columns(len(metrics))

    for i, metric in enumerate(metrics):
        meta = (config_meta or {}).get(metric, {})
        title = meta.get("label", stats_map.get(metric, metric))

        if has_multiindex:
            abs_val = df_subset[metric]['Abs. value']
            pct_val = df_subset[metric]['%']
            val_str = _format_value(metric, abs_val, pct_val)
            cat = "info"
            if qual_subset is not None:
                cat_vals = qual_subset[metric].dropna().values
                cat = cat_vals.item() if cat_vals.size > 0 else "info"
        else:
            val = df_subset[metric]
            val_str = _format_value(metric, val)
            cat = qual_subset[metric] if qual_subset is not None else "info"

        subset_toggle = (meta.get("info_per_subset", {}) or {}).get(subset_name, False)
        if meta.get("info", False) or subset_toggle:
            cat = "info"

        color = color_map.get(cat, '#6c8ebf')
        html = render_card(title, val_str, color=color)
        cols[i].markdown(html, unsafe_allow_html=True)

        if aggregator is not None and section_key:
            th_cfg = (thresholds_cfg or {}).get(metric, {})
            th_ok = th_cfg.get("ok"); th_caution = th_cfg.get("caution")
            value_abs = float(abs_val) if has_multiindex else (float(val) if isinstance(val_str, str) else None)
            value_pct = float(pct_val) if has_multiindex else None
            value_fmt = val_str[0] if isinstance(val_str, list) else val_str

            messages = meta.get("messages", {})
            tpl = messages.get(cat)
            message = _message_from_template(
                tpl,
                value_abs=value_abs,
                value_pct=value_pct,
                value_fmt=value_fmt,
                ok=th_ok, caution=th_caution
            )
            aggregator.add(section=section_key, title=title, metric_id=metric, category=cat, message=message or "")


# ================== Clickable category cards  =================
def render_clickable_card_popover(col, title, value, color, key, tooltip="", popover_html=""):
    """
    Popover implemented with <details>/<summary> to avoid any <style> or JS.
    Robust across page navigation & Streamlit sanitization.
    """
    card_html = render_card(title=title, value=value, color=color, tooltip=tooltip, value_font_size=30, title_font_size=15)

    container_style = "position:relative; display:block; margin:0;"
    summary_style   = "list-style:none; outline:none; cursor:pointer; margin:0; padding:0; display:block;"
    panel_style = (
        "position:absolute; top:calc(100% + 10px); left:0; z-index:9999;"
        "min-width:340px; max-width:560px; background:#ffffff; color:#111;"
        "border:1px solid rgba(0,0,0,.08); box-shadow:0 10px 28px rgba(0,0,0,.18);"
        "border-radius:12px; padding:14px 16px; font-family:'Inter', sans-serif; line-height:1.35;"
    )
    arrow_style = (
        "position:absolute; top:-8px; left:24px; width:16px; height:16px; background:#ffffff;"
        "transform:rotate(45deg); border-left:1px solid rgba(0,0,0,.08); border-top:1px solid rgba(0,0,0,.08);"
        "box-shadow:-3px -3px 10px rgba(0,0,0,.06);"
    )

    html = f"""
    <details id="wrap-{key}" style="{container_style}">
      <summary style="{summary_style}">
        {card_html}
      </summary>
      <div style="{panel_style}">
        <div style="{arrow_style}"></div>
        <div>{popover_html}</div>
      </div>
    </details>
    """
    col.markdown(html, unsafe_allow_html=True)

def display_category_cards(counts: dict, popovers_by_cat: dict  = None):
    categories = ["ok", "need attention", "warning", "info"]
    colors = {"ok":"#22c55e", "need attention":"#f59e0b", "warning":"#ef4444", "info":"#3b82f6"}
    titles = {"ok":"OK","need attention":"Need Attention","warning":"Warning","info":"Info"}

    summary_col, cols = st.columns([0.4, 0.6])
    with cols:
        cols = st.columns(len(categories))
        for i, cat in enumerate(categories):
            color = colors[cat]; title = titles[cat]
            value = "–" if cat == "info" else str(int(counts.get(cat, 0)))

            if cat == "info":
                pop_html = "<p><b>Info</b> metrics are descriptive only and usually do not affect your data quality decisions.</p>"
            else:
                pop_html = (popovers_by_cat or {}).get(cat, "<i>No details.</i>")

            render_clickable_card_popover(
                col=cols[i],
                title=title,
                value=value,
                color=color,
                key=f"card-{cat.replace(' ', '-')}",
                tooltip="Click to open",
                popover_html=pop_html,
            )

    with summary_col:
        if int(counts.get("warning", 0)) > 0:
            status, status_color = "WARNING", color_map["warning"]
        elif int(counts.get("need attention", 0)) > 0:
            status, status_color = "NEED ATTENTION", color_map["need attention"]
        else:
            status, status_color = "OK", color_map["ok"]

        st.markdown(f"""
            <div style="background-color: white20; border-right:4px solid white; padding:8px; text-align:center;">
                <div style="font-size:20px; font-weight:600; color:#111827; margin-bottom: -18px;">Overall Verdict:</div>
                <div style="font-size:50px; font-weight:900; color:{status_color}">{status}</div>
            </div>
        """, unsafe_allow_html=True)


# ================== precompute_category_summary ======================
def precompute_category_summary(
    *,
    base_subset: str,
    stats, qual_df,
    ref_stats, ref_qual_df,
    base_thresholds_ref, subset_overrides_ref, meta_ref,
    base_thresholds_main, subset_overrides_main, meta_main,
    base_df, all_df,
):
    """
    Build the top-row category counts & popover details BEFORE rendering any boxes.
    Keeps reference-vs-main thresholds separate and respects info/info_per_subset.
    """
    agg = CategoryAggregator(ignore_info=True)

    # ---------- helpers ----------
    def _alias(section: str) -> str:
        s = section or ""
        if not s.startswith("Main"):
            return ""
        if "Train" in s:   return "Train"
        if "Targets" in s: return "Targets"
        if "Input" in s:   return "Input"
        return "Split"

    def _render_msg(tpl: str , **kwargs) -> str :
        if not tpl:
            return None
        try:
            return tpl.format(**kwargs)
        except Exception:
            return tpl

    # ---------- resolve thresholds for each area ----------
    th_ref         = build_thresholds_for_subset(base_subset,   base_thresholds_ref,  subset_overrides_ref)
    th_train       = build_thresholds_for_subset("train",       base_thresholds_main, subset_overrides_main)
    th_test_target = build_thresholds_for_subset("test_target", base_thresholds_main, subset_overrides_main)
    th_test_input  = build_thresholds_for_subset("test_input",  base_thresholds_main, subset_overrides_main)

    def _add_from_table(stats_df, qdf, subset, section, metrics, th, meta):
        """Register metrics that live in a stats table (MultiIndex or flat)."""
        has_multi = isinstance(stats_df.columns, pd.MultiIndex)
        subset_alias = _alias(section)

        for m in metrics:
            mmeta  = meta.get(m, {}) or {}
            title  = mmeta.get("label", stats_map.get(m, m))

            subset_info = (mmeta.get("info_per_subset", {}) or {}).get(subset, False)
            if mmeta.get("info", False) or subset_info:
                agg.add(section=section, title=title, metric_id=m, category="info")
                continue

            if has_multi:
                typ = (th.get(m, {}) or {}).get("type", mmeta.get("type", "val"))
                col_type = "Abs. value" if typ == "val" else "%"

                if (m, col_type) not in stats_df.columns:
                    continue

                abs_val = stats_df.loc[subset, (m, "Abs. value")] if (m, "Abs. value") in stats_df.columns else None
                pct_val = stats_df.loc[subset, (m, "%")]          if (m, "%") in stats_df.columns else None

                try:
                    q_series = qdf.loc[subset, m]
                    cat_vals = q_series.dropna().values
                    cat = cat_vals.item() if cat_vals.size > 0 else "info"
                except Exception:
                    cat = "info"

                val_fmt = _format_value(m, abs_val if abs_val is not None else 0, pct_val if pct_val is not None else None)
                th_cfg  = th.get(m, {}) or {}
                tpl     = (mmeta.get("messages", {}) or {}).get(cat)

                msg = _render_msg(
                    tpl,
                    label=title,
                    subset=subset_alias,
                    combined_label=(f"{subset_alias} {title}").strip(),
                    value_abs=float(abs_val) if abs_val is not None else None,
                    value_pct=float(pct_val) if pct_val is not None else None,
                    value_fmt=val_fmt[0] if isinstance(val_fmt, list) else val_fmt,
                    ok=th_cfg.get("ok"),
                    caution=th_cfg.get("caution"),
                )
            else:
                if m not in stats_df.columns:
                    continue
                v = stats_df.loc[subset, m]
                try:
                    cat = qdf.loc[subset, m]
                except Exception:
                    cat = "info"

                val_fmt = _format_value(m, v)
                th_cfg  = th.get(m, {}) or {}
                tpl     = (mmeta.get("messages", {}) or {}).get(cat)

                msg = _render_msg(
                    tpl,
                    label=title,
                    subset=subset_alias,
                    combined_label=(f"{subset_alias} {title}").strip(),
                    value_abs=float(v) if isinstance(v, (int, float, np.floating)) else None,
                    value_pct=None,
                    value_fmt=val_fmt,
                    ok=th_cfg.get("ok"),
                    caution=th_cfg.get("caution"),
                )

            agg.add(section=section, title=title, metric_id=m, category=cat, message=msg or "")

    # ---------- Reference section ----------
    _add_from_table(
        ref_stats, ref_qual_df, base_subset, "Reference • Core",
        ['n_interactions', 'n_users', 'n_items', 'avg_seq_length', 'density'],
        th_ref, meta_ref
    )
    _add_from_table(
        ref_stats, ref_qual_df, base_subset, "Reference • Temporal",
        ['timeframe', 'mean_time_between_interactions', 'mean_user_lifetime', 'mean_item_lifetime', 'timestamp_collisions, %'],
        th_ref, meta_ref
    )

    # Reference • Repetitive
    dups = get_all_duplicates(base_df).mean(axis=0)
    for m in ["consec_duplicate", "item_duplicate"]:
        mmeta  = meta_ref.get(m, {}) or {}
        title  = mmeta.get("label", stats_map.get(m, m))
        th_cfg = base_thresholds_ref.get(m, {}) or {}

        v_raw = float(dups[m])
        value_for_cat = v_raw * 100 if mmeta.get("type", "val") == "pct" else v_raw
        cat = categorize(value_for_cat, th_cfg) if th_cfg else "info"

        tpl = (mmeta.get("messages", {}) or {}).get(cat)
        msg = _render_msg(
            tpl,
            label=title,
            subset="",
            combined_label=title,
            value_abs=value_for_cat,
            value_pct=None,
            value_fmt=f"{v_raw:.2%}",
            ok=th_cfg.get("ok"),
            caution=th_cfg.get("caution"),
        )
        agg.add(section="Reference • Repetitive", title=title, metric_id=m, category=cat, message=msg or "")

    # ---------- Main section ----------
    core_metrics = ['n_interactions', 'n_users', 'n_items', 'avg_seq_length', 'timeframe']

    _add_from_table(stats, qual_df, "test_target", "Main • Targets vs Ref", core_metrics, th_test_target, meta_main)
    _add_from_table(stats, qual_df, "test_input",  "Main • Input vs Ref",   core_metrics, th_test_input,  meta_main)
    if base_subset != "train":
        _add_from_table(
            stats, qual_df, "train",
            "Main • Train vs Ref",
            core_metrics, th_train, meta_main
        )

    # Cold Start
    cold_df = cold_stats(all_df["test_target"], all_df["train"])
    for idx, row in cold_df.iterrows():
        entity = "user" if idx.lower().startswith("cold users") else "item"
        metric = f"cold_{entity}_share"
        mmeta  = meta_main.get(metric, {}) or {}
        title  = mmeta.get("label", stats_map.get(idx, idx))
        th_cfg = th_test_target.get(metric, {}) or {}

        value_pct = row["Share (by count)"] * 100.0
        cat = categorize(value_pct, th_cfg) if th_cfg else "info"

        tpl = (mmeta.get("messages", {}) or {}).get(cat)
        msg = _render_msg(
            tpl,
            label=title,
            subset="Targets",
            combined_label=f"Targets {title}",
            value_abs=value_pct,
            value_pct=None,
            value_fmt=f"{row['Share (by count)']:.2%}",
            ok=th_cfg.get("ok"),
            caution=th_cfg.get("caution"),
        )
        agg.add(section="Main • Cold Start", title=title, metric_id=metric, category=cat, message=msg or "")

    # Data Leakage (vs TRAIN always)
    summary_stats = leak_counts(all_df["test_target"], all_df["train"])
    leak_pct = float(summary_stats["leak_share"]) * 100.0

    mmeta  = meta_main.get("leak_interactions_share", {}) or {}
    th_cfg = base_thresholds_main.get("leak_interactions_share", {}) or {}
    cat    = categorize(leak_pct, th_cfg) if th_cfg else "info"
    tpl    = (mmeta.get("messages", {}) or {}).get(cat)
    msg    = _render_msg(
        tpl,
        label=mmeta.get("label","Leaked Interactions"),
        subset="Targets",
        combined_label=f"Targets {mmeta.get('label','Leaked Interactions')}",
        value_abs=leak_pct,
        value_pct=None,
        value_fmt=f"{leak_pct:.2f}%",
        ok=th_cfg.get("ok"),
        caution=th_cfg.get("caution"),
    )
    agg.add(section="Main • Data Leakage",
            title=mmeta.get("label","Leaked Interactions"),
            metric_id="leak_interactions_share",
            category=cat,
            message=msg or "")

    # Temporal Overlap
    overlap_row = temporal_overlap(all_df["test_target"], all_df["train"]).iloc[0]
    overlap_yes = 1 if overlap_row["overlap_duration_sec"] > 0 else 0
    cat_overlap = "warning" if overlap_yes else "ok"
    meta_overlap = meta_main.get("temporal_overlap", {}) or {}
    label_overlap = meta_overlap.get("label", "Temporal Overlap")
    tpl_ov = (meta_overlap.get("messages", {}) or {}).get(cat_overlap)
    msg_ov = _render_msg(
        tpl_ov,
        label=label_overlap,
        subset="Targets",
        combined_label=f"Targets {label_overlap}",
        value_abs=overlap_yes,
        value_pct=None,
        value_fmt=("Yes" if overlap_yes else "No"),
        ok=0, caution=0,
    )
    agg.add(section="Main • Data Leakage",
            title=label_overlap,
            metric_id="temporal_overlap",
            category=cat_overlap,
            message=msg_ov or "")
    
    # Has Overlapping Interactions (Targets vs Train)
    has_overlap_inter = int(not find_shared_interactions(all_df["test_target"], all_df["train"]).empty)
    cat_overlap_inter = "warning" if has_overlap_inter else "ok"

    meta_overlap_inter = meta_main.get("has_overlapping_interactions", {}) or {}
    label_overlap_inter = meta_overlap_inter.get("label", "Overlapping Interactions")
    tpl_oi = (meta_overlap_inter.get("messages", {}) or {}).get(cat_overlap_inter)
    msg_oi = _render_msg(
        tpl_oi,
        label=label_overlap_inter,
        subset="Targets",
        combined_label=f"Targets {label_overlap_inter}",
        value_abs=has_overlap_inter,
        value_pct=None,
        value_fmt=("Yes" if has_overlap_inter else "No"),
        ok=0, caution=0,
    )
    agg.add(section="Main • Data Leakage",
            title=label_overlap_inter,
            metric_id="has_overlapping_interactions",
            category=cat_overlap_inter,
            message=msg_oi or "")

    return agg.counts(), agg.popovers()


def build_reference_only_summary(
    *,
    base_subset: str,
    ref_stats: pd.DataFrame,
    ref_qual_df: pd.DataFrame,
    base_thresholds_ref: dict,
    subset_overrides_ref: dict,
    meta_ref: dict,
    base_df: pd.DataFrame,
):
    """
    Aggregate ONLY the Reference section into the top-row category popovers.
    Returns: (counts_dict, popovers_dict)
    """
    agg = CategoryAggregator(ignore_info=True)
    th_ref = build_thresholds_for_subset(base_subset, base_thresholds_ref, subset_overrides_ref)
    has_multi = isinstance(ref_stats.columns, pd.MultiIndex)

    def _add_from_table(stats_df, qdf, subset, section, metrics, th, meta):
        for m in metrics:
            mmeta  = meta.get(m, {}) or {}
            title  = mmeta.get("label", stats_map.get(m, m))

            # Respect global 'info'
            if mmeta.get("info", False):
                agg.add(section=section, title=title, metric_id=m, category="info")
                continue

            if has_multi:
                typ = (th.get(m, {}) or {}).get("type", mmeta.get("type", "val"))
                col_type = "Abs. value" if typ == "val" else "%"
                if (m, col_type) not in stats_df.columns:
                    continue

                abs_val = stats_df.loc[subset, (m, "Abs. value")] if (m, "Abs. value") in stats_df.columns else None
                pct_val = stats_df.loc[subset, (m, "%")]          if (m, "%") in stats_df.columns else None

                try:
                    q_series = qdf.loc[subset, m]
                    cat_vals = q_series.dropna().values
                    cat = cat_vals.item() if cat_vals.size > 0 else "info"
                except Exception:
                    cat = "info"

                val_fmt = _format_value(m, abs_val if abs_val is not None else 0, pct_val if pct_val is not None else None)
                th_cfg  = th.get(m, {}) or {}
                tpl     = (mmeta.get("messages", {}) or {}).get(cat)
                try:
                    msg = tpl.format(
                        label=title,
                        subset="",
                        combined_label=title,
                        value_abs=float(abs_val) if abs_val is not None else None,
                        value_pct=float(pct_val) if pct_val is not None else None,
                        value_fmt=val_fmt[0] if isinstance(val_fmt, list) else val_fmt,
                        ok=th_cfg.get("ok"),
                        caution=th_cfg.get("caution"),
                    ) if tpl else None
                except Exception:
                    msg = tpl
            else:
                if m not in stats_df.columns:
                    continue
                v = stats_df.loc[subset, m]
                try:
                    cat = qdf.loc[subset, m]
                except Exception:
                    cat = "info"

                val_fmt = _format_value(m, v)
                th_cfg  = th.get(m, {}) or {}
                tpl     = (mmeta.get("messages", {}) or {}).get(cat)
                try:
                    msg = tpl.format(
                        label=title,
                        subset="",
                        combined_label=title,
                        value_abs=float(v) if isinstance(v, (int, float, np.floating)) else None,
                        value_pct=None,
                        value_fmt=val_fmt,
                        ok=th_cfg.get("ok"),
                        caution=th_cfg.get("caution"),
                    ) if tpl else None
                except Exception:
                    msg = tpl

            agg.add(section=section, title=title, metric_id=m, category=cat, message=msg or "")

    # Reference Core & Temporal
    _add_from_table(
        ref_stats, ref_qual_df, base_subset, "Reference • Core",
        ['n_interactions', 'n_users', 'n_items', 'avg_seq_length', 'density'],
        th_ref, meta_ref
    )
    _add_from_table(
        ref_stats, ref_qual_df, base_subset, "Reference • Temporal",
        ['timeframe', 'mean_time_between_interactions', 'mean_user_lifetime', 'mean_item_lifetime', 'timestamp_collisions, %'],
        th_ref, meta_ref
    )

    # Reference • Repetitive (duplicates)
    dups = get_all_duplicates(base_df).mean(axis=0)
    for m in ["consec_duplicate", "item_duplicate"]:
        mmeta  = meta_ref.get(m, {}) or {}
        title  = mmeta.get("label", stats_map.get(m, m))
        th_cfg = base_thresholds_ref.get(m, {}) or {}

        v_raw = float(dups[m])
        value_for_cat = v_raw * 100 if mmeta.get("type", "val") == "pct" else v_raw
        cat = categorize(value_for_cat, th_cfg) if th_cfg else "info"

        tpl = (mmeta.get("messages", {}) or {}).get(cat)
        try:
            msg = tpl.format(
                label=title,
                subset="",
                combined_label=title,
                value_abs=value_for_cat,
                value_pct=None,
                value_fmt=f"{v_raw:.2%}",
                ok=th_cfg.get("ok"),
                caution=th_cfg.get("caution"),
            ) if tpl else None
        except Exception:
            msg = tpl
        agg.add(section="Reference • Repetitive", title=title, metric_id=m, category=cat, message=msg or "")

    return agg.counts(), agg.popovers()
