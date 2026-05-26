from __future__ import annotations

import pandas as pd
import streamlit as st

from ckpt_viewer.schemas import TensorRecord
from ckpt_viewer.utils import records_to_dataframe


DEFAULT_COLUMNS = [
    "name",
    "role",
    "type_guess",
    "group",
    "shape_str",
    "numel",
    "dtype",
    "abs_max",
    "abs_p999",
    "absmax_p999_ratio",
    "zero_ratio",
    "near_zero_ratio",
    "per_tensor_int8_sqnr",
    "per_channel_int8_sqnr",
    "sqnr_gain",
    "mse_reduction",
    "channel_absmax_max_median_ratio",
    "quant_risk_score",
    "quant_risk_level",
    "fp16_fallback_candidate",
    "recommendation",
]


def _style_risk(df: pd.DataFrame):
    def color(value: str) -> str:
        if value == "High":
            return "background-color: #fecaca; color: #7f1d1d"
        if value == "Medium":
            return "background-color: #fef3c7; color: #78350f"
        if value == "Low":
            return "background-color: #dcfce7; color: #14532d"
        return "background-color: #e5e7eb; color: #374151"

    if "quant_risk_level" not in df.columns:
        return df
    return df.style.map(color, subset=["quant_risk_level"])


def render(
    records: list[TensorRecord],
    *,
    default_exclude_buffers: bool = True,
    default_quantizable_only: bool = False,
) -> None:
    df = records_to_dataframe(records)
    if df.empty:
        st.info("没有 tensor 统计结果。")
        return

    left, right = st.columns(2)
    with left:
        search = st.text_input("Search name", "")
        groups = st.multiselect("Group", sorted(df["group"].dropna().unique().tolist()))
        types = st.multiselect("Type", sorted(df["type_guess"].dropna().unique().tolist()))
        roles = st.multiselect("Role", sorted(df["role"].dropna().unique().tolist()))
    with right:
        dtypes = st.multiselect("Dtype", sorted(df["dtype"].dropna().unique().tolist()))
        levels = st.multiselect(
            "Risk Level",
            sorted(df["quant_risk_level"].dropna().unique().tolist()),
        )
        quantizable_only = st.checkbox("Only quantizable weights", value=default_quantizable_only)
        exclude_buffers = st.checkbox("Exclude buffers", value=default_exclude_buffers)

    abnormal_only = st.checkbox("Only abnormal tensors", value=False)
    fallback_only = st.checkbox("Only FP16 fallback candidates", value=False)

    filtered = df
    if search:
        filtered = filtered[filtered["name"].str.contains(search, case=False, na=False)]
    if groups:
        filtered = filtered[filtered["group"].isin(groups)]
    if types:
        filtered = filtered[filtered["type_guess"].isin(types)]
    if roles:
        filtered = filtered[filtered["role"].isin(roles)]
    if dtypes:
        filtered = filtered[filtered["dtype"].isin(dtypes)]
    if levels:
        filtered = filtered[filtered["quant_risk_level"].isin(levels)]
    if quantizable_only:
        filtered = filtered[filtered["include_in_quant_analysis"]]
    if exclude_buffers:
        filtered = filtered[~filtered["is_buffer"]]
    if abnormal_only:
        filtered = filtered[
            (filtered["nan_count"] > 0)
            | (filtered["inf_count"] > 0)
            | (filtered["quant_risk_score"].fillna(0) >= 30)
            | (filtered["bn_risk_score"].fillna(0) >= 30)
        ]
    if fallback_only:
        filtered = filtered[filtered["fp16_fallback_candidate"]]

    filtered = filtered.sort_values("quant_risk_score", ascending=False, na_position="last")
    st.caption(
        "当前排序：quant_risk_score 降序 | "
        f"当前过滤：quantizable weights only = {quantizable_only}, exclude buffers = {exclude_buffers}"
    )
    st.caption("复制层名或在 Layer Detail 页面选择该层查看详细图表。")

    columns = [column for column in DEFAULT_COLUMNS if column in filtered.columns]
    if filtered.empty:
        st.info("当前过滤条件下没有可显示的 tensor。可以关闭“仅分析可量化权重”或启用“包含 buffer”。")
    else:
        st.dataframe(
            _style_risk(filtered[columns]),
            use_container_width=True,
            hide_index=True,
        )

    st.download_button(
        "下载当前表格 CSV",
        filtered.to_csv(index=False).encode("utf-8-sig"),
        file_name="tensor_stats_filtered.csv",
        mime="text/csv",
    )
