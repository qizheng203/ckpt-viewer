from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from ckpt_viewer.schemas import CheckpointSummary, TensorRecord
from ckpt_viewer.utils import records_to_dataframe


def _empty_hint() -> None:
    st.info("当前过滤条件下没有可显示的 tensor。可以关闭“仅分析可量化权重”或启用“包含 buffer”。")


def render(
    summary: CheckpointSummary,
    records: list[TensorRecord],
    *,
    topk: int = 20,
    include_buffers: bool = False,
) -> None:
    cols = st.columns(6)
    cols[0].metric("Tensors", f"{summary.total_tensors:,}")
    cols[1].metric("Weight-like", f"{summary.weight_like_tensors:,}")
    cols[2].metric("Buffers", f"{summary.buffer_tensors:,}")
    cols[3].metric("Quantizable", f"{summary.quantizable_tensors:,}")
    cols[4].metric("Params", f"{summary.total_params:,}")
    cols[5].metric("Health", f"{summary.weight_health_score:.1f}")

    cols = st.columns(5)
    cols[0].metric("Trainable-like params", f"{summary.trainable_like_params:,}")
    cols[1].metric("FP32 MB", f"{summary.fp32_size_mb:.2f}")
    cols[2].metric("FP16 MB", f"{summary.fp16_size_mb:.2f}")
    cols[3].metric("INT8 MB", f"{summary.int8_size_mb:.2f}")
    cols[4].metric("Risk", summary.quantization_risk.upper())

    st.caption(f"Load mode: {summary.load_mode} | SHA256: {summary.sha256}")
    st.subheader("Diagnosis Summary")
    st.info(summary.executive_summary)
    if summary.warnings:
        for warning in summary.warnings:
            st.warning(warning)

    df = records_to_dataframe(records)
    if df.empty:
        _empty_hint()
        return

    group_df = pd.DataFrame(
        [
            {
                "group": key,
                "params": value,
                "percent": summary.group_param_percent.get(key, 0.0),
                "tensor_count": summary.group_tensor_counts.get(key, 0),
            }
            for key, value in summary.group_param_counts.items()
        ]
    )
    dtype_tensor_df = pd.DataFrame(
        [{"dtype": key, "tensor_count": value} for key, value in summary.dtype_tensor_counts.items()]
    )
    dtype_param_df = pd.DataFrame(
        [{"dtype": key, "params": value} for key, value in summary.dtype_param_counts.items()]
    )

    left, right = st.columns(2)
    with left:
        if not group_df.empty:
            st.plotly_chart(
                px.bar(
                    group_df.sort_values("params", ascending=False),
                    x="group",
                    y="params",
                    title="Parameter distribution by group",
                    hover_data=["percent", "tensor_count"],
                ),
                use_container_width=True,
            )
        if not dtype_tensor_df.empty:
            st.plotly_chart(
                px.bar(
                    dtype_tensor_df.sort_values("tensor_count", ascending=False),
                    x="dtype",
                    y="tensor_count",
                    title="Dtype by tensor count",
                ),
                use_container_width=True,
            )

    with right:
        if not dtype_param_df.empty:
            st.plotly_chart(
                px.bar(
                    dtype_param_df.sort_values("params", ascending=False),
                    x="dtype",
                    y="params",
                    title="Dtype by parameter count",
                ),
                use_container_width=True,
            )

        abs_df = df[df["is_float"] & df["abs_max"].notna()]
        if not include_buffers:
            abs_df = abs_df[abs_df["is_weight_like"]]
        if not abs_df.empty:
            metric_df = abs_df.sort_values("abs_max", ascending=False).head(topk)
            st.plotly_chart(
                px.bar(
                    metric_df.sort_values("abs_max", ascending=True),
                    x="abs_max",
                    y="name",
                    orientation="h",
                    title=(
                        f"Top {topk} abs_max floating tensors"
                        if include_buffers
                        else f"Top {topk} abs_max floating weight-like tensors"
                    ),
                    hover_data=["role", "type_guess", "group", "shape_str", "recommendation"],
                ),
                use_container_width=True,
            )
        else:
            _empty_hint()

    quant_df = df[df["include_in_quant_analysis"] & df["quant_risk_score"].notna()]
    ratio_df = df[
        df["is_float"]
        & df["is_weight_like"]
        & df["absmax_p999_ratio"].notna()
    ]
    left, right = st.columns(2)
    with left:
        if not quant_df.empty:
            top_df = quant_df.sort_values("quant_risk_score", ascending=False).head(topk)
            st.plotly_chart(
                px.bar(
                    top_df.sort_values("quant_risk_score", ascending=True),
                    x="quant_risk_score",
                    y="name",
                    orientation="h",
                    title=f"Top {topk} quant risk layers",
                    hover_data=["role", "type_guess", "group", "shape_str", "recommendation"],
                ),
                use_container_width=True,
            )
        else:
            _empty_hint()
    with right:
        if not ratio_df.empty:
            top_df = ratio_df.sort_values("absmax_p999_ratio", ascending=False).head(topk)
            st.plotly_chart(
                px.bar(
                    top_df.sort_values("absmax_p999_ratio", ascending=True),
                    x="absmax_p999_ratio",
                    y="name",
                    orientation="h",
                    title=f"Top {topk} absmax_p999_ratio",
                    hover_data=["role", "type_guess", "group", "shape_str", "recommendation"],
                ),
                use_container_width=True,
            )
        else:
            _empty_hint()
