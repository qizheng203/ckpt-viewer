from __future__ import annotations

import plotly.express as px
import streamlit as st

from ckpt_viewer.quant_analysis import generate_quant_strategy
from ckpt_viewer.schemas import CheckpointSummary, TensorRecord
from ckpt_viewer.utils import records_to_dataframe


def _empty_hint() -> None:
    st.info("当前过滤条件下没有可显示的 tensor。可以关闭“仅分析可量化权重”或启用“包含 buffer”。")


def render(
    summary: CheckpointSummary,
    records: list[TensorRecord],
    *,
    topk: int = 20,
    include_buffers_default: bool = False,
) -> None:
    st.info(
        "该页面只基于 checkpoint 权重做静态 fake quant 分析，不包含 activation calibration，"
        "不代表最终任务精度。真实 PTQ / INT8 效果仍需结合校准数据、验证集指标和部署后推理结果确认。"
    )

    st.subheader("Recommended Quantization Strategy")
    strategy = generate_quant_strategy(summary, records)
    cols = st.columns(4)
    cols[0].metric("Weight quantization", strategy["recommended_weight_quantization"])
    cols[1].metric("PTQ risk", strategy["ptq_risk"])
    cols[2].metric("FP16 baseline", strategy["fp16_baseline"])
    cols[3].metric("Fallback candidates", len(strategy["fallback_candidates"]))
    st.caption(strategy["next_step"])

    include_buffers = st.checkbox(
        "Include norm/buffer tensors in plots",
        value=include_buffers_default,
    )

    df = records_to_dataframe(records)
    if df.empty:
        _empty_hint()
        return
    if include_buffers:
        plot_df = df[df["is_float"]]
    else:
        plot_df = df[df["include_in_quant_analysis"]]
    plot_df = plot_df.copy()
    if "quant_risk_score" in plot_df.columns:
        plot_df["plot_risk_size"] = plot_df["quant_risk_score"].fillna(1.0).clip(lower=1.0)

    if plot_df.empty:
        _empty_hint()
        return

    hover_cols = [
        "group",
        "role",
        "type_guess",
        "shape_str",
        "absmax_p999_ratio",
        "per_tensor_int8_mse",
        "per_channel_int8_mse",
        "per_tensor_int8_sqnr",
        "per_channel_int8_sqnr",
        "sqnr_gain",
        "quant_risk_score",
        "recommendation",
    ]
    hover_cols = [column for column in hover_cols if column in plot_df.columns]

    left, right = st.columns(2)
    with left:
        sqnr_df = (
            plot_df.dropna(subset=["per_tensor_int8_sqnr"])
            .sort_values("per_tensor_int8_sqnr")
            .head(topk)
        )
        if not sqnr_df.empty:
            st.plotly_chart(
                px.bar(
                    sqnr_df.sort_values("per_tensor_int8_sqnr", ascending=False),
                    x="per_tensor_int8_sqnr",
                    y="name",
                    orientation="h",
                    title=f"per-tensor SQNR 最差 Top {topk}",
                    hover_data=hover_cols,
                ),
                use_container_width=True,
            )

        scatter_df = plot_df.dropna(subset=["absmax_p999_ratio", "per_tensor_int8_mse"])
        if not scatter_df.empty:
            st.plotly_chart(
                px.scatter(
                    scatter_df,
                    x="absmax_p999_ratio",
                    y="per_tensor_int8_mse",
                    color="quant_risk_level",
                    size="plot_risk_size",
                    hover_name="name",
                    hover_data=hover_cols,
                    opacity=0.7,
                    title="absmax_p999_ratio vs per_tensor_mse",
                ),
                use_container_width=True,
            )

        fallback_df = (
            plot_df[plot_df["fp16_fallback_candidate"]]
            .sort_values("quant_risk_score", ascending=False)
            .head(topk)
        )
        if not fallback_df.empty:
            st.plotly_chart(
                px.bar(
                    fallback_df.sort_values("quant_risk_score", ascending=True),
                    x="quant_risk_score",
                    y="name",
                    orientation="h",
                    title=f"FP16 fallback candidate Top {topk}",
                    hover_data=hover_cols + ["fallback_reason"],
                ),
                use_container_width=True,
            )

    with right:
        ch_df = (
            plot_df.dropna(subset=["per_channel_int8_sqnr"])
            .sort_values("per_channel_int8_sqnr")
            .head(topk)
        )
        if not ch_df.empty:
            st.plotly_chart(
                px.bar(
                    ch_df.sort_values("per_channel_int8_sqnr", ascending=False),
                    x="per_channel_int8_sqnr",
                    y="name",
                    orientation="h",
                    title=f"per-channel SQNR 最差 Top {topk}",
                    hover_data=hover_cols,
                ),
                use_container_width=True,
            )

        gain_df = (
            plot_df.dropna(subset=["sqnr_gain"])
            .sort_values("sqnr_gain", ascending=False)
            .head(topk)
        )
        if not gain_df.empty:
            st.plotly_chart(
                px.bar(
                    gain_df.sort_values("sqnr_gain", ascending=True),
                    x="sqnr_gain",
                    y="name",
                    orientation="h",
                    title=f"per-channel 收益 Top {topk}",
                    hover_data=hover_cols,
                ),
                use_container_width=True,
            )

        recommended_df = gain_df[gain_df["sqnr_gain"] > 5] if not gain_df.empty else gain_df
        if recommended_df is not None and not recommended_df.empty:
            st.plotly_chart(
                px.bar(
                    recommended_df.sort_values("mse_reduction", ascending=True),
                    x="mse_reduction",
                    y="name",
                    orientation="h",
                    title=f"per-channel recommended layers Top {topk}",
                    hover_data=hover_cols,
                ),
                use_container_width=True,
            )

    mse_df = plot_df.dropna(subset=["per_tensor_int8_mse", "per_channel_int8_mse"])
    if not mse_df.empty:
        st.plotly_chart(
            px.scatter(
                mse_df,
                    x="per_tensor_int8_mse",
                    y="per_channel_int8_mse",
                    color="quant_risk_level",
                    size="plot_risk_size",
                hover_name="name",
                hover_data=hover_cols,
                opacity=0.7,
                title="per_tensor_mse vs per_channel_mse",
            ),
            use_container_width=True,
        )
