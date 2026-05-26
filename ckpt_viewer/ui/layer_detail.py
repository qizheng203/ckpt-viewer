from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from ckpt_viewer.quant_analysis import (
    compute_channel_arrays,
    compute_channel_quant_mse,
    generate_layer_diagnosis,
    generate_quant_decision,
)
from ckpt_viewer.schemas import TensorRecord
from ckpt_viewer.tensor_stats import prepare_log_abs_values, sample_tensor_for_plot


def _record_by_name(records: list[TensorRecord]) -> dict[str, TensorRecord]:
    return {record.name: record for record in records}


def _metric_table(record: TensorRecord, title: str, fields: list[str]) -> None:
    rows = [{"metric": field, "value": getattr(record, field)} for field in fields]
    with st.expander(title, expanded=title in {"Basic Stats", "Quantization Stats"}):
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render(
    records: list[TensorRecord],
    tensors: dict[str, object],
    *,
    histogram_samples: int = 1_000_000,
) -> None:
    if not records:
        st.info("没有可查看的层。")
        return

    names = [record.name for record in records]
    default_name = st.session_state.get("selected_layer")
    default_index = names.index(default_name) if default_name in names else 0
    selected = st.selectbox("Layer", names, index=default_index)
    if selected is None:
        return
    st.session_state["selected_layer"] = selected

    record = _record_by_name(records)[selected]
    tensor = tensors[selected]
    st.subheader(selected)

    cols = st.columns(5)
    cols[0].metric("Role", record.role)
    cols[1].metric("Type", record.type_guess)
    cols[2].metric("Group", record.group)
    cols[3].metric("Params", f"{record.numel:,}")
    cols[4].metric("Risk", f"{record.quant_risk_score:.1f}" if record.quant_risk_score is not None else "N/A")

    cols = st.columns(4)
    cols[0].metric("Shape", record.shape_str)
    cols[1].metric("Dtype", record.dtype)
    cols[2].metric("Risk level", record.quant_risk_level)
    cols[3].metric("BN risk", record.bn_risk_level)

    st.info(record.recommendation)

    st.subheader("Quantization Decision")
    decision = generate_quant_decision(record)
    decision_cols = st.columns(4)
    for column, (key, value) in zip(decision_cols, decision.items()):
        column.metric(key, value)

    st.subheader("Layer Diagnosis")
    st.write(generate_layer_diagnosis(record))

    _metric_table(
        record,
        "Basic Stats",
        [
            "min",
            "max",
            "mean",
            "std",
            "abs_mean",
            "abs_max",
            "p01",
            "p05",
            "p50",
            "p95",
            "p99",
            "p999",
            "finite_ratio",
            "zero_ratio",
            "near_zero_ratio",
        ],
    )
    _metric_table(
        record,
        "Outlier Stats",
        [
            "abs_p99",
            "abs_p999",
            "absmax_p99_ratio",
            "absmax_p999_ratio",
            "outlier_ratio_3sigma",
            "outlier_ratio_6sigma",
            "skewness",
            "kurtosis",
        ],
    )
    _metric_table(
        record,
        "Quantization Stats",
        [
            "per_tensor_int8_mse",
            "per_tensor_int8_mae",
            "per_tensor_int8_max_error",
            "per_tensor_int8_sqnr",
            "per_tensor_int8_cosine",
            "per_channel_int8_mse",
            "per_channel_int8_mae",
            "per_channel_int8_max_error",
            "per_channel_int8_sqnr",
            "per_channel_int8_cosine",
            "mse_reduction",
            "sqnr_gain",
            "fp16_fallback_candidate",
            "fallback_reason",
        ],
    )
    _metric_table(
        record,
        "Channel Stats",
        [
            "channel_count",
            "channel_absmax_max",
            "channel_absmax_median",
            "channel_absmax_max_median_ratio",
            "channel_absmax_cv",
            "dead_channel_ratio",
        ],
    )
    _metric_table(
        record,
        "BN / Norm Stats",
        [
            "bn_running_var_min",
            "bn_running_var_p001",
            "bn_running_var_p01",
            "bn_running_var_small_ratio_1e_6",
            "bn_running_var_small_ratio_1e_5",
            "bn_running_var_small_ratio_1e_4",
            "bn_risk_score",
            "bn_risk_level",
        ],
    )

    samples = sample_tensor_for_plot(tensor, max_samples=histogram_samples)
    st.caption("核心统计基于 full tensor；histogram 图表基于 sampled tensor。")
    if samples.size:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                px.histogram(samples, nbins=120, title="weight histogram"),
                use_container_width=True,
            )
        with right:
            log_values = prepare_log_abs_values(samples)
            if log_values.size:
                st.plotly_chart(
                    px.histogram(
                        x=log_values,
                        nbins=120,
                        labels={"x": "log10(abs(weight) + eps)"},
                        title="log10(abs(weight) + eps) histogram",
                    ),
                    use_container_width=True,
                )

        percentile_df = pd.DataFrame(
            {
                "percentile": [0, 1, 5, 50, 95, 99, 99.9, 100],
                "value": [
                    record.min,
                    record.p01,
                    record.p05,
                    record.p50,
                    record.p95,
                    record.p99,
                    record.p999,
                    record.max,
                ],
            }
        )
        st.plotly_chart(
            px.line(
                percentile_df,
                x="percentile",
                y="value",
                markers=True,
                title="percentile curve",
            ),
            use_container_width=True,
        )

    arrays = compute_channel_arrays(tensor)
    if arrays:
        channel_mse = compute_channel_quant_mse(tensor)
        channel_df = pd.DataFrame(
            {
                "channel": list(range(int(arrays["channel_absmax"].numel()))),
                "absmax": arrays["channel_absmax"].numpy(),
                "l2norm": arrays["channel_l2norm"].numpy(),
                "scale": arrays["channel_scale"].numpy(),
                "quant_mse": channel_mse.numpy() if channel_mse.numel() else 0.0,
            }
        )
        median_abs = float(channel_df["absmax"].median()) if not channel_df.empty else 0.0
        channel_df["relative_to_median"] = channel_df["absmax"] / (median_abs + 1e-12)

        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                px.bar(channel_df, x="channel", y="absmax", title="per-channel absmax"),
                use_container_width=True,
            )
            st.plotly_chart(
                px.bar(channel_df, x="channel", y="scale", title="per-channel quant scale"),
                use_container_width=True,
            )
        with right:
            st.plotly_chart(
                px.bar(channel_df, x="channel", y="l2norm", title="per-channel L2 norm"),
                use_container_width=True,
            )
            st.plotly_chart(
                px.bar(channel_df, x="channel", y="quant_mse", title="per-channel quant MSE"),
                use_container_width=True,
            )

        st.subheader("Top abnormal channels")
        top_channels = channel_df.sort_values("relative_to_median", ascending=False).head(5)
        st.dataframe(
            top_channels[["channel", "absmax", "l2norm", "scale", "relative_to_median"]],
            use_container_width=True,
            hide_index=True,
        )
