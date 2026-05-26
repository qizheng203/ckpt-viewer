from __future__ import annotations

import plotly.express as px
import streamlit as st

from ckpt_viewer.schemas import TensorRecord
from ckpt_viewer.utils import records_to_dataframe


def render(records: list[TensorRecord], *, topk: int = 20) -> None:
    st.info(
        "BN / Norm Risk 用于提示部署时 Conv-BN folding 可能带来的数值放大风险。"
        "该分析不等价于真实融合后的权重范围。若需要精确判断，应在导出 ONNX 或部署图后再次分析 initializer。"
    )
    df = records_to_dataframe([record for record in records if record.include_in_bn_risk])
    if df.empty:
        st.info("未检测到 running_mean / running_var / norm_weight / norm_bias。")
        return

    cols = st.columns(4)
    cols[0].metric("BN/Norm tensors", f"{len(df):,}")
    cols[1].metric("High risk", f"{int((df['bn_risk_score'].fillna(0) >= 60).sum()):,}")
    cols[2].metric("Medium risk", f"{int(((df['bn_risk_score'].fillna(0) >= 30) & (df['bn_risk_score'].fillna(0) < 60)).sum()):,}")
    cols[3].metric("running_var", f"{int((df['role'] == 'running_var').sum()):,}")

    display_cols = [
        "name",
        "role",
        "shape_str",
        "dtype",
        "min",
        "p01",
        "abs_max",
        "absmax_p999_ratio",
        "bn_running_var_min",
        "bn_running_var_p001",
        "bn_running_var_p01",
        "bn_running_var_small_ratio_1e_5",
        "bn_risk_score",
        "bn_risk_level",
        "recommendation",
    ]
    st.dataframe(
        df.sort_values("bn_risk_score", ascending=False, na_position="last")[
            [column for column in display_cols if column in df.columns]
        ],
        use_container_width=True,
        hide_index=True,
    )

    left, right = st.columns(2)
    with left:
        running_var = df[df["role"] == "running_var"]
        if not running_var.empty:
            top_min = running_var.dropna(subset=["bn_running_var_min"]).sort_values(
                "bn_running_var_min"
            ).head(topk)
            if not top_min.empty:
                st.plotly_chart(
                    px.bar(
                        top_min.sort_values("bn_running_var_min", ascending=False),
                        x="bn_running_var_min",
                        y="name",
                        orientation="h",
                        title=f"running_var min Top {topk} smallest",
                        hover_data=["bn_running_var_p01", "bn_running_var_small_ratio_1e_5"],
                    ),
                    use_container_width=True,
                )

            top_p01 = running_var.dropna(subset=["bn_running_var_p01"]).sort_values(
                "bn_running_var_p01"
            ).head(topk)
            if not top_p01.empty:
                st.plotly_chart(
                    px.bar(
                        top_p01.sort_values("bn_running_var_p01", ascending=False),
                        x="bn_running_var_p01",
                        y="name",
                        orientation="h",
                        title=f"running_var p01 Top {topk} smallest",
                    ),
                    use_container_width=True,
                )
            if running_var["min"].notna().any():
                st.plotly_chart(
                    px.histogram(
                        running_var.dropna(subset=["min"]),
                        x="min",
                        nbins=80,
                        title="running_var distribution histogram",
                    ),
                    use_container_width=True,
                )

    with right:
        gamma = df[df["role"] == "norm_weight"]
        if not gamma.empty:
            gamma_top = gamma.dropna(subset=["abs_max"]).sort_values("abs_max", ascending=False).head(topk)
            if not gamma_top.empty:
                st.plotly_chart(
                    px.bar(
                        gamma_top.sort_values("abs_max", ascending=True),
                        x="abs_max",
                        y="name",
                        orientation="h",
                        title=f"gamma abs_max Top {topk}",
                        hover_data=["absmax_p999_ratio", "bn_risk_score", "recommendation"],
                    ),
                    use_container_width=True,
                )

        risk_top = df.dropna(subset=["bn_risk_score"]).sort_values("bn_risk_score", ascending=False).head(topk)
        if not risk_top.empty:
            st.plotly_chart(
                px.bar(
                    risk_top.sort_values("bn_risk_score", ascending=True),
                    x="bn_risk_score",
                    y="name",
                    orientation="h",
                    color="bn_risk_level",
                    title=f"BN risk Top {topk}",
                    hover_data=["role", "recommendation"],
                ),
                use_container_width=True,
            )
