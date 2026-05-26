from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from ckpt_viewer.report import export_report_bundle
from ckpt_viewer.schemas import CheckpointSummary, TensorRecord
from ckpt_viewer.utils import records_to_dataframe


def render(
    summary: CheckpointSummary,
    records: list[TensorRecord],
    *,
    topk: int = 20,
    include_buffers: bool = False,
    report_level: str = "standard",
) -> None:
    out_dir = st.text_input("导出目录", "report")
    formats = st.multiselect(
        "导出格式",
        ["csv", "json", "md", "html"],
        default=["csv", "json", "md", "html"],
    )
    chosen_report_level = st.selectbox(
        "报告详细程度",
        ["basic", "standard", "detailed"],
        index=["basic", "standard", "detailed"].index(report_level.lower())
        if report_level.lower() in {"basic", "standard", "detailed"}
        else 1,
    )
    if st.button("导出报告"):
        exported = export_report_bundle(
            summary,
            records,
            Path(out_dir),
            topk=topk,
            formats=set(formats),
            include_buffers=include_buffers,
            report_level=chosen_report_level,
        )
        st.success(f"已导出到：{Path(out_dir).resolve()}")
        st.json({key: str(value) for key, value in exported.items()})

    df = records_to_dataframe(records)
    st.download_button(
        "下载 tensor_stats.csv",
        df.to_csv(index=False).encode("utf-8-sig"),
        file_name="tensor_stats.csv",
        mime="text/csv",
    )
    st.download_button(
        "下载 summary.json",
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="summary.json",
        mime="application/json",
    )
    st.download_button(
        "下载 recommendations.json",
        json.dumps(summary.recommended_strategy, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="recommendations.json",
        mime="application/json",
    )
