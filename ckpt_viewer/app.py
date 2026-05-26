"""Streamlit Web UI entrypoint."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

import streamlit as st

from ckpt_viewer.analyzer import analyze_checkpoint
from ckpt_viewer.loaders import get_file_info
from ckpt_viewer.schemas import AnalysisResult
from ckpt_viewer.ui import bn_norm, export, layer_detail, overview, quantization, tensor_table


def _write_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded_file.getbuffer())
        return Path(handle.name)


def _progress_callback(progress_bar, status) -> Callable[[str, float | None], None]:
    def callback(message: str, fraction: float | None = None) -> None:
        if fraction is not None:
            progress_bar.progress(min(max(float(fraction), 0.0), 1.0))
        status.write(message)

    return callback


def _run_analysis(
    path: Path,
    *,
    trusted: bool,
    near_zero_threshold: float,
    cache_key: tuple,
) -> AnalysisResult:
    cache: dict[tuple, AnalysisResult] = st.session_state.setdefault("analysis_cache", {})
    if cache_key in cache:
        return cache[cache_key]

    progress_bar = st.progress(0.0)
    status = st.empty()
    result = analyze_checkpoint(
        path,
        trusted=trusted,
        near_zero_threshold=near_zero_threshold,
        progress_callback=_progress_callback(progress_bar, status),
    )
    cache[cache_key] = result
    return result


def main() -> None:
    st.set_page_config(
        page_title="Ckpt_viewer V2",
        page_icon="CK",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Ckpt_viewer V2")
    st.caption("本工具只做静态权重分析，不执行模型 forward，不替代校准集和验证集评估。")

    with st.sidebar:
        st.header("Input")
        uploaded_file = st.file_uploader(
            "拖拽上传权重文件",
            type=["pth", "pt", "ckpt", "bin", "safetensors"],
        )
        local_path = st.text_input("本地文件路径", "")
        trusted = st.checkbox(
            "信任该文件并使用 unsafe load",
            value=False,
            help="仅当权重文件来自可信来源且安全加载失败时启用。未知来源的 .pth 可能包含不安全的 pickle 内容。",
        )
        near_zero_threshold = st.selectbox(
            "near_zero 阈值",
            [1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
            index=2,
            format_func=lambda value: f"{value:g}",
            help="用于统计 abs(weight) 小于该阈值的近零权重比例，主要用于稀疏性和剪枝潜力分析。",
        )
        histogram_samples = st.number_input(
            "直方图最大采样数",
            min_value=1_000,
            max_value=5_000_000,
            value=1_000_000,
            step=10_000,
            help="仅影响图表绘制，不应影响完整统计指标。大模型可适当调小以提升交互速度。",
        )
        topk = st.number_input(
            "Top-K 数量",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            help="控制排行榜图表展示前 K 个 tensor。",
        )
        exclude_buffers = st.checkbox(
            "默认排除 buffer",
            value=True,
            help="影响 Overview / Quantization Risk / Tensor Table 的默认展示过滤。",
        )
        quantizable_only = st.checkbox(
            "仅分析可量化权重",
            value=False,
            help="如果启用，Tensor Table 默认只显示 include_in_quant_analysis=True 的 tensor。",
        )
        report_level = st.selectbox("报告详细程度", ["basic", "standard", "detailed"], index=1)
        analyze = st.button("开始分析", type="primary", use_container_width=True)

    if analyze:
        try:
            if local_path.strip():
                path = Path(local_path.strip()).expanduser()
            elif uploaded_file is not None:
                path = _write_uploaded_file(uploaded_file)
            else:
                st.error("请上传权重文件，或输入本地文件路径。")
                return

            file_info = get_file_info(path)
            cache_key = (
                str(path.resolve()),
                file_info["sha256"],
                bool(trusted),
                float(near_zero_threshold),
                int(histogram_samples),
                int(topk),
                bool(exclude_buffers),
                bool(quantizable_only),
                str(report_level),
            )
            result = _run_analysis(
                path,
                trusted=trusted,
                near_zero_threshold=float(near_zero_threshold),
                cache_key=cache_key,
            )
            st.session_state["analysis_result"] = result
            st.success("分析完成")
        except Exception as exc:
            st.error(str(exc))
            with st.expander("错误详情"):
                st.exception(exc)

    result = st.session_state.get("analysis_result")
    if result is None:
        st.info("选择权重文件后点击开始分析。")
        return

    tabs = st.tabs(
        [
            "Overview",
            "Tensor Table",
            "Layer Detail",
            "Quantization Risk",
            "BN / Norm Risk",
            "Export",
        ]
    )
    with tabs[0]:
        overview.render(
            result.summary,
            result.records,
            topk=int(topk),
            include_buffers=not bool(exclude_buffers),
        )
    with tabs[1]:
        tensor_table.render(
            result.records,
            default_exclude_buffers=bool(exclude_buffers),
            default_quantizable_only=bool(quantizable_only),
        )
    with tabs[2]:
        layer_detail.render(
            result.records,
            result.tensors,
            histogram_samples=int(histogram_samples),
        )
    with tabs[3]:
        quantization.render(
            result.summary,
            result.records,
            topk=int(topk),
            include_buffers_default=not bool(exclude_buffers),
        )
    with tabs[4]:
        bn_norm.render(result.records, topk=int(topk))
    with tabs[5]:
        export.render(
            result.summary,
            result.records,
            topk=int(topk),
            include_buffers=not bool(exclude_buffers),
            report_level=str(report_level),
        )


if __name__ == "__main__":
    main()
