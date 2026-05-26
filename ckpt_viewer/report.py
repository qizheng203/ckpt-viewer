"""CSV, JSON, Markdown, and HTML report export."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.express as px

from ckpt_viewer.quant_analysis import quant_fields
from ckpt_viewer.schemas import CheckpointSummary, TensorRecord
from ckpt_viewer.utils import ensure_dir, records_to_dataframe, write_json


def export_summary_json(summary: CheckpointSummary, out_dir: str | Path) -> Path:
    out = ensure_dir(out_dir)
    path = out / "summary.json"
    write_json(path, summary.to_dict())
    return path


def export_warnings_json(summary: CheckpointSummary, out_dir: str | Path) -> Path:
    out = ensure_dir(out_dir)
    path = out / "warnings.json"
    write_json(path, {"warnings": summary.warnings})
    return path


def export_recommendations_json(summary: CheckpointSummary, out_dir: str | Path) -> Path:
    out = ensure_dir(out_dir)
    path = out / "recommendations.json"
    write_json(path, summary.recommended_strategy)
    return path


def export_tensor_stats_csv(records: Iterable[TensorRecord], out_dir: str | Path) -> Path:
    out = ensure_dir(out_dir)
    path = out / "tensor_stats.csv"
    records_to_dataframe(records).to_csv(path, index=False)
    return path


def export_quant_stats_csv(records: Iterable[TensorRecord], out_dir: str | Path) -> Path:
    out = ensure_dir(out_dir)
    path = out / "quant_stats.csv"
    pd.DataFrame([quant_fields(record) for record in records]).to_csv(path, index=False)
    return path


def export_bn_norm_stats_csv(records: Iterable[TensorRecord], out_dir: str | Path) -> Path:
    out = ensure_dir(out_dir)
    path = out / "bn_norm_stats.csv"
    df = records_to_dataframe([record for record in records if record.include_in_bn_risk])
    columns = [
        "name",
        "role",
        "shape_str",
        "dtype",
        "min",
        "p01",
        "p05",
        "p50",
        "abs_max",
        "abs_p999",
        "absmax_p999_ratio",
        "bn_running_var_min",
        "bn_running_var_p001",
        "bn_running_var_p01",
        "bn_running_var_small_ratio_1e_6",
        "bn_running_var_small_ratio_1e_5",
        "bn_running_var_small_ratio_1e_4",
        "bn_risk_score",
        "bn_risk_level",
        "recommendation",
    ]
    if not df.empty:
        df = df[[column for column in columns if column in df.columns]]
    df.to_csv(path, index=False)
    return path


def _records_df(records: Iterable[TensorRecord]) -> pd.DataFrame:
    df = records_to_dataframe(records)
    if df.empty:
        return df
    return df.sort_values("quant_risk_score", ascending=False, na_position="last")


def _top_bar(df: pd.DataFrame, metric: str, title: str, topk: int):
    plot_df = df.dropna(subset=[metric]).sort_values(metric, ascending=False).head(topk)
    if plot_df.empty:
        return None
    return px.bar(
        plot_df.sort_values(metric, ascending=True),
        x=metric,
        y="name",
        orientation="h",
        title=title,
        hover_data=[
            column
            for column in [
                "role",
                "type_guess",
                "group",
                "shape_str",
                "quant_risk_score",
                "recommendation",
            ]
            if column in plot_df.columns
        ],
    )


def create_overview_figures(
    summary: CheckpointSummary,
    records: Iterable[TensorRecord],
    *,
    topk: int = 20,
    include_buffers: bool = False,
):
    df = _records_df(records)
    figures = {}

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
    if not group_df.empty:
        figures["overview_group_params"] = px.bar(
            group_df.sort_values("params", ascending=False),
            x="group",
            y="params",
            title="Parameter distribution by group",
            hover_data=["percent", "tensor_count"],
        )

    dtype_tensor_df = pd.DataFrame(
        [{"dtype": key, "tensor_count": value} for key, value in summary.dtype_tensor_counts.items()]
    )
    if not dtype_tensor_df.empty:
        figures["overview_dtype_tensor_count"] = px.bar(
            dtype_tensor_df.sort_values("tensor_count", ascending=False),
            x="dtype",
            y="tensor_count",
            title="Dtype by tensor count",
        )

    dtype_param_df = pd.DataFrame(
        [{"dtype": key, "params": value} for key, value in summary.dtype_param_counts.items()]
    )
    if not dtype_param_df.empty:
        figures["overview_dtype_param_count"] = px.bar(
            dtype_param_df.sort_values("params", ascending=False),
            x="dtype",
            y="params",
            title="Dtype by parameter count",
        )

    if not df.empty:
        if include_buffers:
            abs_df = df[(df["is_float"]) & df["abs_max"].notna()]
            abs_title = f"Top {topk} abs_max floating tensors"
        else:
            abs_df = df[(df["is_float"]) & (df["is_weight_like"]) & df["abs_max"].notna()]
            abs_title = f"Top {topk} abs_max floating weight-like tensors"
        fig = _top_bar(abs_df, "abs_max", abs_title, topk)
        if fig is not None:
            figures["overview_absmax"] = fig

        quant_df = df[(df["include_in_quant_analysis"]) & df["quant_risk_score"].notna()]
        fig = _top_bar(quant_df, "quant_risk_score", f"Top {topk} quant risk layers", topk)
        if fig is not None:
            figures["top_quant_risk"] = fig

        ratio_df = df[
            (df["is_float"])
            & (df["is_weight_like"])
            & df["absmax_p999_ratio"].notna()
        ]
        fig = _top_bar(ratio_df, "absmax_p999_ratio", f"Top {topk} absmax_p999_ratio", topk)
        if fig is not None:
            figures["top_absmax_p999_ratio"] = fig

        scatter_df = quant_df.dropna(subset=["per_tensor_int8_mse", "per_channel_int8_mse"])
        if not scatter_df.empty:
            figures["quant_scatter_mse"] = px.scatter(
                scatter_df,
                x="per_tensor_int8_mse",
                y="per_channel_int8_mse",
                color="quant_risk_level",
                size="quant_risk_score",
                hover_name="name",
                hover_data=[
                    "group",
                    "role",
                    "type_guess",
                    "shape_str",
                    "absmax_p999_ratio",
                    "per_tensor_int8_sqnr",
                    "per_channel_int8_sqnr",
                    "sqnr_gain",
                    "recommendation",
                ],
                title="per_tensor_mse vs per_channel_mse",
                opacity=0.7,
            )

        bn_df = df[(df["include_in_bn_risk"]) & df["bn_risk_score"].notna()]
        fig = _top_bar(bn_df, "bn_risk_score", f"Top {topk} BN / Norm risk", topk)
        if fig is not None:
            figures["bn_risk"] = fig

    return figures


def export_figures(
    summary: CheckpointSummary,
    records: Iterable[TensorRecord],
    out_dir: str | Path,
    *,
    topk: int = 20,
    include_buffers: bool = False,
) -> dict[str, Path]:
    out = ensure_dir(Path(out_dir) / "figures")
    exported: dict[str, Path] = {}
    for name, fig in create_overview_figures(
        summary,
        records,
        topk=topk,
        include_buffers=include_buffers,
    ).items():
        path = out / f"{name}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        exported[name] = path
    return exported


def export_markdown_report(
    summary: CheckpointSummary,
    records: list[TensorRecord],
    out_dir: str | Path,
    *,
    report_level: str = "standard",
) -> Path:
    out = ensure_dir(out_dir)
    quant_records = [record for record in records if record.include_in_quant_analysis]
    top_risk = sorted(quant_records, key=lambda item: item.quant_risk_score or 0.0, reverse=True)[:10]
    channel_gain = sorted(
        [record for record in quant_records if record.sqnr_gain is not None],
        key=lambda item: item.sqnr_gain or 0.0,
        reverse=True,
    )[:10]
    bn_records = sorted(
        [record for record in records if record.bn_risk_score is not None],
        key=lambda item: item.bn_risk_score or 0.0,
        reverse=True,
    )[:10]

    def table(records_for_table: list[TensorRecord], fields: list[str]) -> str:
        if not records_for_table:
            return "No records.\n"
        header = "| " + " | ".join(fields) + " |\n"
        sep = "| " + " | ".join("---" for _ in fields) + " |\n"
        rows = []
        for record in records_for_table:
            rows.append(
                "| "
                + " | ".join(str(getattr(record, field, "")) for field in fields)
                + " |"
            )
        return header + sep + "\n".join(rows) + "\n"

    content = [
        "# Ckpt_viewer Analysis Report",
        "",
        "## 1. Executive Summary",
        summary.executive_summary,
        "",
        "## 2. File Information",
        f"- File: {summary.file_name}",
        f"- Size MB: {summary.file_size_mb}",
        f"- SHA256: {summary.sha256}",
        f"- Load mode: {summary.load_mode}",
        "",
        "## 3. Checkpoint Structure",
        f"- State dict key: {summary.state_dict_key}",
        f"- Top-level keys: {', '.join(summary.top_level_keys) if summary.top_level_keys else 'N/A'}",
        "",
        "## 4. Parameter Overview",
        f"- Total tensors: {summary.total_tensors}",
        f"- Weight-like tensors: {summary.weight_like_tensors}",
        f"- Buffer tensors: {summary.buffer_tensors}",
        f"- Quantizable tensors: {summary.quantizable_tensors}",
        f"- Total params: {summary.total_params}",
        f"- Trainable-like params: {summary.trainable_like_params}",
        f"- Health score: {summary.weight_health_score}",
        "",
        "## 5. Quantization Risk Summary",
        f"- Checkpoint quant risk: {summary.quantization_risk}",
        f"- BN / Norm risk: {summary.bn_norm_risk}",
        "",
        "## 6. Top Risk Layers",
        table(top_risk, ["name", "role", "type_guess", "quant_risk_score", "recommendation"]),
        "## 7. Per-channel Quantization Benefit",
        table(channel_gain, ["name", "sqnr_gain", "mse_reduction", "recommendation"]),
        "## 8. BN / Norm Risk",
        table(bn_records, ["name", "role", "bn_risk_score", "recommendation"]),
        "## 9. Recommended Deployment Strategy",
    ]
    for key, value in summary.recommended_strategy.items():
        content.append(f"- {key}: {value}")
    content.extend(
        [
            "",
            "## 10. Limitations",
            "本报告只基于 checkpoint 权重做静态分析，不执行 forward，不分析 activation 分布，"
            "不能替代 calibration data、验证集指标和部署后 engine 验证。",
            "",
        ]
    )
    if report_level.lower() == "detailed":
        content.extend(
            [
                "## Appendix. Warnings",
                "\n".join(f"- {warning}" for warning in summary.warnings),
                "",
            ]
        )
    path = out / "report.md"
    path.write_text("\n".join(content), encoding="utf-8")
    return path


def export_html_report(
    summary: CheckpointSummary,
    records: Iterable[TensorRecord],
    figures: dict[str, Path] | None,
    out_dir: str | Path,
) -> Path:
    out = ensure_dir(out_dir)
    df = _records_df(records)
    top_df = df[df["include_in_quant_analysis"]].head(50) if not df.empty else df
    figure_links = figures or {}
    warning_items = "\n".join(
        f"<li>{html.escape(warning)}</li>" for warning in summary.warnings
    )
    link_items = "\n".join(
        f'<li><a href="figures/{html.escape(path.name)}">{html.escape(name)}</a></li>'
        for name, path in figure_links.items()
    )
    strategy_items = "\n".join(
        f"<li><strong>{html.escape(str(key))}</strong>: {html.escape(str(value))}</li>"
        for key, value in summary.recommended_strategy.items()
    )
    table_html = top_df.to_html(index=False, escape=True) if not top_df.empty else "<p>No tensors.</p>"

    body = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Ckpt_viewer V2 Report - {html.escape(summary.file_name)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2937; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 6px 8px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
    .metric {{ border: 1px solid #d1d5db; border-radius: 6px; padding: 12px; }}
    .metric strong {{ display: block; font-size: 12px; color: #6b7280; }}
  </style>
</head>
<body>
  <h1>Ckpt_viewer V2 诊断报告</h1>
  <h2>Executive Summary</h2>
  <p>{html.escape(summary.executive_summary)}</p>
  <div class="grid">
    <div class="metric"><strong>File</strong>{html.escape(summary.file_name)}</div>
    <div class="metric"><strong>Load Mode</strong>{html.escape(summary.load_mode)}</div>
    <div class="metric"><strong>Total Tensors</strong>{summary.total_tensors}</div>
    <div class="metric"><strong>Weight-like</strong>{summary.weight_like_tensors}</div>
    <div class="metric"><strong>Buffers</strong>{summary.buffer_tensors}</div>
    <div class="metric"><strong>Quantizable</strong>{summary.quantizable_tensors}</div>
    <div class="metric"><strong>Health Score</strong>{summary.weight_health_score}</div>
    <div class="metric"><strong>Quant Risk</strong>{html.escape(summary.quantization_risk)}</div>
  </div>
  <h2>Recommended Deployment Strategy</h2>
  <ul>{strategy_items}</ul>
  <h2>Warnings</h2>
  <ul>{warning_items}</ul>
  <h2>Figures</h2>
  <ul>{link_items}</ul>
  <h2>Top Quantization Risk Layers</h2>
  {table_html}
</body>
</html>
"""
    path = out / "report.html"
    path.write_text(body, encoding="utf-8")
    return path


def export_report_bundle(
    summary: CheckpointSummary,
    records: list[TensorRecord],
    out_dir: str | Path,
    *,
    topk: int = 20,
    formats: set[str] | None = None,
    include_buffers: bool = False,
    report_level: str = "standard",
) -> dict[str, Path]:
    formats = formats or {"csv", "json", "md", "html"}
    out = ensure_dir(out_dir)
    exported: dict[str, Path] = {}

    if "json" in formats:
        exported["summary_json"] = export_summary_json(summary, out)
        exported["warnings_json"] = export_warnings_json(summary, out)
        exported["recommendations_json"] = export_recommendations_json(summary, out)
    if "csv" in formats:
        exported["tensor_stats_csv"] = export_tensor_stats_csv(records, out)
        exported["quant_stats_csv"] = export_quant_stats_csv(records, out)
        exported["bn_norm_stats_csv"] = export_bn_norm_stats_csv(records, out)
    if "md" in formats:
        exported["markdown_report"] = export_markdown_report(
            summary,
            records,
            out,
            report_level=report_level,
        )
    if "html" in formats:
        figures = export_figures(
            summary,
            records,
            out,
            topk=topk,
            include_buffers=include_buffers,
        )
        exported.update({f"figure_{key}": value for key, value in figures.items()})
        exported["html_report"] = export_html_report(summary, records, figures, out)

    return exported
