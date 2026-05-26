"""Command line interface for Ckpt_viewer."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from ckpt_viewer import __version__
from ckpt_viewer.analyzer import analyze_checkpoint
from ckpt_viewer.report import export_report_bundle
from ckpt_viewer.utils import records_to_dataframe


console = Console()
cli = typer.Typer(
    add_completion=False,
    invoke_without_command=True,
    no_args_is_help=False,
    help="Ckpt_viewer: local checkpoint weight inspector.",
)


def _launch_streamlit(host: str = "localhost", port: int = 8501) -> None:
    app_path = Path(__file__).with_name("app.py")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    try:
        subprocess.run(command, check=True)
    except ModuleNotFoundError as exc:
        raise typer.BadParameter("未安装 streamlit，请先执行 pip install -e .") from exc
    except subprocess.CalledProcessError as exc:
        raise typer.Exit(exc.returncode) from exc


@cli.callback()
def callback(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
) -> None:
    if version:
        console.print(f"Ckpt_viewer {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        _launch_streamlit()
        raise typer.Exit()


@cli.command()
def ui(
    host: str = typer.Option("localhost", "--host", help="Streamlit server host."),
    port: int = typer.Option(8501, "--port", help="Streamlit server port."),
) -> None:
    """Start the local Web UI."""

    _launch_streamlit(host=host, port=port)


@cli.command()
def analyze(
    weight_file: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False),
    out: Path = typer.Option(Path("report"), "--out", "-o", help="Report output directory."),
    trusted: bool = typer.Option(False, "--trusted", help="Allow unsafe torch.load fallback."),
    near_zero: float = typer.Option(1e-6, "--near-zero", help="Near-zero threshold."),
    sample_size: int = typer.Option(1_000_000, "--sample-size", help="Histogram sample size hint."),
    topk: int = typer.Option(20, "--topk", help="Top-K layers for report figures."),
    exclude_buffers: bool = typer.Option(True, "--exclude-buffers", help="Exclude buffers in report plots."),
    quantizable_only: bool = typer.Option(
        False,
        "--quantizable-only",
        help="Export only quantizable weight records in CSV/HTML tables.",
    ),
    filter_role: Optional[str] = typer.Option(
        None,
        "--filter-role",
        help="Comma-separated roles to keep, e.g. weight,bias,norm_weight.",
    ),
    report_level: str = typer.Option(
        "standard",
        "--report-level",
        help="basic / standard / detailed.",
    ),
    export: str = typer.Option(
        "csv,json,md,html",
        "--export",
        help="Comma-separated: csv,json,md,html.",
    ),
    csv_path: Optional[Path] = typer.Option(None, "--csv", help="Only export tensor stats CSV to this path."),
) -> None:
    """Analyze a single checkpoint and export reports."""
    del sample_size

    messages: list[str] = []

    def progress(message: str, fraction: float | None = None) -> None:
        if not messages or messages[-1] != message:
            messages.append(message)
            console.print(f"[dim]{message}[/dim]")

    try:
        result = analyze_checkpoint(
            weight_file,
            trusted=trusted,
            near_zero_threshold=near_zero,
            progress_callback=progress,
        )
    except Exception as exc:
        console.print(f"[red]分析失败：{exc}[/red]")
        raise typer.Exit(1) from exc

    exported_records = result.records
    if quantizable_only:
        exported_records = [record for record in exported_records if record.include_in_quant_analysis]
    if filter_role:
        roles = {role.strip() for role in filter_role.split(",") if role.strip()}
        exported_records = [record for record in exported_records if record.role in roles]

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        records_to_dataframe(exported_records).to_csv(csv_path, index=False)
        console.print(f"[green]CSV 已导出：{csv_path.resolve()}[/green]")
    else:
        formats = {item.strip().lower() for item in export.split(",") if item.strip()}
        invalid = formats - {"csv", "json", "md", "html"}
        if invalid:
            console.print(f"[red]不支持的导出格式：{', '.join(sorted(invalid))}[/red]")
            raise typer.Exit(1)
        if report_level.lower() not in {"basic", "standard", "detailed"}:
            console.print("[red]--report-level 只能是 basic / standard / detailed[/red]")
            raise typer.Exit(1)
        exported = export_report_bundle(
            result.summary,
            exported_records,
            out,
            topk=topk,
            formats=formats,
            include_buffers=not exclude_buffers,
            report_level=report_level,
        )
        for key, path in exported.items():
            console.print(f"[green]{key}: {path.resolve()}[/green]")

    table = Table(title="Checkpoint Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("File", result.summary.file_name)
    table.add_row("Load mode", result.summary.load_mode)
    table.add_row("State dict key", str(result.summary.state_dict_key))
    table.add_row("Tensors", f"{result.summary.total_tensors:,}")
    table.add_row("Weight-like", f"{result.summary.weight_like_tensors:,}")
    table.add_row("Buffers", f"{result.summary.buffer_tensors:,}")
    table.add_row("Quantizable", f"{result.summary.quantizable_tensors:,}")
    table.add_row("Params", f"{result.summary.total_params:,}")
    table.add_row("Health score", f"{result.summary.weight_health_score:.1f}")
    table.add_row("Quant risk", result.summary.quantization_risk)
    table.add_row("BN / Norm risk", result.summary.bn_norm_risk)
    console.print(table)

    top_records = sorted(
        [record for record in exported_records if record.quant_risk_score is not None],
        key=lambda record: record.quant_risk_score or 0.0,
        reverse=True,
    )[: min(topk, 10)]
    top_df = pd.DataFrame([record.to_dict() for record in top_records])
    if not top_df.empty:
        risk_table = Table(title="Top Risk Layers")
        for column in ["name", "role", "quant_risk_score", "quant_risk_level", "recommendation"]:
            risk_table.add_column(column)
        for _, row in top_df.iterrows():
            risk_table.add_row(
                str(row["name"]),
                str(row["role"]),
                f"{float(row['quant_risk_score']):.1f}" if pd.notna(row["quant_risk_score"]) else "N/A",
                str(row["quant_risk_level"]),
                str(row["recommendation"]),
            )
        console.print(risk_table)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
