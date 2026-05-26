"""Small utility helpers shared across the project."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ckpt_viewer.schemas import CheckpointSummary, TensorRecord


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return result
    return result


def json_default(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def write_json(path: str | Path, data: Any) -> None:
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def records_to_dataframe(records: Iterable[TensorRecord]) -> pd.DataFrame:
    return pd.DataFrame([record.to_dict() for record in records])


def summary_to_dataframe(summary: CheckpointSummary) -> pd.DataFrame:
    return pd.DataFrame([summary.to_dict()])


def format_shape(shape: tuple[int, ...]) -> str:
    return "(" + ", ".join(str(dim) for dim in shape) + ")"


def ratio_to_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.4f}%"

