"""Tensor-level descriptive statistics and plotting value preparation."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from ckpt_viewer.grouping import guess_group, guess_tensor_type, semantic_flags
from ckpt_viewer.schemas import TensorRecord


EPS = 1e-12


def _as_stat_values(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().cpu()
    if tensor.is_complex():
        return tensor.abs().to(torch.float32).flatten()
    if tensor.dtype == torch.bool:
        return tensor.to(torch.float32).flatten()
    return tensor.to(torch.float32).flatten()


def shape_to_string(shape: tuple[int, ...]) -> str:
    return "[" + ", ".join(str(dim) for dim in shape) + "]"


def _empty_record(name: str, tensor: torch.Tensor, near_zero_threshold: float) -> TensorRecord:
    del near_zero_threshold
    shape = tuple(int(dim) for dim in tensor.shape)
    type_guess = guess_tensor_type(name, tensor)
    flags = semantic_flags(name, tensor, type_guess=type_guess)
    return TensorRecord(
        name=name,
        clean_name=name,
        shape=shape,
        shape_str=shape_to_string(shape),
        dtype=str(tensor.dtype).replace("torch.", ""),
        numel=int(tensor.numel()),
        size_mb=float(tensor.numel() * tensor.element_size() / (1024 * 1024)),
        group=guess_group(name),
        type_guess=type_guess,
        role=str(flags["role"]),
        is_float=bool(flags["is_float"]),
        is_integer=bool(flags["is_integer"]),
        is_weight_like=bool(flags["is_weight_like"]),
        is_buffer=bool(flags["is_buffer"]),
        is_norm_stat=bool(flags["is_norm_stat"]),
        is_quantizable_weight=bool(flags["is_quantizable_weight"]),
        include_in_weight_health=bool(flags["include_in_weight_health"]),
        include_in_quant_analysis=bool(flags["include_in_quant_analysis"]),
        include_in_bn_risk=bool(flags["include_in_bn_risk"]),
    )


def _percentile_name(percentile: float) -> str:
    if percentile == 0.999:
        return "p999"
    if percentile == 0.001:
        return "p001"
    return f"p{int(round(percentile * 100)):02d}"


def compute_percentiles(
    values: torch.Tensor,
    percentiles: Iterable[float] = (0.01, 0.05, 0.5, 0.95, 0.99, 0.999),
) -> dict[str, float | None]:
    percentiles = tuple(percentiles)
    if values.numel() == 0:
        return {_percentile_name(percentile): None for percentile in percentiles}

    q = torch.tensor(percentiles, dtype=torch.float32, device=values.device)
    result = torch.quantile(values.to(torch.float32), q)
    return {
        _percentile_name(percentile): float(value.item())
        for percentile, value in zip(percentiles, result)
    }


def compute_outlier_metrics(values: torch.Tensor) -> dict[str, float | None]:
    if values.numel() == 0:
        return {
            "outlier_ratio_3sigma": None,
            "outlier_ratio_6sigma": None,
        }
    mean = torch.mean(values)
    std = torch.std(values, unbiased=False)
    if float(std.item()) <= EPS:
        return {
            "outlier_ratio_3sigma": 0.0,
            "outlier_ratio_6sigma": 0.0,
        }
    centered = torch.abs(values - mean)
    return {
        "outlier_ratio_3sigma": float((centered > 3 * std).float().mean().item()),
        "outlier_ratio_6sigma": float((centered > 6 * std).float().mean().item()),
    }


def compute_distribution_metrics(values: torch.Tensor) -> dict[str, float | None]:
    if values.numel() == 0:
        return {
            "skewness": None,
            "kurtosis": None,
            "positive_ratio": None,
            "negative_ratio": None,
        }
    positive_ratio = float((values > 0).to(torch.float32).mean().item())
    negative_ratio = float((values < 0).to(torch.float32).mean().item())
    std = torch.std(values, unbiased=False)
    if float(std.item()) <= EPS:
        return {
            "skewness": 0.0,
            "kurtosis": 0.0,
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
        }
    centered = values - torch.mean(values)
    normalized = centered / (std + EPS)
    return {
        "skewness": float(torch.mean(normalized**3).item()),
        "kurtosis": float(torch.mean(normalized**4).item() - 3.0),
        "positive_ratio": positive_ratio,
        "negative_ratio": negative_ratio,
    }


def _apply_bn_running_var_stats(record: TensorRecord, finite: torch.Tensor) -> None:
    if record.role != "running_var" or finite.numel() == 0:
        return
    percentiles = compute_percentiles(finite, percentiles=(0.001, 0.01, 0.05, 0.5))
    record.bn_running_var_min = float(torch.min(finite).item())
    record.bn_running_var_p001 = percentiles.get("p001")
    record.bn_running_var_p01 = percentiles.get("p01")
    record.bn_running_var_small_ratio_1e_6 = float((finite < 1e-6).to(torch.float32).mean().item())
    record.bn_running_var_small_ratio_1e_5 = float((finite < 1e-5).to(torch.float32).mean().item())
    record.bn_running_var_small_ratio_1e_4 = float((finite < 1e-4).to(torch.float32).mean().item())
    record.bn_running_var_small_ratio = record.bn_running_var_small_ratio_1e_5


def compute_tensor_stats(
    name: str,
    tensor: torch.Tensor,
    near_zero_threshold: float = 1e-6,
) -> TensorRecord:
    record = _empty_record(name, tensor, near_zero_threshold)
    if record.numel == 0:
        record.recommendation = "空 tensor，未进行统计。"
        return record

    values = _as_stat_values(tensor)
    nan_mask = torch.isnan(values)
    inf_mask = torch.isinf(values)
    finite_mask = torch.isfinite(values)

    record.nan_count = int(nan_mask.sum().item())
    record.inf_count = int(inf_mask.sum().item())
    record.finite_ratio = float(finite_mask.to(torch.float32).mean().item())

    finite = values[finite_mask]
    if finite.numel() == 0:
        record.recommendation = "权重全部为 NaN/Inf，需先修复 checkpoint。"
        return record

    abs_finite = torch.abs(finite)
    record.min = float(torch.min(finite).item())
    record.max = float(torch.max(finite).item())
    record.mean = float(torch.mean(finite).item())
    record.std = float(torch.std(finite, unbiased=False).item())
    record.abs_mean = float(torch.mean(abs_finite).item())
    record.abs_max = float(torch.max(abs_finite).item())

    percentiles = compute_percentiles(finite)
    for key, value in percentiles.items():
        setattr(record, key, value)

    abs_percentiles = compute_percentiles(abs_finite, percentiles=(0.99, 0.999))
    record.abs_p99 = abs_percentiles.get("p99")
    record.abs_p999 = abs_percentiles.get("p999")

    total = float(record.numel)
    record.zero_ratio = float((values == 0).float().sum().item() / total)
    record.near_zero_ratio = float((abs_finite < near_zero_threshold).float().mean().item())
    record.near_zero_ratio_1e_6 = float((abs_finite < 1e-6).float().mean().item())
    record.near_zero_ratio_1e_5 = float((abs_finite < 1e-5).float().mean().item())
    record.near_zero_ratio_1e_4 = float((abs_finite < 1e-4).float().mean().item())

    if record.abs_p99 is not None:
        record.absmax_p99_ratio = float(record.abs_max / (record.abs_p99 + EPS))
    if record.abs_p999 is not None:
        record.absmax_p999_ratio = float(record.abs_max / (record.abs_p999 + EPS))

    outlier_metrics = compute_outlier_metrics(finite)
    record.outlier_ratio_3sigma = outlier_metrics["outlier_ratio_3sigma"]
    record.outlier_ratio_6sigma = outlier_metrics["outlier_ratio_6sigma"]

    distribution_metrics = compute_distribution_metrics(finite)
    record.skewness = distribution_metrics["skewness"]
    record.kurtosis = distribution_metrics["kurtosis"]
    record.positive_ratio = distribution_metrics["positive_ratio"]
    record.negative_ratio = distribution_metrics["negative_ratio"]

    _apply_bn_running_var_stats(record, finite)
    return record


def sample_tensor_for_plot(tensor: torch.Tensor, max_samples: int = 1_000_000) -> np.ndarray:
    values = _as_stat_values(tensor)
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return np.array([], dtype=np.float32)
    if finite.numel() <= max_samples:
        sampled = finite
    else:
        indices = torch.randint(0, finite.numel(), (max_samples,))
        sampled = finite[indices]
    return sampled.numpy()


def prepare_log_abs_values(values: np.ndarray, eps: float = EPS) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([], dtype=np.float64)
    return np.log10(np.abs(x) + eps)
