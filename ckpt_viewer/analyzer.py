"""End-to-end checkpoint analysis orchestration."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import torch

from ckpt_viewer.checkpoint_parser import find_state_dict
from ckpt_viewer.loaders import get_file_info, load_weight_file
from ckpt_viewer.quant_analysis import (
    aggregate_bn_risk,
    aggregate_checkpoint_quant_risk,
    apply_quant_analysis,
    generate_overview_diagnosis,
    generate_quant_strategy,
)
from ckpt_viewer.schemas import AnalysisResult, CheckpointSummary, TensorRecord
from ckpt_viewer.tensor_stats import compute_tensor_stats


ProgressCallback = Callable[[str, float | None], None]


def _noop_progress(message: str, fraction: float | None = None) -> None:
    del message, fraction


def _risk_sort_value(record: TensorRecord) -> float:
    return float(record.quant_risk_score or record.bn_risk_score or 0.0)


def analyze_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    near_zero_threshold: float = 1e-6,
    progress_callback: ProgressCallback | None = None,
) -> list[TensorRecord]:
    progress = progress_callback or _noop_progress
    records: list[TensorRecord] = []
    total = max(len(state_dict), 1)

    for index, (name, tensor) in enumerate(state_dict.items(), start=1):
        progress(f"正在统计 tensor {index} / {total}", index / total)
        record = compute_tensor_stats(name, tensor, near_zero_threshold=near_zero_threshold)
        record = apply_quant_analysis(record, tensor)
        records.append(record)
        del tensor

    records.sort(key=_risk_sort_value, reverse=True)
    return records


def _checkpoint_health_score(records: list[TensorRecord]) -> float:
    scores = [
        record.weight_health_score
        for record in records
        if record.include_in_weight_health and record.weight_health_score is not None
    ]
    if not scores:
        return 100.0
    worst = min(scores)
    avg_top_bad = sum(sorted(scores)[: min(len(scores), 10)]) / max(min(len(scores), 10), 1)
    return round(min(worst, avg_top_bad), 2)


def build_summary(
    *,
    file_info: dict,
    load_mode: str,
    top_level_keys: list[str],
    state_dict_key: str | None,
    records: list[TensorRecord],
) -> CheckpointSummary:
    total_params = sum(record.numel for record in records)
    trainable_like_params = sum(record.numel for record in records if record.is_weight_like)
    dtype_tensor_counts = Counter(record.dtype for record in records)
    dtype_param_counts: defaultdict[str, int] = defaultdict(int)
    role_counts = Counter(record.role for record in records)
    group_param_counts: defaultdict[str, int] = defaultdict(int)
    group_tensor_counts: defaultdict[str, int] = defaultdict(int)

    for record in records:
        dtype_param_counts[record.dtype] += record.numel
        group_param_counts[record.group] += record.numel
        group_tensor_counts[record.group] += 1

    group_param_percent = {
        group: round(params / total_params * 100.0, 4) if total_params else 0.0
        for group, params in group_param_counts.items()
    }

    nan_tensor_count = sum(
        1 for record in records if record.include_in_weight_health and record.nan_count > 0
    )
    inf_tensor_count = sum(
        1 for record in records if record.include_in_weight_health and record.inf_count > 0
    )
    quant_records = [record for record in records if record.include_in_quant_analysis]
    bn_records = [record for record in records if record.include_in_bn_risk]

    quantization_risk = aggregate_checkpoint_quant_risk(records)
    bn_norm_risk = aggregate_bn_risk(records)
    weight_health_score = _checkpoint_health_score(records)

    warnings: list[str] = []
    if nan_tensor_count:
        warnings.append(f"检测到 {nan_tensor_count} 个参与权重健康评估的 tensor 包含 NaN。")
    if inf_tensor_count:
        warnings.append(f"检测到 {inf_tensor_count} 个参与权重健康评估的 tensor 包含 Inf。")
    high_risk = [
        record.name
        for record in quant_records
        if record.quant_risk_score is not None and record.quant_risk_score >= 60
    ]
    if high_risk:
        warnings.append(f"检测到 {len(high_risk)} 个 High 量化风险权重层。")
    high_bn = [
        record.name
        for record in bn_records
        if record.bn_risk_score is not None and record.bn_risk_score >= 60
    ]
    if high_bn:
        warnings.append(f"检测到 {len(high_bn)} 个 High BN/Norm folding 风险 tensor。")
    warnings.append("量化风险仅基于权重静态分析，不代表最终任务精度。")

    top_risk_layers = [
        record.name
        for record in sorted(quant_records, key=lambda item: item.quant_risk_score or 0.0, reverse=True)[
            :10
        ]
    ]
    fallback_candidates = [
        {"name": record.name, "reason": record.fallback_reason or record.recommendation}
        for record in quant_records
        if record.fp16_fallback_candidate
    ][:10]

    summary = CheckpointSummary(
        file_name=file_info["name"],
        file_size_mb=round(file_info["size_mb"], 4),
        sha256=file_info["sha256"],
        load_mode=load_mode,
        top_level_keys=top_level_keys,
        state_dict_key=state_dict_key,
        total_tensors=len(records),
        weight_like_tensors=sum(1 for record in records if record.is_weight_like),
        buffer_tensors=sum(1 for record in records if record.is_buffer),
        quantizable_tensors=len(quant_records),
        total_params=int(total_params),
        trainable_like_params=int(trainable_like_params),
        fp32_size_mb=round(total_params * 4 / (1024 * 1024), 4),
        fp16_size_mb=round(total_params * 2 / (1024 * 1024), 4),
        int8_size_mb=round(total_params / (1024 * 1024), 4),
        dtype_counts=dict(dtype_tensor_counts),
        dtype_tensor_counts=dict(dtype_tensor_counts),
        dtype_param_counts=dict(dtype_param_counts),
        role_counts=dict(role_counts),
        group_param_counts=dict(group_param_counts),
        group_tensor_counts=dict(group_tensor_counts),
        group_param_percent=group_param_percent,
        nan_tensor_count=nan_tensor_count,
        inf_tensor_count=inf_tensor_count,
        weight_health_score=weight_health_score,
        quantization_risk=quantization_risk,
        bn_norm_risk=bn_norm_risk,
        warnings=warnings,
        top_risk_layers=top_risk_layers,
        fallback_candidates=fallback_candidates,
    )
    summary.recommended_strategy = generate_quant_strategy(summary, records)
    summary.executive_summary = generate_overview_diagnosis(summary, records)
    return summary


def analyze_checkpoint(
    path: str | Path,
    *,
    trusted: bool = False,
    near_zero_threshold: float = 1e-6,
    progress_callback: ProgressCallback | None = None,
) -> AnalysisResult:
    progress = progress_callback or _noop_progress
    file_path = Path(path)

    progress("正在读取文件信息", 0.02)
    file_info = get_file_info(file_path)

    progress("正在加载文件", 0.08)
    obj, load_mode = load_weight_file(file_path, trusted=trusted)

    top_level_keys = [str(key) for key in obj.keys()] if isinstance(obj, dict) else []

    progress("正在解析 state_dict", 0.15)
    match = find_state_dict(obj)

    records = analyze_state_dict(
        match.state_dict,
        near_zero_threshold=near_zero_threshold,
        progress_callback=lambda message, fraction: progress(
            message, 0.15 + (fraction or 0.0) * 0.75
        ),
    )

    progress("正在生成总览", 0.95)
    summary = build_summary(
        file_info=file_info,
        load_mode=load_mode,
        top_level_keys=top_level_keys,
        state_dict_key=match.key,
        records=records,
    )

    progress("分析完成", 1.0)
    return AnalysisResult(
        summary=summary,
        records=records,
        tensors=match.state_dict,
        source_path=str(file_path),
    )
