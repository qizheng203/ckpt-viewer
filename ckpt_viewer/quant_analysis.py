"""Static INT8 fake-quantization, risk scoring, and diagnosis generation."""

from __future__ import annotations

from typing import Any

import torch

from ckpt_viewer.schemas import CheckpointSummary, TensorRecord


EPS = 1e-12
QMAX = 127.0
SENSITIVE_KEYWORDS = ("head", "output", "classifier", "regressor", "disp", "depth", "flow")


def _is_supported_float_tensor(x: torch.Tensor) -> bool:
    return x.is_floating_point() and not x.is_complex() and x.numel() > 0


def _finite_float(x: torch.Tensor) -> torch.Tensor | None:
    if not _is_supported_float_tensor(x):
        return None
    y = x.detach().cpu().to(torch.float32)
    if not torch.isfinite(y).all():
        return None
    return y


def fake_quant_symmetric_per_tensor(x: torch.Tensor) -> torch.Tensor:
    y = x.detach().cpu().to(torch.float32)
    abs_max = torch.max(torch.abs(y))
    if float(abs_max.item()) <= EPS:
        return torch.zeros_like(y)
    scale = abs_max / QMAX
    q = torch.round(y / scale).clamp(-QMAX, QMAX)
    return q * scale


def fake_quant_symmetric_per_channel(x: torch.Tensor, axis: int = 0) -> torch.Tensor:
    y = x.detach().cpu().to(torch.float32)
    if y.ndim not in (2, 4):
        raise ValueError("per-channel INT8 only supports 2D or 4D weights in v2.")
    reduce_dims = tuple(dim for dim in range(y.ndim) if dim != axis)
    abs_max = torch.amax(torch.abs(y), dim=reduce_dims, keepdim=True)
    scale = torch.clamp(abs_max / QMAX, min=EPS)
    q = torch.round(y / scale).clamp(-QMAX, QMAX)
    return q * scale


def compute_mse_mae_maxerror(x: torch.Tensor, x_hat: torch.Tensor) -> tuple[float, float, float]:
    error = x.to(torch.float32) - x_hat.to(torch.float32)
    abs_error = torch.abs(error)
    return (
        float(torch.mean(error**2).item()),
        float(torch.mean(abs_error).item()),
        float(torch.max(abs_error).item()),
    )


def compute_sqnr(x: torch.Tensor, x_hat: torch.Tensor, eps: float = EPS) -> float:
    x = x.to(torch.float32)
    x_hat = x_hat.to(torch.float32)
    signal = torch.mean(x**2)
    noise = torch.mean((x - x_hat) ** 2)
    signal_value = float(signal.item())
    noise_value = float(noise.item())
    if noise_value <= eps:
        return 100.0
    if signal_value <= eps:
        return 0.0
    return float((10.0 * torch.log10(signal / (noise + eps))).item())


def compute_cosine(x: torch.Tensor, x_hat: torch.Tensor, eps: float = EPS) -> float:
    a = x.to(torch.float32).flatten()
    b = x_hat.to(torch.float32).flatten()
    norm_a = torch.norm(a)
    norm_b = torch.norm(b)
    if float(norm_a.item()) <= eps and float(norm_b.item()) <= eps:
        return 1.0
    denominator = norm_a * norm_b + eps
    return float((torch.dot(a, b) / denominator).item())


def compute_channel_arrays(x: torch.Tensor) -> dict[str, torch.Tensor]:
    y = x.detach().cpu().to(torch.float32)
    if y.ndim not in (2, 4):
        return {}
    flat = y.reshape(y.shape[0], -1)
    channel_absmax = torch.amax(torch.abs(flat), dim=1)
    return {
        "channel_absmax": channel_absmax,
        "channel_l2norm": torch.linalg.vector_norm(flat, ord=2, dim=1),
        "channel_mean": torch.mean(flat, dim=1),
        "channel_std": torch.std(flat, dim=1, unbiased=False),
        "channel_zero_ratio": torch.mean((flat == 0).to(torch.float32), dim=1),
        "channel_scale": torch.clamp(channel_absmax / QMAX, min=EPS),
    }


def compute_channel_quant_mse(x: torch.Tensor) -> torch.Tensor:
    y = x.detach().cpu().to(torch.float32)
    if y.ndim not in (2, 4):
        return torch.empty(0)
    q = fake_quant_symmetric_per_channel(y, axis=0)
    flat_error = (y - q).reshape(y.shape[0], -1)
    return torch.mean(flat_error**2, dim=1)


def compute_channel_metrics(x: torch.Tensor) -> dict[str, float | None]:
    arrays = compute_channel_arrays(x)
    if not arrays:
        return {
            "channel_count": None,
            "channel_absmax_max": None,
            "channel_absmax_median": None,
            "channel_absmax_max_median_ratio": None,
            "channel_absmax_cv": None,
            "dead_channel_ratio": None,
        }

    channel_absmax = arrays["channel_absmax"]
    channel_l2norm = arrays["channel_l2norm"]
    finite_absmax = channel_absmax[torch.isfinite(channel_absmax)]
    if finite_absmax.numel() == 0:
        return {
            "channel_count": int(channel_absmax.numel()),
            "channel_absmax_max": None,
            "channel_absmax_median": None,
            "channel_absmax_max_median_ratio": None,
            "channel_absmax_cv": None,
            "dead_channel_ratio": None,
        }

    max_abs = torch.max(finite_absmax)
    median_abs = torch.median(finite_absmax)
    mean_abs = torch.mean(finite_absmax)
    std_abs = torch.std(finite_absmax, unbiased=False)
    dead = torch.mean((channel_l2norm <= EPS).to(torch.float32))

    return {
        "channel_count": int(channel_absmax.numel()),
        "channel_absmax_max": float(max_abs.item()),
        "channel_absmax_median": float(median_abs.item()),
        "channel_absmax_max_median_ratio": float(max_abs.item() / (median_abs.item() + EPS)),
        "channel_absmax_cv": float(std_abs.item() / (mean_abs.item() + EPS)),
        "dead_channel_ratio": float(dead.item()),
    }


def compute_weight_health_score(record: TensorRecord) -> float | None:
    if not record.include_in_weight_health:
        return None

    penalty = 0.0
    if record.nan_count > 0 or record.inf_count > 0:
        penalty += 70
    if record.abs_max is not None and record.abs_max > 1e6:
        penalty += 20
    if record.absmax_p999_ratio is not None:
        if record.absmax_p999_ratio > 20:
            penalty += 20
        elif record.absmax_p999_ratio > 8:
            penalty += 12
        elif record.absmax_p999_ratio > 5:
            penalty += 8
    if record.kurtosis is not None and record.kurtosis > 50:
        penalty += 8
    if record.finite_ratio is not None and record.finite_ratio < 1.0:
        penalty += min(20.0, (1.0 - record.finite_ratio) * 100.0)
    return max(0.0, min(100.0, 100.0 - penalty))


def compute_quant_risk_score(record: TensorRecord) -> float | None:
    if not record.include_in_quant_analysis:
        return None
    if record.nan_count > 0 or record.inf_count > 0:
        return 100.0

    risk = 0.0

    if record.absmax_p999_ratio is not None:
        if record.absmax_p999_ratio > 8:
            risk += 25
        elif record.absmax_p999_ratio > 5:
            risk += 20
        elif record.absmax_p999_ratio > 2:
            risk += 10

    if record.per_tensor_int8_sqnr is not None:
        if record.per_tensor_int8_sqnr < 20:
            risk += 25
        elif record.per_tensor_int8_sqnr < 30:
            risk += 18
        elif record.per_tensor_int8_sqnr < 35:
            risk += 10

    if record.per_channel_int8_sqnr is not None:
        if record.per_channel_int8_sqnr < 25:
            risk += 15
        elif record.per_channel_int8_sqnr < 30:
            risk += 8

    if record.channel_absmax_max_median_ratio is not None:
        if record.channel_absmax_max_median_ratio > 15:
            risk += 20
        elif record.channel_absmax_max_median_ratio > 10:
            risk += 15
        elif record.channel_absmax_max_median_ratio > 5:
            risk += 8

    if any(keyword in record.name.lower() for keyword in SENSITIVE_KEYWORDS) and risk >= 30:
        risk += 10

    return min(risk, 100.0)


def compute_bn_risk_score(record: TensorRecord) -> float | None:
    if not record.include_in_bn_risk:
        return None

    risk = 0.0
    if record.role == "running_var":
        if record.bn_running_var_min is not None:
            if record.bn_running_var_min < 1e-6:
                risk += 45
            elif record.bn_running_var_min < 1e-5:
                risk += 30
            elif record.bn_running_var_min < 1e-4:
                risk += 15
        if record.bn_running_var_p01 is not None:
            if record.bn_running_var_p01 < 1e-6:
                risk += 35
            elif record.bn_running_var_p01 < 1e-5:
                risk += 25
            elif record.bn_running_var_p01 < 1e-4:
                risk += 10
        if record.bn_running_var_small_ratio_1e_5 is not None:
            if record.bn_running_var_small_ratio_1e_5 > 0.05:
                risk += 20
            elif record.bn_running_var_small_ratio_1e_5 > 0.01:
                risk += 10
    elif record.role == "norm_weight":
        if record.abs_max is not None and record.abs_max > 10:
            risk += 35
        if record.absmax_p999_ratio is not None:
            if record.absmax_p999_ratio > 8:
                risk += 30
            elif record.absmax_p999_ratio > 5:
                risk += 20

    return min(risk, 100.0)


def risk_level_from_score(score: float | None) -> str:
    if score is None:
        return "N/A"
    if score < 30:
        return "Low"
    if score < 60:
        return "Medium"
    return "High"


def compute_fallback(record: TensorRecord) -> tuple[bool, str]:
    if record.nan_count > 0 or record.inf_count > 0:
        return False, "invalid: tensor contains NaN/Inf and must be fixed first"
    if record.quant_risk_score is None:
        return False, ""
    if record.quant_risk_score >= 50 and (
        record.per_channel_int8_sqnr is None or record.per_channel_int8_sqnr < 35
    ):
        return True, "quant_risk_score >= 50 and per-channel SQNR is still below 35 dB"
    if any(keyword in record.name.lower() for keyword in SENSITIVE_KEYWORDS) and record.quant_risk_score >= 40:
        return True, "sensitive output-related layer with quant_risk_score >= 40"
    if (
        record.per_tensor_int8_sqnr is not None
        and record.per_tensor_int8_sqnr < 25
        and (record.sqnr_gain is None or record.sqnr_gain < 3)
    ):
        return True, "per-tensor SQNR < 25 dB and per-channel gain is limited"
    return False, ""


def generate_recommendation(record: TensorRecord) -> str:
    if record.nan_count > 0 or record.inf_count > 0:
        return "权重存在 NaN/Inf，需先修复 checkpoint。"

    if record.role == "running_var" and record.bn_risk_score is not None and record.bn_risk_score >= 30:
        return "running_var 存在较小值，Conv-BN folding 可能放大权重，需要在导出后检查部署图。"

    if record.role == "norm_weight" and record.bn_risk_score is not None and record.bn_risk_score >= 30:
        return "norm gamma 范围或离群值偏大，folding 后可能放大权重，部署时建议重点检查。"

    if not record.include_in_quant_analysis:
        return "该 tensor 不是可量化浮点权重，因此跳过 INT8 fake quant 分析。"

    if (
        record.per_tensor_int8_sqnr is not None
        and record.per_tensor_int8_sqnr < 30
        and record.per_channel_int8_sqnr is not None
        and record.sqnr_gain is not None
        and record.sqnr_gain > 5
    ):
        return "建议使用 per-channel INT8 weight quantization。"

    if record.absmax_p999_ratio is not None and record.absmax_p999_ratio > 5:
        return "存在明显离群权重，MinMax per-tensor INT8 可能受影响。"

    if record.per_channel_int8_sqnr is not None and record.per_channel_int8_sqnr < 30:
        return "per-channel INT8 仍存在较大量化误差，建议部署时重点验证，必要时 FP16 fallback。"

    if record.fp16_fallback_candidate:
        return "该层靠近输出或量化风险较高，建议优先纳入 FP16 fallback 候选。"

    if record.mse_reduction is not None and record.mse_reduction > 0.3:
        return "per-channel 相比 per-tensor 误差明显降低，建议优先使用 per-channel 权重量化。"

    return "静态权重量化风险较低。"


def apply_quant_analysis(record: TensorRecord, tensor: torch.Tensor) -> TensorRecord:
    y = _finite_float(tensor)
    record.weight_health_score = compute_weight_health_score(record)

    if y is not None and y.ndim in (2, 4):
        channel_metrics = compute_channel_metrics(y)
        for key, value in channel_metrics.items():
            setattr(record, key, value)

    if y is not None and record.include_in_quant_analysis:
        x_hat_tensor = fake_quant_symmetric_per_tensor(y)
        mse, mae, max_error = compute_mse_mae_maxerror(y, x_hat_tensor)
        record.per_tensor_int8_mse = mse
        record.per_tensor_int8_mae = mae
        record.per_tensor_int8_max_error = max_error
        record.per_tensor_int8_sqnr = compute_sqnr(y, x_hat_tensor)
        record.per_tensor_int8_cosine = compute_cosine(y, x_hat_tensor)

        if y.ndim in (2, 4):
            x_hat_channel = fake_quant_symmetric_per_channel(y, axis=0)
            mse_c, mae_c, max_error_c = compute_mse_mae_maxerror(y, x_hat_channel)
            record.per_channel_int8_mse = mse_c
            record.per_channel_int8_mae = mae_c
            record.per_channel_int8_max_error = max_error_c
            record.per_channel_int8_sqnr = compute_sqnr(y, x_hat_channel)
            record.per_channel_int8_cosine = compute_cosine(y, x_hat_channel)
            if record.per_tensor_int8_mse is not None and record.per_tensor_int8_mse > EPS:
                record.mse_reduction = float(1 - mse_c / record.per_tensor_int8_mse)
            if record.per_tensor_int8_sqnr is not None:
                record.sqnr_gain = float(record.per_channel_int8_sqnr - record.per_tensor_int8_sqnr)

    record.quant_risk_score = compute_quant_risk_score(record)
    record.quant_risk_level = risk_level_from_score(record.quant_risk_score)
    record.bn_risk_score = compute_bn_risk_score(record)
    record.bn_fold_risk_score = record.bn_risk_score
    record.bn_risk_level = risk_level_from_score(record.bn_risk_score)
    record.fp16_fallback_candidate, record.fallback_reason = compute_fallback(record)
    record.recommendation = generate_recommendation(record)
    return record


def generate_quant_decision(record: TensorRecord) -> dict[str, str]:
    per_tensor = "N/A"
    if record.per_tensor_int8_sqnr is not None:
        per_tensor = "Not recommended" if record.per_tensor_int8_sqnr < 25 else "Usable with validation"

    per_channel = "N/A"
    if record.per_channel_int8_sqnr is not None:
        if record.per_channel_int8_sqnr >= 30 and (record.sqnr_gain or 0.0) > 5:
            per_channel = "Recommended"
        elif record.per_channel_int8_sqnr >= 30:
            per_channel = "Usable with validation"
        else:
            per_channel = "Risky"

    fallback = "Candidate if final accuracy drops" if (record.quant_risk_score or 0.0) >= 50 else "Not primary"
    qat = "Recommended if PTQ fails" if (record.quant_risk_score or 0.0) >= 70 else "Optional if PTQ fails"
    return {
        "Per-tensor INT8": per_tensor,
        "Per-channel INT8": per_channel,
        "FP16 fallback": fallback,
        "QAT": qat,
    }


def generate_layer_diagnosis(record: TensorRecord) -> str:
    if not record.include_in_quant_analysis and not record.include_in_bn_risk:
        return "该 tensor 不属于 v2 默认量化或 BN/Norm 风险分析对象。"
    parts: list[str] = []
    if record.channel_absmax_max_median_ratio is not None and record.channel_absmax_max_median_ratio > 5:
        parts.append(
            f"该层存在通道尺度不均衡，最大通道 absmax 约为中位通道的 "
            f"{record.channel_absmax_max_median_ratio:.2f} 倍。"
        )
    if record.mse_reduction is not None and record.sqnr_gain is not None:
        parts.append(
            f"per-channel INT8 将 MSE 降低约 {record.mse_reduction * 100:.2f}%，"
            f"SQNR 提升约 {record.sqnr_gain:.2f} dB。"
        )
    if record.per_tensor_int8_sqnr is not None and record.per_tensor_int8_sqnr < 25:
        parts.append("该层不适合简单 per-tensor INT8。")
    if record.fp16_fallback_candidate:
        parts.append("如果完整 INT8 精度下降，该层可作为 FP16 fallback 候选。")
    if record.bn_risk_score is not None and record.bn_risk_score >= 30:
        parts.append("该 BN/Norm 相关 tensor 在 folding 后可能带来数值放大风险。")
    if not parts:
        parts.append("该层静态权重指标未显示明显异常，仍需结合验证集确认最终部署精度。")
    return "".join(parts)


def aggregate_checkpoint_quant_risk(records: list[TensorRecord]) -> str:
    scores = [record.quant_risk_score for record in records if record.quant_risk_score is not None]
    if not scores:
        return "LOW"
    max_risk = max(scores)
    high_count = sum(score >= 60 for score in scores)
    medium_count = sum(30 <= score < 60 for score in scores)
    if max_risk >= 70 or high_count >= 3:
        return "HIGH"
    if max_risk >= 50 or medium_count >= 3:
        return "MEDIUM"
    return "LOW"


def aggregate_bn_risk(records: list[TensorRecord]) -> str:
    scores = [record.bn_risk_score for record in records if record.bn_risk_score is not None]
    if not scores:
        return "LOW"
    if max(scores) >= 60 or sum(score >= 60 for score in scores) >= 2:
        return "HIGH"
    if max(scores) >= 30 or sum(score >= 30 for score in scores) >= 2:
        return "MEDIUM"
    return "LOW"


def generate_quant_strategy(
    summary: CheckpointSummary | None,
    records: list[TensorRecord],
) -> dict[str, Any]:
    quantizable = [record for record in records if record.include_in_quant_analysis]
    fallback_candidates = [
        {"name": record.name, "reason": record.fallback_reason or record.recommendation}
        for record in quantizable
        if record.fp16_fallback_candidate
    ][:10]
    per_channel_benefit = [
        record
        for record in quantizable
        if (record.sqnr_gain is not None and record.sqnr_gain > 5)
        or (record.mse_reduction is not None and record.mse_reduction > 0.3)
    ]
    checkpoint_risk = summary.quantization_risk.upper() if summary else aggregate_checkpoint_quant_risk(records)
    recommended_weight_quantization = (
        "per-channel INT8"
        if per_channel_benefit or checkpoint_risk in {"MEDIUM", "HIGH"}
        else "per-tensor INT8 acceptable, validate before deployment"
    )
    return {
        "checkpoint_quant_risk": checkpoint_risk,
        "recommended_weight_quantization": recommended_weight_quantization,
        "activation_quantization": "not analyzed; requires calibration data",
        "fp16_baseline": "recommended",
        "fallback_candidates": fallback_candidates,
        "ptq_risk": checkpoint_risk.title(),
        "next_step": "validate FP16 baseline, then INT8 PTQ with per-channel weight quantization",
        "next_steps": [
            "Run FP16 baseline",
            "Run INT8 PTQ with per-channel weight quantization",
            "Validate on task-specific metrics",
            "Inspect high-risk layers if accuracy drops",
        ],
    }


def generate_overview_diagnosis(
    summary: CheckpointSummary,
    records: list[TensorRecord],
) -> str:
    quantizable = [record for record in records if record.include_in_quant_analysis]
    top = max(quantizable, key=lambda item: item.quant_risk_score or 0.0, default=None)
    group = top.group if top is not None else "N/A"
    layer = top.name if top is not None else "N/A"
    recommendation = top.recommendation if top is not None else "暂无可量化权重层。"
    return (
        f"该 checkpoint 通过 {summary.load_mode} 模式加载，解析到 {summary.total_tensors} 个 tensor，"
        f"其中可量化权重 {summary.quantizable_tensors} 个，总参数量约 {summary.total_params:,}。"
        f"当前静态量化风险为 {summary.quantization_risk.upper()}。"
        f"风险主要集中在 {group}，最高风险层为 {layer}。{recommendation}"
    )


def quant_fields(record: TensorRecord) -> dict[str, Any]:
    keys = [
        "name",
        "role",
        "type_guess",
        "group",
        "shape_str",
        "is_quantizable_weight",
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
        "channel_absmax_max_median_ratio",
        "quant_risk_score",
        "quant_risk_level",
        "fp16_fallback_candidate",
        "fallback_reason",
        "recommendation",
    ]
    data = record.to_dict()
    return {key: data.get(key) for key in keys}
