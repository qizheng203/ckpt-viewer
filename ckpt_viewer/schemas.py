"""Shared dataclasses used by the analyzer, CLI, UI, and reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


TensorRole = Literal[
    "weight",
    "bias",
    "norm_weight",
    "norm_bias",
    "running_mean",
    "running_var",
    "num_batches_tracked",
    "buffer",
    "optimizer_state",
    "unknown",
]


@dataclass
class TensorRecord:
    # identity
    name: str
    clean_name: str
    shape: tuple[int, ...]
    shape_str: str
    dtype: str
    numel: int
    size_mb: float

    # semantic classification
    group: str
    type_guess: str
    role: str
    is_float: bool
    is_integer: bool
    is_weight_like: bool
    is_buffer: bool
    is_norm_stat: bool
    is_quantizable_weight: bool
    include_in_weight_health: bool
    include_in_quant_analysis: bool
    include_in_bn_risk: bool

    # basic stats
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None
    abs_mean: float | None = None
    abs_max: float | None = None
    p01: float | None = None
    p05: float | None = None
    p50: float | None = None
    p95: float | None = None
    p99: float | None = None
    p999: float | None = None
    abs_p99: float | None = None
    abs_p999: float | None = None

    # health stats
    zero_ratio: float | None = None
    near_zero_ratio: float | None = None
    near_zero_ratio_1e_6: float | None = None
    near_zero_ratio_1e_5: float | None = None
    near_zero_ratio_1e_4: float | None = None
    nan_count: int = 0
    inf_count: int = 0
    finite_ratio: float | None = None

    # outlier stats
    absmax_p99_ratio: float | None = None
    absmax_p999_ratio: float | None = None
    outlier_ratio_3sigma: float | None = None
    outlier_ratio_6sigma: float | None = None

    # distribution stats
    skewness: float | None = None
    kurtosis: float | None = None
    positive_ratio: float | None = None
    negative_ratio: float | None = None

    # quant stats
    per_tensor_int8_mse: float | None = None
    per_tensor_int8_mae: float | None = None
    per_tensor_int8_max_error: float | None = None
    per_tensor_int8_sqnr: float | None = None
    per_tensor_int8_cosine: float | None = None

    per_channel_int8_mse: float | None = None
    per_channel_int8_mae: float | None = None
    per_channel_int8_max_error: float | None = None
    per_channel_int8_sqnr: float | None = None
    per_channel_int8_cosine: float | None = None

    mse_reduction: float | None = None
    sqnr_gain: float | None = None

    # channel stats
    channel_count: int | None = None
    channel_absmax_max: float | None = None
    channel_absmax_median: float | None = None
    channel_absmax_max_median_ratio: float | None = None
    channel_absmax_cv: float | None = None
    dead_channel_ratio: float | None = None

    # norm / BN stats
    bn_running_var_min: float | None = None
    bn_running_var_p001: float | None = None
    bn_running_var_p01: float | None = None
    bn_running_var_small_ratio: float | None = None
    bn_running_var_small_ratio_1e_6: float | None = None
    bn_running_var_small_ratio_1e_5: float | None = None
    bn_running_var_small_ratio_1e_4: float | None = None
    bn_fold_risk_score: float | None = None

    # risk
    weight_health_score: float | None = None
    quant_risk_score: float | None = None
    quant_risk_level: str = "N/A"
    bn_risk_score: float | None = None
    bn_risk_level: str = "N/A"
    fp16_fallback_candidate: bool = False
    fallback_reason: str = ""
    recommendation: str = "该 tensor 不是可量化浮点权重，因此跳过 INT8 fake quant 分析。"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CheckpointSummary:
    file_name: str
    file_size_mb: float
    sha256: str
    load_mode: str
    top_level_keys: list[str]
    state_dict_key: str | None

    total_tensors: int
    weight_like_tensors: int
    buffer_tensors: int
    quantizable_tensors: int
    total_params: int
    trainable_like_params: int
    fp32_size_mb: float
    fp16_size_mb: float
    int8_size_mb: float

    dtype_counts: dict[str, int]
    dtype_tensor_counts: dict[str, int]
    dtype_param_counts: dict[str, int]
    role_counts: dict[str, int]
    group_param_counts: dict[str, int]
    group_tensor_counts: dict[str, int]
    group_param_percent: dict[str, float]

    nan_tensor_count: int
    inf_tensor_count: int

    weight_health_score: float
    quantization_risk: str
    bn_norm_risk: str
    warnings: list[str] = field(default_factory=list)
    top_risk_layers: list[str] = field(default_factory=list)
    fallback_candidates: list[dict[str, str]] = field(default_factory=list)
    executive_summary: str = ""
    recommended_strategy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisResult:
    summary: CheckpointSummary
    records: list[TensorRecord]
    tensors: dict[str, Any]
    source_path: str
