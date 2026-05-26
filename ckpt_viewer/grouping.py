"""Heuristic tensor grouping, semantic role detection, and record filters."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from ckpt_viewer.schemas import TensorRecord


GROUP_KEYWORDS = [
    ("backbone", ("backbone",)),
    ("encoder", ("encoder", "enc.")),
    ("decoder", ("decoder", "dec.")),
    ("neck", ("neck", "fpn", "pafpn")),
    ("head", ("head", "output", "out_proj")),
    ("classifier", ("classifier", "cls", "fc")),
    ("regressor", ("regressor", "reg", "bbox")),
    ("attention", ("attention", "attn", "q_proj", "k_proj", "v_proj")),
    ("transformer", ("transformer", "self_attn", "mlp")),
    ("cost_volume", ("cost_volume", "cost.volume", "corr", "correlation")),
    ("update_block", ("update_block", "update.block", "update")),
    ("gru", ("gru",)),
    ("norm", ("bn", "norm", "running_mean", "running_var", "gamma", "beta")),
]

VISION_KEYWORDS = (
    "conv",
    "bn",
    "norm",
    "backbone",
    "encoder",
    "decoder",
    "neck",
    "head",
    "feature",
    "corr",
    "cost",
    "volume",
    "attention",
    "gru",
    "update",
    "disp",
    "flow",
    "depth",
)

NORM_KEYWORDS = ("bn", "norm", "ln", "gn", "batchnorm", "layernorm", "groupnorm")


def guess_tensor_type(name: str, tensor: torch.Tensor) -> str:
    lower = name.lower()

    if tensor.ndim == 4:
        return "conv_weight"

    if tensor.ndim == 2:
        if "embed" in lower or "embedding" in lower:
            return "embedding"
        return "linear_weight"

    if tensor.ndim == 1:
        if lower.endswith(".bias") or "bias" in lower:
            return "bias"

        if any(
            keyword in lower
            for keyword in ["bn", "norm", "running_mean", "running_var", "gamma", "beta"]
        ):
            return "norm_param"

        return "vector_param"

    return "other"


def guess_group(name: str) -> str:
    lower = name.lower()
    for group, keywords in GROUP_KEYWORDS:
        if any(keyword in lower for keyword in keywords):
            return group
    if any(keyword in lower for keyword in VISION_KEYWORDS):
        return "other"
    return "other"


def guess_tensor_role(name: str, tensor: torch.Tensor) -> str:
    lower = name.lower()

    if lower.endswith("num_batches_tracked") or "num_batches_tracked" in lower:
        return "num_batches_tracked"

    if lower.endswith("running_mean") or "running_mean" in lower:
        return "running_mean"

    if lower.endswith("running_var") or "running_var" in lower:
        return "running_var"

    if lower.endswith(".weight") or lower.endswith("_weight"):
        if any(keyword in lower for keyword in NORM_KEYWORDS):
            return "norm_weight"
        return "weight"

    if lower.endswith(".bias") or lower.endswith("_bias"):
        if any(keyword in lower for keyword in NORM_KEYWORDS):
            return "norm_bias"
        return "bias"

    if torch.is_floating_point(tensor):
        return "buffer"

    return "unknown"


def is_integer_tensor(tensor: torch.Tensor) -> bool:
    return not torch.is_floating_point(tensor) and not tensor.is_complex() and tensor.dtype != torch.bool


def is_weight_like_role(role: str) -> bool:
    return role in {"weight", "bias", "norm_weight", "norm_bias"}


def is_buffer_role(role: str) -> bool:
    return role in {"running_mean", "running_var", "num_batches_tracked", "buffer"}


def is_norm_stat_role(role: str) -> bool:
    return role in {"running_mean", "running_var", "num_batches_tracked"}


def is_quantizable_weight(
    tensor: torch.Tensor,
    *,
    role: str,
    type_guess: str,
) -> bool:
    return (
        torch.is_floating_point(tensor)
        and role in {"weight", "bias"}
        and type_guess in {"conv_weight", "linear_weight", "bias"}
    )


def semantic_flags(name: str, tensor: torch.Tensor, type_guess: str | None = None) -> dict[str, bool | str]:
    role = guess_tensor_role(name, tensor)
    guessed_type = type_guess or guess_tensor_type(name, tensor)
    is_float = bool(torch.is_floating_point(tensor))
    is_integer = bool(is_integer_tensor(tensor))
    weight_like = is_weight_like_role(role)
    buffer = is_buffer_role(role)
    norm_stat = is_norm_stat_role(role)
    quantizable = is_quantizable_weight(tensor, role=role, type_guess=guessed_type)

    return {
        "role": role,
        "is_float": is_float,
        "is_integer": is_integer,
        "is_weight_like": weight_like,
        "is_buffer": buffer,
        "is_norm_stat": norm_stat,
        "is_quantizable_weight": quantizable,
        "include_in_weight_health": is_float and role != "num_batches_tracked",
        "include_in_quant_analysis": quantizable,
        "include_in_bn_risk": role in {"running_mean", "running_var", "norm_weight", "norm_bias"},
    }


def filter_quantizable(records: Iterable[TensorRecord]) -> list[TensorRecord]:
    return [record for record in records if record.include_in_quant_analysis]


def filter_weight_like(records: Iterable[TensorRecord]) -> list[TensorRecord]:
    return [record for record in records if record.is_weight_like]


def filter_buffers(records: Iterable[TensorRecord]) -> list[TensorRecord]:
    return [record for record in records if record.is_buffer]


def filter_bn_norm(records: Iterable[TensorRecord]) -> list[TensorRecord]:
    return [record for record in records if record.include_in_bn_risk]


def filter_weight_health(records: Iterable[TensorRecord]) -> list[TensorRecord]:
    return [record for record in records if record.include_in_weight_health]
