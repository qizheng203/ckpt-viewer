"""Checkpoint state_dict discovery and key normalization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


CANDIDATE_KEYS = [
    "state_dict",
    "model",
    "model_state_dict",
    "net",
    "network",
    "module",
    "ema",
    "ema_state_dict",
    "student",
    "teacher",
]

SKIP_KEYS = {
    "optimizer",
    "optim",
    "scheduler",
    "lr_scheduler",
    "scaler",
    "amp_scaler",
    "callbacks",
    "hyper_parameters",
}

COMMON_PREFIXES = (
    "module.",
    "model.",
    "net.",
    "network.",
    "student.",
    "teacher.",
)


class StateDictNotFoundError(RuntimeError):
    """Raised when a loaded checkpoint does not contain model tensors."""


@dataclass(frozen=True)
class StateDictMatch:
    state_dict: dict[str, torch.Tensor]
    key: str | None


def _is_mapping(obj: Any) -> bool:
    return isinstance(obj, Mapping)


def _tensor_ratio(obj: Mapping[Any, Any]) -> tuple[int, float]:
    if not obj:
        return 0, 0.0
    tensor_count = sum(torch.is_tensor(value) for value in obj.values())
    return tensor_count, tensor_count / max(len(obj), 1)


def is_state_dict(obj: object) -> bool:
    if not _is_mapping(obj) or len(obj) == 0:  # type: ignore[arg-type]
        return False
    tensor_count, ratio = _tensor_ratio(obj)  # type: ignore[arg-type]
    return tensor_count >= 1 and ratio > 0.5


def strip_common_prefix(name: str) -> str:
    stripped = name
    changed = True
    while changed:
        changed = False
        for prefix in COMMON_PREFIXES:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :]
                changed = True
                break
    return stripped


def normalize_state_dict_keys(
    state_dict: Mapping[Any, Any], *, strip_prefixes: bool = True
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for raw_name, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        name = str(raw_name)
        if strip_prefixes:
            name = strip_common_prefix(name)
        if name in tensors:
            name = str(raw_name)
        tensors[name] = value.detach().cpu()
    return tensors


def _state_dict_from_module(obj: Any) -> dict[str, torch.Tensor] | None:
    state_dict_fn = getattr(obj, "state_dict", None)
    if not callable(state_dict_fn):
        return None
    try:
        maybe_state_dict = state_dict_fn()
    except Exception:
        return None
    if is_state_dict(maybe_state_dict):
        return normalize_state_dict_keys(maybe_state_dict)
    return None


def _find_in_mapping(
    obj: Mapping[Any, Any],
    *,
    parent_key: str | None,
    depth: int,
) -> StateDictMatch | None:
    if is_state_dict(obj):
        return StateDictMatch(normalize_state_dict_keys(obj), parent_key)

    if depth <= 0:
        return None

    for key in CANDIDATE_KEYS:
        if key not in obj:
            continue
        value = obj[key]
        if is_state_dict(value):
            return StateDictMatch(normalize_state_dict_keys(value), key)
        module_state = _state_dict_from_module(value)
        if module_state is not None:
            return StateDictMatch(module_state, key)
        if _is_mapping(value):
            nested = _find_in_mapping(value, parent_key=key, depth=depth - 1)
            if nested is not None:
                nested_key = key if nested.key is None else f"{key}.{nested.key}"
                return StateDictMatch(nested.state_dict, nested_key)

    best: StateDictMatch | None = None
    best_tensor_count = 0
    for raw_key, value in obj.items():
        key = str(raw_key)
        if key.lower() in SKIP_KEYS:
            continue
        if _is_mapping(value):
            nested = _find_in_mapping(value, parent_key=key, depth=depth - 1)
            if nested is None:
                continue
            tensor_count = len(nested.state_dict)
            if tensor_count > best_tensor_count:
                best = nested
                best_tensor_count = tensor_count
        else:
            module_state = _state_dict_from_module(value)
            if module_state is None:
                continue
            tensor_count = len(module_state)
            if tensor_count > best_tensor_count:
                best = StateDictMatch(module_state, key)
                best_tensor_count = tensor_count

    return best


def find_state_dict(obj: Any) -> StateDictMatch:
    """Find and normalize a model state_dict from a loaded checkpoint object."""

    if _is_mapping(obj):
        match = _find_in_mapping(obj, parent_key=None, depth=3)
        if match is not None and match.state_dict:
            return match

    module_state = _state_dict_from_module(obj)
    if module_state is not None:
        return StateDictMatch(module_state, "__module__.state_dict")

    raise StateDictNotFoundError(
        "未检测到模型权重。该文件可能只包含 optimizer/scheduler 状态，"
        "或 checkpoint 使用了非常规字段名。请检查文件中是否包含 "
        "state_dict/model/model_state_dict/net/network 等字段。"
    )

