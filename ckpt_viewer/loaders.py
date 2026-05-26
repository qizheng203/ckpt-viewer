"""Weight file loading utilities.

The default path uses PyTorch's ``weights_only=True`` loading mode. Unsafe
pickle loading is only used when the caller explicitly passes ``trusted=True``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch


SUPPORTED_EXTENSIONS = {".pth", ".pt", ".ckpt", ".bin", ".safetensors"}


class SafeLoadError(RuntimeError):
    """Raised when safe loading fails and unsafe loading was not requested."""


class UnsupportedFormatError(ValueError):
    """Raised when the file suffix is not supported by the MVP."""


def compute_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_file_info(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"文件不存在：{file_path}。请检查路径是否正确，或在 Web UI 中重新选择文件。"
        )
    if not file_path.is_file():
        raise FileNotFoundError(f"目标不是文件：{file_path}。请选择具体权重文件。")

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFormatError(
            f"不支持的权重格式：{suffix or '<无后缀>'}。当前版本支持：{supported}。"
        )

    size_bytes = file_path.stat().st_size
    return {
        "path": str(file_path),
        "name": file_path.name,
        "suffix": suffix,
        "size_bytes": size_bytes,
        "size_mb": size_bytes / (1024 * 1024),
        "sha256": compute_sha256(file_path),
    }


def _torch_load(path: Path, *, weights_only: bool) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=weights_only)
    except TypeError:
        if weights_only:
            raise SafeLoadError(
                "当前 PyTorch 版本不支持 weights_only=True 安全加载。"
                "请升级 PyTorch，或确认文件可信后使用 trusted unsafe load。"
            )
        return torch.load(path, map_location="cpu")


def _load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError(
            "加载 .safetensors 需要安装 safetensors。请执行：pip install safetensors"
        ) from exc
    return load_file(str(path), device="cpu")


def load_weight_file(path: str | Path, trusted: bool = False) -> tuple[Any, str]:
    """Load a checkpoint or safetensors file.

    Args:
        path: Weight file path.
        trusted: Whether to allow unsafe ``torch.load(..., weights_only=False)``.

    Returns:
        A tuple of ``(loaded_object, load_mode)``.

    Raises:
        SafeLoadError: Safe PyTorch loading failed and trusted mode is disabled.
        UnsupportedFormatError: File suffix is unsupported.
        FileNotFoundError: The path does not exist.
    """

    file_path = Path(path)
    info = get_file_info(file_path)
    suffix = info["suffix"]

    if suffix == ".safetensors":
        return _load_safetensors(file_path), "safetensors"

    try:
        return _torch_load(file_path, weights_only=True), "torch_safe"
    except Exception as safe_exc:
        if not trusted:
            raise SafeLoadError(
                "安全模式加载失败。该文件可能包含自定义对象、完整 nn.Module、优化器对象或旧格式 pickle 内容。"
                "如果这是你自己训练得到且确认可信的 checkpoint，请开启 trusted unsafe load 后重试。"
            ) from safe_exc

    try:
        return _torch_load(file_path, weights_only=False), "torch_unsafe"
    except Exception as unsafe_exc:
        raise RuntimeError(
            "unsafe load 也加载失败。可能原因：文件损坏、格式不是 PyTorch checkpoint、"
            "或缺少保存该对象时依赖的 Python 类。请优先确认文件来源和格式。"
        ) from unsafe_exc
