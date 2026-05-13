"""Shared reproducibility helpers for seeded and deterministic runtime execution."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    Args:
        seed: Seed value applied to all repository-owned RNG surfaces.

    Returns:
        None. Global RNG state is updated in place.

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch_runtime(deterministic: bool = True) -> None:
    """Configure PyTorch backend flags for efficient and deterministic execution.

    Efficiency note: float32 matmul precision is always ``"medium"`` (bfloat16
    internally, 7 mantissa bits) for faster throughput on supported hardware.
    Use ``"high"`` (TF32, 10 mantissa bits) or ``"highest"`` (full float32,
    23 mantissa bits) if numeric precision is critical.

    When ``deterministic=True`` (default):
    - Sets ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` for reproducible cuBLAS kernels.
    - Calls ``torch.use_deterministic_algorithms(True)``.
    - Disables cuDNN algorithm benchmarking (``benchmark=False``).
    - Forces deterministic cuDNN convolution algorithms (``deterministic=True``).
    - Disables TF32 on matmul and cuDNN (``allow_tf32=False``) for consistency.

    When ``deterministic=False``:
    - Enables cuDNN benchmarking (selects fastest algorithm per input shape).
    - Enables TF32 on Ampere+ GPUs for faster matmul and convolutions.

    In both cases, ``TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT`` is capped at 256
    execution plans (~51 MiB) to conserve VRAM over the 2 GiB default.

    Note: ``cudnn.allow_tf32`` is scheduled for deprecation; prefer controlling
    TF32 via ``torch.backends.cuda.matmul.allow_tf32`` in future PyTorch versions.

    Args:
        deterministic: Whether to force deterministic backend behavior. Defaults to
            True for reproducible thesis experiments.

    Returns:
        None. PyTorch global backend settings are updated in place.

    """
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic)
    torch.set_float32_matmul_precision("medium")
    if not torch.cuda.is_available():
        return

    os.environ.setdefault("TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT", "256")
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = not deterministic
    torch.backends.cudnn.allow_tf32 = not deterministic


def build_torch_generator(
    seed: int,
    device: torch.device | str | None = None,
) -> torch.Generator:
    """Return a seeded torch.Generator on the requested device family.

    Args:
        seed: Seed value applied to the generator.
        device: Optional torch device or device string.

    Returns:
        torch.Generator: Seeded generator for deterministic sampling calls.

    """
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    generator = torch.Generator(device=target_device.type)
    generator.manual_seed(seed)
    return generator
