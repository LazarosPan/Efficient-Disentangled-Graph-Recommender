"""GPU profiler: stage-level timing + VRAM tracking, with PyG profiling utilities."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch_geometric.profile import count_parameters, get_model_size, get_data_size


@dataclass
class StageMetrics:
    name: str
    elapsed_ms: float = 0.0
    vram_before_mb: float = 0.0
    vram_after_mb: float = 0.0
    vram_peak_mb: float = 0.0

    @property
    def vram_delta_mb(self) -> float:
        return self.vram_after_mb - self.vram_before_mb

    def __repr__(self) -> str:
        return (
            f"{self.name}: {self.elapsed_ms:.1f}ms | "
            f"VRAM: {self.vram_before_mb:.0f} -> {self.vram_after_mb:.0f} MB "
            f"(peak {self.vram_peak_mb:.0f} MB, delta {self.vram_delta_mb:+.0f} MB)"
        )


@dataclass
class GPUProfiler:
    """Collects per-stage timing and VRAM usage across an epoch."""

    stages: list[StageMetrics] = field(default_factory=list)
    _enabled: bool = False  # TODO: Change to True to profile the GPU

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable stage collection for the current epoch."""
        self._enabled = enabled

    def reset(self) -> None:
        self.stages.clear()
        if self._enabled and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @contextmanager
    def stage(self, name: str):
        """Context manager for profiling a named stage."""
        if not self._enabled or not torch.cuda.is_available():
            yield
            return

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        vram_before = torch.cuda.memory_allocated() / 1024 / 1024
        t0 = time.perf_counter()

        yield

        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        vram_after = torch.cuda.memory_allocated() / 1024 / 1024
        vram_peak = torch.cuda.max_memory_allocated() / 1024 / 1024

        self.stages.append(
            StageMetrics(
                name=name,
                elapsed_ms=elapsed,
                vram_before_mb=vram_before,
                vram_after_mb=vram_after,
                vram_peak_mb=vram_peak,
            )
        )

    def summary(self) -> str:
        lines = ["=== GPU Profile ==="]
        total_ms = sum(s.elapsed_ms for s in self.stages)
        for s in self.stages:
            pct = (s.elapsed_ms / total_ms * 100) if total_ms > 0 else 0
            lines.append(
                f"  {s.name:20s} {s.elapsed_ms:8.1f}ms ({pct:5.1f}%) | "
                f"VRAM peak {s.vram_peak_mb:.0f} MB"
            )
        lines.append(f"  {'TOTAL':20s} {total_ms:8.1f}ms")
        return "\n".join(lines)

    @staticmethod
    def model_summary(model: torch.nn.Module) -> str:
        """Return parameter count and model size via PyG utilities."""
        n_params = count_parameters(model)
        size_bytes = get_model_size(model)
        return f"Parameters: {n_params:,} | Size: {size_bytes / 1024 / 1024:.1f} MB"

    @staticmethod
    def data_summary(data) -> str:
        """Return data object size via PyG utilities."""
        size_bytes = get_data_size(data)
        return f"Data size: {size_bytes / 1024 / 1024:.1f} MB"


@contextmanager
def profile_stage(name: str, profiler: GPUProfiler | None = None):
    """Convenience wrapper -- no-op if profiler is None."""
    if profiler is not None:
        with profiler.stage(name):
            yield
    else:
        yield
