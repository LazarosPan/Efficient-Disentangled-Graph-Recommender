"""GPU profiler: stage-level timing + VRAM tracking, with PyG profiling utilities."""

from __future__ import annotations

import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch_geometric.profile import count_parameters, get_data_size, get_model_size


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
            f"{self.name}: {self.elapsed_ms:.1f}ms | VRAM: {self.vram_before_mb:.0f} -> "
            f"{self.vram_after_mb:.0f} MB (peak {self.vram_peak_mb:.0f} MB, "
            f"delta {self.vram_delta_mb:+.0f} MB)"
        )


@dataclass(frozen=True)
class GPUResourceSnapshot:
    """One nvidia-smi sample for GPU utilization and memory use."""

    utilization_pct: float | None
    memory_used_mb: float | None


@dataclass(frozen=True)
class TrainingResourceStats:
    """Aggregate training-window GPU resource measurements."""

    pytorch_peak_allocated_mb: float | None = None
    pytorch_peak_reserved_mb: float | None = None
    nvidia_peak_memory_used_mb: float | None = None
    avg_gpu_utilization_pct: float | None = None
    max_gpu_utilization_pct: float | None = None

    @property
    def peak_vram_mb(self) -> float | None:
        """Return the peak VRAM value to use for summary reporting."""
        if self.nvidia_peak_memory_used_mb is not None:
            return self.nvidia_peak_memory_used_mb
        return self.pytorch_peak_allocated_mb

    @classmethod
    def from_current_cuda_peaks(cls, device: torch.device) -> TrainingResourceStats:
        """Build stats from PyTorch allocator peaks for the active CUDA device."""
        if device.type != "cuda" or not torch.cuda.is_available():
            return cls()
        mib = 1024 * 1024
        return cls(
            pytorch_peak_allocated_mb=torch.cuda.max_memory_allocated() / mib,
            pytorch_peak_reserved_mb=torch.cuda.max_memory_reserved() / mib,
        )


@dataclass
class GPUProfiler:
    """Collects per-stage timing and VRAM usage across an epoch."""

    stages: list[StageMetrics] = field(default_factory=list)
    epoch_elapsed_ms: float = 0.0
    _enabled: bool = False  # Change to True if we want to profile the GPU

    def reset(self, enabled: bool) -> None:
        """Set enabled state and clear stage data for a new epoch.

        Args:
            enabled: Whether stage profiling is active this epoch.

        """
        self._enabled = enabled
        self.stages.clear()
        self.epoch_elapsed_ms = 0.0
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
            ),
        )

    def summary(self) -> str:
        """Return a formatted multi-line profiling summary for the last epoch."""
        lines = ["=== GPU Profile ==="]
        total_ms = sum(s.elapsed_ms for s in self.stages)
        for s in self.stages:
            pct = (s.elapsed_ms / total_ms * 100) if total_ms > 0 else 0
            lines.append(
                ""
                f"  {s.name:20s} {s.elapsed_ms:8.1f}ms ({pct:5.1f}%) | "
                f"VRAM peak {s.vram_peak_mb:.0f} MB",
            )
        lines.append(f"  {'TOTAL':20s} {total_ms:8.1f}ms")
        if self.epoch_elapsed_ms > 0:
            lines.append(f"  {'EPOCH WALL':20s} {self.epoch_elapsed_ms:8.1f}ms")
        return "\n".join(lines)

    @staticmethod
    def peak_vram_mb() -> float | None:
        """Return peak VRAM allocated (MB) since the last reset, without sync.

        Returns None when CUDA is unavailable.
        """
        if not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_allocated() / 1024 / 1024

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


def _parse_optional_float(value: str) -> float | None:
    """Parse a numeric nvidia-smi field, returning None for unsupported values."""
    cleaned = value.strip()
    if not cleaned or cleaned.upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def sample_gpu_resource_snapshot(device: torch.device) -> GPUResourceSnapshot | None:
    """Return current GPU utilization and memory use via ``nvidia-smi``.

    Args:
        device: Active runtime device.

    Returns:
        GPUResourceSnapshot for the current CUDA device, or ``None`` when the
        device is not CUDA or the system utility is unavailable.

    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    device_id = device.index if device.index is not None else torch.cuda.current_device()
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
                "--id",
                str(device_id),
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    if not output:
        return None
    fields = [part.strip() for part in output.splitlines()[0].split(",")]
    if len(fields) < 2:
        return None
    return GPUResourceSnapshot(
        utilization_pct=_parse_optional_float(fields[0]),
        memory_used_mb=_parse_optional_float(fields[1]),
    )


class TrainingResourceMonitor:
    """Sample GPU resources while the training batch loop is active."""

    def __init__(
        self,
        device: torch.device,
        *,
        sample_interval_s: float = 0.5,
    ) -> None:
        """Initialize a monitor for one training window.

        Args:
            device: Active training device.
            sample_interval_s: Background nvidia-smi sampling interval.

        """
        self.device = device
        self.sample_interval_s = max(0.05, float(sample_interval_s))
        self._samples: list[GPUResourceSnapshot] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._enabled = device.type == "cuda" and torch.cuda.is_available()

    def start(self) -> TrainingResourceMonitor:
        """Start collecting training-window samples."""
        if not self._enabled:
            return self
        self._record_sample()
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="gpu-training-resource-monitor",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> TrainingResourceStats:
        """Stop sampling and return aggregate training-window stats."""
        if not self._enabled:
            return TrainingResourceStats()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.sample_interval_s * 2.0))
        self._record_sample()
        stats = TrainingResourceStats.from_current_cuda_peaks(self.device)
        utilization_values = [
            sample.utilization_pct for sample in self._samples if sample.utilization_pct is not None
        ]
        memory_values = [
            sample.memory_used_mb for sample in self._samples if sample.memory_used_mb is not None
        ]
        return TrainingResourceStats(
            pytorch_peak_allocated_mb=stats.pytorch_peak_allocated_mb,
            pytorch_peak_reserved_mb=stats.pytorch_peak_reserved_mb,
            nvidia_peak_memory_used_mb=max(memory_values) if memory_values else None,
            avg_gpu_utilization_pct=(
                sum(utilization_values) / len(utilization_values) if utilization_values else None
            ),
            max_gpu_utilization_pct=max(utilization_values) if utilization_values else None,
        )

    def _sample_loop(self) -> None:
        """Collect nvidia-smi samples until the monitor is stopped."""
        while not self._stop_event.wait(self.sample_interval_s):
            self._record_sample()

    def _record_sample(self) -> None:
        """Append one nvidia-smi sample when available."""
        snapshot = sample_gpu_resource_snapshot(self.device)
        if snapshot is not None:
            self._samples.append(snapshot)
