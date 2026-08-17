"""
GPU-only compute policy.

**All model training and batch inference in PrismPrice runs on GPU.** Silent CPU
fallback is prohibited: a run that quietly drops to CPU is a run that takes 40x
longer, blows its SLA, and — because seeds and kernels differ across devices —
may not be bit-reproducible against the artefacts recorded in the decision log.
Determinism is a stated guarantee (README §5), so the device is part of the
contract, not an implementation detail.

Every ML entry point must therefore open with::

    from prismprice.compute import require_gpu

    device = require_gpu("estimation.demand")

which raises :class:`GPUUnavailableError` rather than degrading.

The single escape hatch is the ``PRISMPRICE_ALLOW_CPU=1`` environment variable,
intended only for CI and for the pure-Python governance tests that touch no ML
code. It is loud: every fallback emits a warning naming the component.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

from prismprice.config import ALLOW_CPU_ENV_VAR, cpu_fallback_allowed

logger = logging.getLogger(__name__)

__all__ = [
    "GPUInfo",
    "GPUUnavailableError",
    "gpu_report",
    "lightgbm_device_params",
    "require_gpu",
    "torch_device",
    "xgboost_device_params",
]


class GPUUnavailableError(RuntimeError):
    """Raised when a GPU-only component cannot obtain a CUDA device."""


@dataclass(frozen=True)
class GPUInfo:
    """Snapshot of CUDA availability, recorded alongside model artefacts."""

    available: bool
    reason: str
    device_count: int = 0
    device_name: str | None = None
    total_memory_gb: float | None = None
    torch_version: str | None = None
    cuda_version: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "device_count": self.device_count,
            "device_name": self.device_name,
            "total_memory_gb": self.total_memory_gb,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
        }


def gpu_report() -> GPUInfo:
    """Inspect CUDA availability without raising.

    Distinguishes the three failure modes that look identical from the outside:
    torch missing, torch built without CUDA, and CUDA present but no visible
    device. Each needs a different fix, so each gets its own message.
    """
    try:
        import torch
    except ImportError:
        return GPUInfo(
            available=False,
            reason=(
                "PyTorch is not installed. Install the CUDA build: "
                "pip install torch --index-url https://download.pytorch.org/whl/cu124"
            ),
        )

    torch_version = torch.__version__
    cuda_version = getattr(torch.version, "cuda", None)

    if cuda_version is None:
        return GPUInfo(
            available=False,
            reason=(
                f"PyTorch {torch_version} is a CPU-only build (torch.version.cuda is None). "
                "Reinstall the CUDA wheel: pip uninstall -y torch && "
                "pip install torch --index-url https://download.pytorch.org/whl/cu124"
            ),
            torch_version=torch_version,
        )

    if not torch.cuda.is_available():
        return GPUInfo(
            available=False,
            reason=(
                f"PyTorch {torch_version} was built against CUDA {cuda_version} but no CUDA "
                "device is visible. Check the NVIDIA driver and CUDA_VISIBLE_DEVICES."
            ),
            torch_version=torch_version,
            cuda_version=cuda_version,
        )

    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return GPUInfo(
        available=True,
        reason="CUDA device available",
        device_count=torch.cuda.device_count(),
        device_name=props.name,
        total_memory_gb=round(props.total_memory / 1024**3, 2),
        torch_version=torch_version,
        cuda_version=cuda_version,
    )


def require_gpu(component: str) -> Any:
    """Return the CUDA ``torch.device`` for *component*, or raise.

    Args:
        component: Dotted name of the calling component, e.g.
            ``"estimation.elasticity"``. Used in the error and warning text so a
            failure identifies itself without a traceback read.

    Returns:
        ``torch.device("cuda")``, or ``torch.device("cpu")`` when the
        ``PRISMPRICE_ALLOW_CPU`` escape hatch is set.

    Raises:
        GPUUnavailableError: CUDA is unavailable and the escape hatch is not set.
    """
    info = gpu_report()

    if info.available:
        import torch

        logger.info(
            "%s acquired GPU: %s (%.2f GB, CUDA %s)",
            component,
            info.device_name,
            info.total_memory_gb or 0.0,
            info.cuda_version,
        )
        return torch.device("cuda")

    if cpu_fallback_allowed():
        warnings.warn(
            f"{component} is running on CPU because {ALLOW_CPU_ENV_VAR} is set. "
            f"Results are NOT reproducible against GPU-trained artefacts. "
            f"Reason GPU was unavailable: {info.reason}",
            RuntimeWarning,
            stacklevel=2,
        )
        logger.warning("%s falling back to CPU (%s set)", component, ALLOW_CPU_ENV_VAR)
        import torch

        return torch.device("cpu")

    raise GPUUnavailableError(
        f"{component} requires a CUDA GPU and PrismPrice does not fall back to CPU "
        f"for model training or inference.\n"
        f"  Reason: {info.reason}\n"
        f"  To run without a GPU anyway (CI / tests only, results not reproducible), "
        f"set {ALLOW_CPU_ENV_VAR}=1."
    )


def torch_device(component: str) -> Any:
    """Alias of :func:`require_gpu`, for call sites that read better this way."""
    return require_gpu(component)


def lightgbm_device_params(component: str = "lightgbm") -> dict[str, Any]:
    """LightGBM parameters pinning training to the GPU.

    LightGBM does not raise when its GPU build is missing — it warns and trains
    on CPU. Calling :func:`require_gpu` first converts that silent degradation
    into the same hard failure every other component gets.
    """
    device = require_gpu(component)
    if getattr(device, "type", "cpu") == "cpu":
        return {"device_type": "cpu"}
    return {
        "device_type": "cuda",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        # Required for reproducibility: GPU histogram construction is otherwise
        # order-dependent across runs.
        "deterministic": True,
        "force_row_wise": True,
    }


def xgboost_device_params(component: str = "xgboost") -> dict[str, Any]:
    """XGBoost parameters pinning training to the GPU."""
    device = require_gpu(component)
    if getattr(device, "type", "cpu") == "cpu":
        return {"device": "cpu", "tree_method": "hist"}
    return {"device": "cuda", "tree_method": "hist"}
