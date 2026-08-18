"""
GPU compute policy.

**Model training and batch inference run on GPU wherever the library supports
it, and the device actually used is recorded either way.** Silent fallback is
prohibited: a run that quietly drops to CPU is a run that may take 40x longer,
blow its SLA, and — where kernels and reduction orders differ across devices —
fail to reproduce the artefacts recorded in the decision log. Determinism is a
stated guarantee (README §5), so the device is part of the contract.

**The probe is per library, not per process.** An earlier version of this module
asked *PyTorch* whether CUDA was available and then handed ``device_type: cuda``
to *LightGBM* — a different library, built separately, with its own CUDA
support. On a machine with a working CUDA PyTorch and a stock PyPI LightGBM
wheel (which ships **without** the CUDA tree learner) that combination fails at
training time with ``CUDA Tree Learner was not enabled in this build``. The bug
was invisible only because a CPU-only PyTorch was masking it: fixing the PyTorch
install would have broken the demand model. Each backend is now asked about
itself.

The policy this leaves is narrower and true, rather than broad and aspirational:

* PyTorch components (survival CLV, embeddings, Monte-Carlo simulation, policy
  learning) **require** CUDA and raise :class:`GPUUnavailableError` without it.
  These are the workloads where the GPU is worth several multiples of wall clock.
* Gradient-boosting components use CUDA **when the installed build provides
  it**, and otherwise run on CPU with a warning naming the reason. This is not a
  reproducibility hole: CPU histogram training is deterministic, and the device
  is written into the artefact, so the record still says what produced it. It
  is also frequently the faster choice — CUDA LightGBM loses to CPU histogram at
  panel sizes in the low millions of rows, so requiring GPU here would cost time
  rather than save it.

Every ML entry point opens with either::

    device = require_gpu("estimation.retention")     # torch: hard requirement

or::

    params = lightgbm_device_params("estimation.demand")   # best available

The ``PRISMPRICE_ALLOW_CPU=1`` escape hatch downgrades the hard requirement to a
loud warning, and exists for CI and for the pure-Python governance tests that
touch no ML code.
"""

from __future__ import annotations

import functools
import logging
import warnings
from dataclasses import dataclass
from typing import Any

from prismprice.config import ALLOW_CPU_ENV_VAR, cpu_fallback_allowed

logger = logging.getLogger(__name__)

__all__ = [
    "BackendSupport",
    "GPUInfo",
    "GPUUnavailableError",
    "compute_report",
    "gpu_report",
    "lightgbm_device_params",
    "lightgbm_gpu_support",
    "require_gpu",
    "torch_device",
    "xgboost_device_params",
    "xgboost_gpu_support",
]


class GPUUnavailableError(RuntimeError):
    """Raised when a GPU-only component cannot obtain a CUDA device."""


@dataclass(frozen=True)
class GPUInfo:
    """Snapshot of CUDA availability via PyTorch, recorded with model artefacts."""

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


@dataclass(frozen=True)
class BackendSupport:
    """Whether one specific ML library can use CUDA in *this* installation.

    Separate from :class:`GPUInfo` on purpose. ``GPUInfo`` answers "does this
    machine have a usable CUDA device", which is a hardware and driver question.
    This answers "was the installed build of this library compiled to use it",
    which is a packaging question with a completely different fix.
    """

    library: str
    available: bool
    reason: str
    version: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "library": self.library,
            "available": self.available,
            "reason": self.reason,
            "version": self.version,
        }


def gpu_report() -> GPUInfo:
    """Inspect CUDA availability through PyTorch, without raising.

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
                "pip install torch --index-url https://download.pytorch.org/whl/cu128"
            ),
        )

    torch_version = torch.__version__
    cuda_version = getattr(torch.version, "cuda", None)

    if cuda_version is None:
        return GPUInfo(
            available=False,
            reason=(
                f"PyTorch {torch_version} is a CPU-only build (torch.version.cuda is None). "
                "Reinstall the CUDA wheel: pip install --upgrade "
                "--index-url https://download.pytorch.org/whl/cu128 torch"
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
            ``"estimation.retention"``. Used in the error and warning text so a
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
        f"for this component.\n"
        f"  Reason: {info.reason}\n"
        f"  To run without a GPU anyway (CI / tests only, results not reproducible), "
        f"set {ALLOW_CPU_ENV_VAR}=1."
    )


def torch_device(component: str) -> Any:
    """Alias of :func:`require_gpu`, for call sites that read better this way."""
    return require_gpu(component)


@functools.lru_cache(maxsize=1)
def lightgbm_gpu_support() -> BackendSupport:
    """Whether the installed LightGBM was compiled with the CUDA tree learner.

    There is no public API that reports this, and the build flag is not exposed
    on the module, so the only reliable check is to attempt a two-round fit on a
    trivial dataset and read the error. That costs a few milliseconds once per
    process; the result is cached.

    Stock PyPI wheels are CPU-only. CUDA support requires building from source
    with ``-DUSE_CUDA=1``.
    """
    try:
        import lightgbm as lgb
        import numpy as np
    except ImportError as exc:
        return BackendSupport(
            library="lightgbm",
            available=False,
            reason=f"lightgbm is not installed ({exc}). Install with: pip install lightgbm",
        )

    version = getattr(lgb, "__version__", None)

    if not gpu_report().available:
        return BackendSupport(
            library="lightgbm",
            available=False,
            reason=(
                "No usable CUDA device on this machine, so the LightGBM build flag is moot. "
                f"{gpu_report().reason}"
            ),
            version=version,
        )

    rng = np.random.default_rng(0)
    x = rng.random((64, 3))
    y = rng.random(64)
    try:
        lgb.train(
            {"objective": "regression", "device_type": "cuda", "verbose": -1, "num_leaves": 2},
            lgb.Dataset(x, label=y),
            num_boost_round=2,
        )
    except Exception as exc:  # lgb.basic.LightGBMError, but the probe must never escape
        return BackendSupport(
            library="lightgbm",
            available=False,
            reason=(
                f"LightGBM {version} was not built with CUDA support ({str(exc).strip()[:160]}). "
                "PyPI wheels are CPU-only; CUDA requires a source build with -DUSE_CUDA=1. "
                "CPU histogram training is deterministic and is often faster at this data "
                "size, so this is not treated as a failure."
            ),
            version=version,
        )

    return BackendSupport(
        library="lightgbm",
        available=True,
        reason="LightGBM CUDA tree learner available",
        version=version,
    )


@functools.lru_cache(maxsize=1)
def xgboost_gpu_support() -> BackendSupport:
    """Whether the installed XGBoost can use CUDA.

    Unlike LightGBM, official XGBoost wheels do ship with CUDA support, so this
    usually reduces to whether a device is visible.
    """
    try:
        import xgboost as xgb
    except ImportError as exc:
        return BackendSupport(
            library="xgboost",
            available=False,
            reason=f"xgboost is not installed ({exc})",
        )

    version = getattr(xgb, "__version__", None)
    info = gpu_report()
    if not info.available:
        return BackendSupport(
            library="xgboost", available=False, reason=info.reason, version=version
        )
    return BackendSupport(
        library="xgboost", available=True, reason="XGBoost CUDA available", version=version
    )


def lightgbm_device_params(component: str = "lightgbm") -> dict[str, Any]:
    """LightGBM parameters pinning training to the best available device.

    Asks LightGBM about LightGBM. Returns CUDA parameters only when the
    installed build can actually honour them; otherwise returns CPU parameters
    and warns with the specific reason, so the fallback appears in the run log
    rather than being inferred from a timing anomaly.

    ``deterministic`` and ``force_row_wise`` are set on both paths: they are
    what make a fit reproducible, which is the guarantee that actually matters
    here, and it is not a GPU-only concern.
    """
    support = lightgbm_gpu_support()

    if not support.available:
        warnings.warn(
            f"{component} is training LightGBM on CPU. {support.reason}",
            RuntimeWarning,
            stacklevel=2,
        )
        logger.info("%s using LightGBM CPU backend: %s", component, support.reason)
        return {"device_type": "cpu", "deterministic": True, "force_row_wise": True}

    logger.info("%s using LightGBM CUDA backend (%s)", component, support.version)
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
    """XGBoost parameters pinning training to the best available device."""
    support = xgboost_gpu_support()
    if not support.available:
        warnings.warn(
            f"{component} is training XGBoost on CPU. {support.reason}",
            RuntimeWarning,
            stacklevel=2,
        )
        return {"device": "cpu", "tree_method": "hist"}
    return {"device": "cuda", "tree_method": "hist"}


def compute_report() -> dict[str, Any]:
    """Full device picture: the machine, plus each backend's own capability.

    Attached to model artefacts and printed by the diagnostics playbook. The
    per-backend rows are the ones that explain a slow run, because a green
    ``GPUInfo`` with a red ``lightgbm`` row is exactly the state that used to be
    reported as "GPU enabled".
    """
    return {
        "device": gpu_report().as_dict(),
        "backends": {
            "lightgbm": lightgbm_gpu_support().as_dict(),
            "xgboost": xgboost_gpu_support().as_dict(),
        },
    }
