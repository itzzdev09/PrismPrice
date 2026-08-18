"""
GPU-only compute policy.

These tests must pass on CPU-only CI runners, so GPU state is simulated by
patching :func:`prismprice.compute.gpu_report` rather than by requiring hardware.
"""

import pytest

from prismprice import compute
from prismprice.compute import BackendSupport, GPUInfo, GPUUnavailableError
from prismprice.config import ALLOW_CPU_ENV_VAR, cpu_fallback_allowed

UNAVAILABLE = GPUInfo(available=False, reason="simulated: no CUDA device")
AVAILABLE = GPUInfo(
    available=True,
    reason="CUDA device available",
    device_count=1,
    device_name="Simulated RTX",
    total_memory_gb=6.0,
    torch_version="2.10.0+cu124",
    cuda_version="12.4",
)


@pytest.fixture(autouse=True)
def _clear_escape_hatch(monkeypatch):
    monkeypatch.delenv(ALLOW_CPU_ENV_VAR, raising=False)


# ---------------------------------------------------------------------------
# Hard failure is the default
# ---------------------------------------------------------------------------


def test_require_gpu_raises_rather_than_falling_back(monkeypatch):
    monkeypatch.setattr(compute, "gpu_report", lambda: UNAVAILABLE)
    with pytest.raises(GPUUnavailableError) as excinfo:
        compute.require_gpu("estimation.demand")
    message = str(excinfo.value)
    assert "estimation.demand" in message
    assert "simulated: no CUDA device" in message
    assert ALLOW_CPU_ENV_VAR in message


def _support(library: str, available: bool, reason: str) -> BackendSupport:
    return BackendSupport(library=library, available=available, reason=reason, version="test")


NO_LGB_CUDA = _support("lightgbm", False, "CUDA Tree Learner was not enabled in this build")
LGB_CUDA = _support("lightgbm", True, "LightGBM CUDA tree learner available")
NO_XGB_CUDA = _support("xgboost", False, "simulated: no CUDA device")
XGB_CUDA = _support("xgboost", True, "XGBoost CUDA available")


def test_lightgbm_falls_back_to_cpu_loudly_not_silently(monkeypatch):
    """A CPU fallback must appear in the run log, not be inferred from a timing anomaly."""
    monkeypatch.setattr(compute, "lightgbm_gpu_support", lambda: NO_LGB_CUDA)
    with pytest.warns(RuntimeWarning, match="CUDA Tree Learner"):
        params = compute.lightgbm_device_params("estimation.demand")
    assert params["device_type"] == "cpu"


def test_lightgbm_cpu_fallback_stays_deterministic(monkeypatch):
    """Reproducibility is the actual guarantee, and it is not a GPU-only concern."""
    monkeypatch.setattr(compute, "lightgbm_gpu_support", lambda: NO_LGB_CUDA)
    with pytest.warns(RuntimeWarning):
        params = compute.lightgbm_device_params()
    assert params["deterministic"] is True
    assert params["force_row_wise"] is True


def test_working_torch_cuda_does_not_imply_lightgbm_cuda(monkeypatch):
    """Regression: the probe must ask LightGBM about LightGBM.

    An earlier version asked *torch* whether CUDA existed and then handed
    ``device_type: cuda`` to LightGBM. On a machine with a CUDA torch and a stock
    PyPI LightGBM wheel that fails at training time with "CUDA Tree Learner was
    not enabled in this build" — a bug masked entirely by having a CPU-only
    torch installed, so that repairing the torch install was what would have
    broken the demand model.
    """
    monkeypatch.setattr(compute, "gpu_report", lambda: AVAILABLE)
    monkeypatch.setattr(compute, "lightgbm_gpu_support", lambda: NO_LGB_CUDA)
    with pytest.warns(RuntimeWarning):
        params = compute.lightgbm_device_params()
    assert params["device_type"] == "cpu", (
        "torch CUDA availability must not be used as a proxy for LightGBM CUDA support"
    )


def test_xgboost_falls_back_to_cpu_loudly(monkeypatch):
    monkeypatch.setattr(compute, "xgboost_gpu_support", lambda: NO_XGB_CUDA)
    with pytest.warns(RuntimeWarning):
        assert compute.xgboost_device_params()["device"] == "cpu"


# ---------------------------------------------------------------------------
# GPU present
# ---------------------------------------------------------------------------


def test_lightgbm_params_pin_to_cuda_and_stay_deterministic(monkeypatch):
    monkeypatch.setattr(compute, "lightgbm_gpu_support", lambda: LGB_CUDA)
    params = compute.lightgbm_device_params()
    assert params["device_type"] == "cuda"
    assert params["deterministic"] is True


def test_xgboost_params_pin_to_cuda(monkeypatch):
    monkeypatch.setattr(compute, "xgboost_gpu_support", lambda: XGB_CUDA)
    assert compute.xgboost_device_params()["device"] == "cuda"


def test_compute_report_separates_machine_from_backend():
    """A green device with a red backend is the state that used to read as 'GPU enabled'."""
    report = compute.compute_report()
    assert "device" in report and "backends" in report
    assert {"lightgbm", "xgboost"} <= set(report["backends"])
    for row in report["backends"].values():
        assert isinstance(row["available"], bool)
        assert row["reason"].strip(), "an unavailable backend must say why"


class _FakeDevice:
    def __init__(self, type_: str) -> None:
        self.type = type_


# ---------------------------------------------------------------------------
# The escape hatch is explicit and loud
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [("1", True), ("true", True), ("YES", True), ("0", False), ("", False), ("no", False)],
)
def test_escape_hatch_parsing(monkeypatch, value, expected):
    monkeypatch.setenv(ALLOW_CPU_ENV_VAR, value)
    assert cpu_fallback_allowed() is expected


def test_escape_hatch_warns_loudly_when_used(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(compute, "gpu_report", lambda: UNAVAILABLE)
    monkeypatch.setenv(ALLOW_CPU_ENV_VAR, "1")
    with pytest.warns(RuntimeWarning, match="running on CPU"):
        device = compute.require_gpu("estimation.retention")
    assert device.type == "cpu"


def test_no_fallback_without_the_env_var(monkeypatch):
    monkeypatch.setattr(compute, "gpu_report", lambda: UNAVAILABLE)
    monkeypatch.setenv(ALLOW_CPU_ENV_VAR, "0")
    with pytest.raises(GPUUnavailableError):
        compute.require_gpu("decision.optimiser")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_gpu_report_never_raises():
    """Diagnostics must work on any machine, with or without torch installed."""
    info = compute.gpu_report()
    assert isinstance(info, GPUInfo)
    assert isinstance(info.available, bool)
    assert info.reason


def test_gpu_report_distinguishes_a_cpu_only_torch_build():
    """A CPU-only wheel and a missing driver need different fixes, so they get
    different messages."""
    info = compute.gpu_report()
    if not info.available and info.torch_version and info.cuda_version is None:
        assert "CPU-only build" in info.reason
        assert "download.pytorch.org" in info.reason


def test_gpu_info_serialises_for_the_artefact_record():
    assert AVAILABLE.as_dict()["device_name"] == "Simulated RTX"
