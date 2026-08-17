"""
GPU-only compute policy.

These tests must pass on CPU-only CI runners, so GPU state is simulated by
patching :func:`prismprice.compute.gpu_report` rather than by requiring hardware.
"""

import pytest

from prismprice import compute
from prismprice.compute import GPUInfo, GPUUnavailableError
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


def test_lightgbm_params_refuse_to_train_on_cpu_silently(monkeypatch):
    """LightGBM warns and trains on CPU when its GPU build is absent; we don't."""
    monkeypatch.setattr(compute, "gpu_report", lambda: UNAVAILABLE)
    with pytest.raises(GPUUnavailableError):
        compute.lightgbm_device_params()


def test_xgboost_params_refuse_to_train_on_cpu_silently(monkeypatch):
    monkeypatch.setattr(compute, "gpu_report", lambda: UNAVAILABLE)
    with pytest.raises(GPUUnavailableError):
        compute.xgboost_device_params()


# ---------------------------------------------------------------------------
# GPU present
# ---------------------------------------------------------------------------


def test_lightgbm_params_pin_to_cuda_and_stay_deterministic(monkeypatch):
    monkeypatch.setattr(compute, "gpu_report", lambda: AVAILABLE)
    monkeypatch.setattr(compute, "require_gpu", lambda component="lightgbm": _FakeDevice("cuda"))
    params = compute.lightgbm_device_params()
    assert params["device_type"] == "cuda"
    assert params["deterministic"] is True


def test_xgboost_params_pin_to_cuda(monkeypatch):
    monkeypatch.setattr(compute, "require_gpu", lambda component="xgboost": _FakeDevice("cuda"))
    assert compute.xgboost_device_params()["device"] == "cuda"


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
