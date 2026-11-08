"""
Model store abstraction for the serving layer (L6).

The service must keep answering when the models do not. That is not a niceties
requirement: a pricing API that returns a 500 leaves the caller with no price at
all, and whatever they do next — retry, cache, guess, fall back to a hardcoded
markup — happens outside the audit trail. Returning the previous price with
``CRITICAL_MAINTAIN_PREV`` keeps the failure inside the system, where it is
recorded and can be alerted on.

The store distinguishes missing request context from failed forecasts.
:class:`ModelStoreUnavailable` means neither model has usable inputs, so the
service holds the supplied current price. :class:`ModelPredictionUnavailable`
means a forecast failed; the service tries a separately validated standby
before holding.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from prismprice.governance.schemas import DecisionEvidence, PriceRequest

__all__ = [
    "DecisionModelContext",
    "DecisionModelFallback",
    "InMemoryModelStore",
    "ModelStore",
    "ModelPredictionUnavailable",
    "ModelStoreUnavailable",
    "UnavailableModelStore",
]


class ModelStoreUnavailable(RuntimeError):
    """Required model artefacts or point-in-time request context are unavailable.

    Deliberately distinct from a forecast failure. Missing or unsupported
    features cannot be repaired by another demand model, so the service holds
    the supplied current price.
    """


class ModelPredictionUnavailable(RuntimeError):
    """A model failed while forecasting, allowing a validated standby to run."""


class ModelStore(Protocol):
    """What the decision service needs from the model layer."""

    def demand_at(self, sku: str, price: float) -> tuple[float, float, float]:
        """``(p10, p50, p90)`` units for *sku* at *price*."""
        ...

    def delta_clv_at(self, sku: str, price: float) -> float:
        """Change in discounted future margin from pricing *sku* at *price*."""
        ...

    def model_versions(self) -> dict[str, str]:
        """Artefact versions, recorded on every decision for reconstructibility."""
        ...


@dataclass(frozen=True)
class DecisionModelFallback:
    """Forecast functions and provenance for a separately validated standby."""

    demand_at: Callable[[float], tuple[float, float, float]]
    demand_at_prices: Callable[[list[float]], list[tuple[float, float, float]]]
    evidence: DecisionEvidence
    model_versions: dict[str, str]


@dataclass(frozen=True)
class DecisionModelContext:
    """Request-scoped model inputs prepared from point-in-time features."""

    demand_at: Callable[[float], tuple[float, float, float]]
    demand_at_prices: Callable[[list[float]], list[tuple[float, float, float]]] | None = None
    delta_clv_at: Callable[[float], float] | None = None
    evidence: DecisionEvidence | None = None
    elasticity: object | None = None
    model_versions: dict[str, str] = field(default_factory=dict)
    fallback: DecisionModelFallback | None = None
    fallback_on_load: str | None = None


@dataclass
class InMemoryModelStore:
    """A store backed by plain callables.

    Used by the tests and by anyone wiring a fitted model in. ``available`` is
    the switch the chaos test flips: setting it ``False`` makes every lookup
    raise :class:`ModelStoreUnavailable`, which is the only honest way to test a
    degradation path — mocking the *response* would test the mock, not the
    service's behaviour when the dependency is genuinely gone.
    """

    demand: Callable[[str, float], tuple[float, float, float]]
    clv: Callable[[str, float], float] | None = None
    versions: dict[str, str] = field(default_factory=dict)
    available: bool = True

    def context_for(self, request: PriceRequest) -> DecisionModelContext:
        """Adapt the simple callable store to the request-scoped service path."""
        self._require_available()
        return DecisionModelContext(
            demand_at=lambda price: self.demand_at(request.sku, price),
            delta_clv_at=lambda price: self.delta_clv_at(request.sku, price),
            model_versions=dict(self.versions),
        )

    def demand_at(self, sku: str, price: float) -> tuple[float, float, float]:
        self._require_available()
        return self.demand(sku, price)

    def delta_clv_at(self, sku: str, price: float) -> float:
        self._require_available()
        return self.clv(sku, price) if self.clv else 0.0

    def model_versions(self) -> dict[str, str]:
        # Checks availability like every other accessor. Without this the health
        # endpoint reports a store that died after startup as healthy, which is
        # worse than having no health check at all: an operator reads green
        # while every decision degrades to rung 5.
        self._require_available()
        return dict(self.versions)

    def _require_available(self) -> None:
        if not self.available:
            raise ModelStoreUnavailable("model store is unavailable")


@dataclass(frozen=True)
class UnavailableModelStore:
    """Explicit offline state when a required, verified bundle cannot load."""

    reason: str

    def demand_at(self, sku: str, price: float) -> tuple[float, float, float]:
        raise ModelStoreUnavailable(self.reason)

    def delta_clv_at(self, sku: str, price: float) -> float:
        raise ModelStoreUnavailable(self.reason)

    def model_versions(self) -> dict[str, str]:
        raise ModelStoreUnavailable(self.reason)

    def workbench_context(self) -> dict[str, object]:
        return {
            "model_store_available": False,
            "unavailable_reason": self.reason,
            "model_name": None,
            "model_version": None,
            "dataset_name": None,
            "data_through": None,
            "as_of": None,
            "demand_unit": None,
            "supported_skus": [],
            "validation": None,
            "assumptions": [],
            "capabilities": {
                "demand_forecast": "unavailable",
                "causal_elasticity": "not_connected",
                "retention_clv": "not_connected",
                "portfolio_substitution": "not_connected",
                "live_costs_inventory_competitors": "not_connected",
                "durable_decisions": "not_connected",
                "price_publishing": "not_implemented",
            },
        }
