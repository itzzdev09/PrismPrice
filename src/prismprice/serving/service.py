"""
Decision service (L6) — the layer between HTTP and the engine.

Kept deliberately separate from the FastAPI app. Everything that decides
anything lives here and is testable by calling a function; ``app.py`` only
translates HTTP to Python and back. A service whose behaviour can only be
reached through a test client tends to grow logic in its route handlers, and
route handlers are the one place nobody writes unit tests for.

The whole point of this layer is the **degradation matrix**. Rungs 1, 2 and 4
are decided inside the engine and its guardrails; this module owns rung 5, the
one that only exists because the service is a running process with dependencies
that can disappear::

    1  OK_OPTIMAL                 full health
    2  WARN_STALE_COMPETITOR      competitor feed is old        (guardrail engine)
    3  FALLBACK_POOLED_ELASTICITY per-SKU elasticity untrusted  (estimation layer)
    4  FALLBACK_RULE_ENGINE       no feasible candidate         (guardrail engine)
    5  CRITICAL_MAINTAIN_PREV     model store unreachable       (here)

Rung 5 returns the **current price**, not a computed one. That is the point: the
system stops having an opinion and says so, rather than degrading to an opinion
it cannot justify. A caller that receives a 500 instead does something
uninstrumented and outside the audit log, which is strictly worse than holding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from prismprice.decision.engine import DecisionEngine, DecisionOutcome
from prismprice.governance.schemas import (
    DecisionInputs,
    DecisionRecord,
    DegradationRung,
    PriceRequest,
)
from prismprice.serving.models import ModelStore, ModelStoreUnavailable

__all__ = [
    "DecisionService",
    "ServiceHealth",
]


@dataclass(frozen=True)
class ServiceHealth:
    """What ``/health`` reports.

    ``model_store_available`` is probed rather than remembered, so a store that
    died since startup shows as dead instead of as whatever it was when the
    process booted.
    """

    healthy: bool
    model_store_available: bool
    policy_version: str
    compute: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "healthy": self.healthy,
            "model_store_available": self.model_store_available,
            "policy_version": self.policy_version,
            "compute": self.compute,
        }


@dataclass
class DecisionService:
    """Turns requests into decision records, and never raises on model failure.

    Args:
        store: Model artefacts. Allowed to be unavailable.
        engine: Decision engine. Defaults to the standard one.
    """

    store: ModelStore
    engine: DecisionEngine = field(default_factory=DecisionEngine)

    def decide(self, request: PriceRequest) -> DecisionOutcome:
        """Recommend a price, degrading rather than failing.

        Raises nothing for a model-store outage: the outage is expressed as a
        rung-5 record holding the current price. Any *other* exception is left
        to propagate, because an unexpected fault should be loud and is not
        something a held price can honestly paper over.
        """
        try:
            return self.engine.decide(
                request,
                demand_at=lambda price: self.store.demand_at(request.sku, price),
                delta_clv_at=lambda price: self.store.delta_clv_at(request.sku, price),
            )
        except ModelStoreUnavailable:
            return self._maintain_previous(request)

    def decide_batch(self, requests: list[PriceRequest]) -> list[DecisionOutcome]:
        """Score many SKUs.

        One SKU's failure must not take the batch with it. A catalogue refresh
        that returns nothing because a single artefact is missing is how a
        nightly job silently leaves every price stale, so each request degrades
        on its own and the batch always returns one outcome per input.
        """
        return [self.decide(request) for request in requests]

    def health(self) -> ServiceHealth:
        from prismprice.compute import compute_report

        available = True
        try:
            self.store.model_versions()
        except ModelStoreUnavailable:
            available = False

        return ServiceHealth(
            healthy=available,
            model_store_available=available,
            policy_version=self.engine.policy_version,
            compute=compute_report(),
        )

    def _maintain_previous(self, request: PriceRequest) -> DecisionOutcome:
        """Rung 5: hold the current price and record why.

        No candidates are recorded, because none were scored. Emitting a ladder
        here would make the record look like a decision that considered
        alternatives and chose to stand still, which is a different and much
        more reassuring event than the one that actually happened.
        """
        record = DecisionRecord(  # type: ignore[call-arg]
            decision_id=str(uuid4()),
            sku=request.sku,
            as_of=request.as_of,
            created_at=datetime.now(timezone.utc),
            policy_version=self.engine.policy_version,
            model_versions={},
            inputs=DecisionInputs.from_request(request),
            candidates=[],
            recommended_price=request.current_price,
            binding_constraints=[],
            guardrail_results=[],
            degradation_reason_code=DegradationRung.CRITICAL_MAINTAIN_PREV,
            seed=self.engine.seed,
        )
        return DecisionOutcome(record=record, outcomes=(), ladder=(), feasible_prices=())
