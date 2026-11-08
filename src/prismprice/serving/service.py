"""
Decision service (L6) — the layer between HTTP and the engine.

Kept deliberately separate from the FastAPI app. Everything that decides
anything lives here and is testable by calling a function; ``app.py`` only
translates HTTP to Python and back. A service whose behaviour can only be
reached through a test client tends to grow logic in its route handlers, and
route handlers are the one place nobody writes unit tests for.

The whole point of this layer is the **degradation matrix**. Rungs 1, 2 and 4
are decided inside the engine and its guardrails; this module owns standby-model
failover and rung 5, which only exists because a running process has dependencies
that can disappear::

    1  OK_OPTIMAL                 full health
    2  WARN_STALE_COMPETITOR      competitor feed is old        (guardrail engine)
    3  FALLBACK_POOLED_ELASTICITY per-SKU elasticity untrusted  (estimation layer)
    3  FALLBACK_BACKUP_MODEL      primary forecast failed; standby used (here)
    4  FALLBACK_RULE_ENGINE       no feasible candidate         (guardrail engine)
    5  CRITICAL_MAINTAIN_PREV     no usable demand model        (here)

Rung 5 returns the **current price**, not a computed one. That is the point: the
system stops having an opinion and says so, rather than degrading to an opinion
it cannot justify. A caller that receives a 500 instead does something
uninstrumented and outside the audit log, which is strictly worse than holding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4

from prismprice.decision.engine import DecisionEngine, DecisionOutcome
from prismprice.governance.schemas import (
    DecisionInputs,
    DecisionRecord,
    DegradationRung,
    PriceRequest,
)
from prismprice.serving.models import (
    DecisionModelContext,
    ModelPredictionUnavailable,
    ModelStore,
    ModelStoreUnavailable,
)

__all__ = [
    "DecisionService",
    "DecisionPersistenceError",
    "ServiceHealth",
]


class DecisionRepository(Protocol):
    """Durable append-only sink for decisions returned by the service."""

    def record_decision(self, record: dict[str, Any]) -> None: ...


class DecisionPersistenceError(RuntimeError):
    """A scored decision could not be durably recorded."""


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
    repository: DecisionRepository | None = None

    def decide(self, request: PriceRequest) -> DecisionOutcome:
        """Recommend a price, degrading rather than failing.

        Forecast failures try a separately validated standby. If neither model
        can produce a valid forecast, or the request has no supported feature
        context, the service records a rung-5 hold. Unexpected faults propagate.
        """
        context: DecisionModelContext | None = None
        try:
            context_for = getattr(self.store, "context_for", None)
            context = context_for(request) if context_for else None
            outcome = self._run_engine(request, context)
            if context is not None and context.fallback_on_load:
                outcome = self._mark_backup_used(
                    outcome,
                    "The primary model could not be loaded; the separately validated "
                    f"backup model was promoted. {context.fallback_on_load}",
                )
        except ModelPredictionUnavailable as primary_error:
            fallback = context.fallback if context is not None else None
            if fallback is None:
                outcome = self._maintain_previous(
                    request,
                    reason=(
                        "The primary demand model failed during forecasting and no "
                        f"validated backup was available: {primary_error}"
                    ),
                )
            else:
                try:
                    outcome = self._run_engine(request, context, use_fallback=True)
                except ModelPredictionUnavailable as backup_error:
                    outcome = self._maintain_previous(
                        request,
                        reason=(
                            "Both the primary and separately validated backup demand models "
                            f"failed during forecasting. Primary: {primary_error}. Backup: {backup_error}"
                        ),
                    )
                else:
                    outcome = self._mark_backup_used(
                        outcome,
                        "The primary model failed during forecasting; the separately "
                        f"validated backup model produced this decision. {primary_error}",
                    )
        except ModelStoreUnavailable as exc:
            outcome = self._maintain_previous(request, reason=str(exc))

        if self.repository is not None:
            try:
                self.repository.record_decision(outcome.record.model_dump(mode="json"))
            except Exception as exc:
                raise DecisionPersistenceError(
                    "The decision was scored but could not be saved to the audit store."
                ) from exc
        return outcome

    def _run_engine(
        self,
        request: PriceRequest,
        context: DecisionModelContext | None,
        *,
        use_fallback: bool = False,
    ) -> DecisionOutcome:
        fallback = context.fallback if context is not None and use_fallback else None
        if use_fallback and fallback is None:
            raise ValueError("No validated backup model is available for this decision context.")
        if fallback is not None:
            demand_at = fallback.demand_at
            demand_at_prices = fallback.demand_at_prices
            model_evidence = fallback.evidence
            model_versions = fallback.model_versions
        else:
            demand_at = (
                context.demand_at
                if context is not None
                else lambda price: self.store.demand_at(request.sku, price)
            )
            demand_at_prices = context.demand_at_prices if context is not None else None
            model_evidence = context.evidence if context is not None else None
            model_versions = None if context is None else context.model_versions or None
        return self.engine.decide(
            request,
            demand_at=demand_at,
            delta_clv_at=(
                context.delta_clv_at
                if context is not None and fallback is None
                else None
            ),
            elasticity=(
                context.elasticity
                if context is not None and fallback is None
                else None
            ),
            model_evidence=model_evidence,
            model_versions=model_versions,
            demand_at_prices=demand_at_prices,
        )

    @staticmethod
    def _mark_backup_used(outcome: DecisionOutcome, detail: str) -> DecisionOutcome:
        """Record standby use unless a more severe rule-engine hold already applies."""
        code = outcome.record.degradation_reason_code
        if code.level < DegradationRung.FALLBACK_RULE_ENGINE.level:
            code = DegradationRung.FALLBACK_BACKUP_MODEL
        existing_detail = outcome.record.degradation_detail
        if existing_detail and existing_detail not in detail:
            detail = f"{detail} Existing decision degradation: {existing_detail}"
        record = outcome.record.model_copy(
            update={
                "degradation_reason_code": code,
                "degradation_rung": code.level,
                "degradation_detail": detail,
            }
        )
        return DecisionOutcome(
            record=record,
            outcomes=outcome.outcomes,
            ladder=outcome.ladder,
            feasible_prices=outcome.feasible_prices,
        )

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

        available = self.model_store_available()

        return ServiceHealth(
            healthy=available,
            model_store_available=available,
            policy_version=self.engine.policy_version,
            compute=compute_report(),
        )

    def model_store_available(self) -> bool:
        """Probe model availability without collecting compute diagnostics."""
        try:
            self.store.model_versions()
        except ModelStoreUnavailable:
            return False
        return True

    def workbench_context(self) -> dict[str, Any]:
        """Describe connected model evidence and explicitly unavailable layers."""
        provider = getattr(self.store, "workbench_context", None)
        if provider is not None:
            context = provider()
            capabilities = context.setdefault("capabilities", {})
            capabilities["durable_decisions"] = (
                "connected" if self.repository is not None else "not_connected"
            )
            return context
        return {
            "model_store_available": self.model_store_available(),
            "model_name": None,
            "model_version": None,
            "dataset_name": None,
            "as_of": None,
            "supported_skus": [],
            "validation": None,
            "capabilities": {
                "demand_forecast": "unavailable",
                "causal_elasticity": "not_connected",
                "retention_clv": "not_connected",
                "portfolio_substitution": "not_connected",
                "live_costs_inventory_competitors": "request_inputs_only",
                "durable_decisions": "available" if self.repository else "not_connected",
                "price_publishing": "not_implemented",
            },
        }

    def _maintain_previous(
        self, request: PriceRequest, reason: str = "The demand model is unavailable."
    ) -> DecisionOutcome:
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
            degradation_detail=reason,
            seed=self.engine.seed,
        )
        return DecisionOutcome(record=record, outcomes=(), ladder=(), feasible_prices=())
