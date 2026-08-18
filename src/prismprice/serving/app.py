"""
FastAPI application (L6).

Thin on purpose. Every decision lives in :mod:`prismprice.serving.service`; the
routes translate HTTP to Python and back, and nothing else.

Two contract choices worth stating, because both look like bugs from outside:

**A degraded decision is still a 200.** Rung 5 means the models are unreachable
and the service is holding the previous price — that is a *successful*
recommendation of "do not move", not a server error. Returning 5xx would push
the caller into an uninstrumented fallback of their own, which is the outcome
the degradation matrix exists to prevent. The rung is in the body, and a caller
who wants to alert on it reads ``degradation_reason_code``.

**Batch never partially fails.** One outcome per input, always, in order. A
catalogue refresh that returns nothing because one artefact is missing is how
every price silently goes stale overnight.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from prismprice.governance.schemas import PriceRequest
from prismprice.serving.service import DecisionService

__all__ = ["BatchRequest", "create_app"]


class BatchRequest(BaseModel):
    """Payload for the batch endpoint.

    Defined at module scope, not inside :func:`create_app`. Nested in the
    factory it becomes an unresolvable ForwardRef under
    ``from __future__ import annotations`` — FastAPI builds its validator from
    the string annotation and cannot see a class local to a function, which
    surfaces as a PydanticUserError at request time and takes /openapi.json
    down with it.

    The cap is a real limit, not a formality: a batch is scored
    synchronously, so an unbounded list is a request that never returns.
    """

    requests: list[PriceRequest] = Field(..., min_length=1, max_length=1000)


def create_app(service: DecisionService) -> Any:
    """Build the API around an already-wired *service*.

    A factory rather than a module-level ``app`` so the service (and therefore
    the model store) is injected. Tests get to run against a store they control,
    and nothing reaches for a global at import time.
    """
    from fastapi import FastAPI

    application = FastAPI(
        title="PrismPrice",
        version=service.engine.policy_version,
        description=(
            "Pricing decision support. Recommendations are not auto-published "
            "prices: every response carries its assumptions, its uncertainty, "
            "the constraints that bound it, and a reason code."
        ),
    )

    @application.get("/health")
    def health() -> dict[str, Any]:
        return service.health().as_dict()

    @application.post("/decide")
    def decide(request: PriceRequest) -> dict[str, Any]:
        outcome = service.decide(request)
        return _response(outcome)

    @application.post("/decide/batch")
    def decide_batch(payload: BatchRequest) -> dict[str, Any]:
        outcomes = service.decide_batch(payload.requests)
        return {
            "count": len(outcomes),
            "decisions": [_response(outcome) for outcome in outcomes],
        }

    return application


def _response(outcome: Any) -> dict[str, Any]:
    """Serialise a decision for the wire.

    The full record goes out, not just the price. A recommendation without its
    reason code, binding constraints and candidate scores is a number the caller
    has to trust on faith, and this system's entire claim is that they should
    not have to.
    """
    record = outcome.record
    return {
        "decision_id": record.decision_id,
        "sku": record.sku,
        "recommended_price": record.recommended_price,
        "degradation_reason_code": record.degradation_reason_code.value,
        "degradation_rung": record.degradation_rung,
        "binding_constraints": [code.value for code in record.binding_constraints],
        "policy_version": record.policy_version,
        "seed": record.seed,
        "record": record.model_dump(mode="json"),
    }
