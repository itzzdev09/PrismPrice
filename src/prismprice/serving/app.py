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

import os
import secrets
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from prismprice.governance.schemas import PriceRequest
from prismprice.serving.service import DecisionService

__all__ = ["BatchRequest", "create_app"]

_WEB_DIR = Path(__file__).resolve().parents[1] / "web"


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

    model_config = {"extra": "forbid"}

    requests: list[PriceRequest] = Field(..., min_length=1, max_length=100)


def create_app(
    service: DecisionService,
    *,
    api_key: str | None = None,
    allow_anonymous: bool = False,
    trusted_hosts: Sequence[str] = ("localhost", "127.0.0.1"),
    expose_docs: bool = False,
) -> Any:
    """Build the API around an already-wired *service*.

    A factory rather than a module-level ``app`` so the service (and therefore
    the model store) is injected. Tests get to run against a store they control,
    and nothing reaches for a global at import time.
    """
    from fastapi import Depends, FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from fastapi.security import APIKeyHeader
    from fastapi.staticfiles import StaticFiles
    from starlette.middleware.trustedhost import TrustedHostMiddleware

    from prismprice.serving.security import RequestBodyLimitMiddleware, SecurityHeadersMiddleware

    configured_api_key = api_key if api_key is not None else os.getenv("PRISMPRICE_API_KEY")
    if not allow_anonymous:
        if not configured_api_key:
            raise RuntimeError(
                "Set PRISMPRICE_API_KEY or pass api_key when creating a network-accessible PrismPrice app."
            )
        if len(configured_api_key) < 32:
            raise RuntimeError("PRISMPRICE_API_KEY must contain at least 32 characters.")
    if not trusted_hosts:
        raise ValueError("trusted_hosts must contain at least one allowed host.")

    application = FastAPI(
        title="PrismPrice",
        version=service.engine.policy_version,
        description=(
            "Pricing decision support. Recommendations are not auto-published "
            "prices: every response carries its assumptions, its uncertainty, "
            "the constraints that bound it, and a reason code."
        ),
        docs_url="/docs" if expose_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if expose_docs else None,
    )
    application.add_middleware(RequestBodyLimitMiddleware, max_bytes=256 * 1024)
    application.add_middleware(SecurityHeadersMiddleware)
    application.add_middleware(TrustedHostMiddleware, allowed_hosts=list(trusted_hosts))

    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def require_api_key(token: str | None = Depends(api_key_header)) -> None:
        if allow_anonymous:
            return
        if (
            configured_api_key is None
            or token is None
            or not secrets.compare_digest(token, configured_api_key)
        ):
            raise HTTPException(
                status_code=401,
                detail="Missing or invalid API key.",
                headers={"WWW-Authenticate": "APIKey"},
            )

    @application.get("/", include_in_schema=False)
    def home() -> Any:
        return FileResponse(_WEB_DIR / "index.html")

    @application.get("/health")
    def health() -> dict[str, Any]:
        report = service.health()
        return {
            "healthy": report.healthy,
            "model_store_available": report.model_store_available,
            "policy_version": report.policy_version,
        }

    @application.get("/internal/health", dependencies=[Depends(require_api_key)])
    def internal_health() -> dict[str, Any]:
        return service.health().as_dict()

    @application.post("/decide", dependencies=[Depends(require_api_key)])
    def decide(request: PriceRequest) -> dict[str, Any]:
        outcome = service.decide(request)
        return _response(outcome)

    @application.post("/decide/batch", dependencies=[Depends(require_api_key)])
    def decide_batch(payload: BatchRequest) -> dict[str, Any]:
        outcomes = service.decide_batch(payload.requests)
        return {
            "count": len(outcomes),
            "decisions": [_response(outcome) for outcome in outcomes],
        }

    application.mount("/assets", StaticFiles(directory=_WEB_DIR), name="assets")

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
