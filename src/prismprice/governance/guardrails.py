"""
Pure-predicate governance guardrails for price feasibility.

Three rules govern everything in this module:

**Guardrails are hard.** They run after scoring, so a model bug cannot produce an
illegal price. The only numeric tolerance permitted is an allowance for IEEE-754
representation error (``config.FLOAT_REL_TOL``); there is no business slack.

**Guardrails fail closed.** Where a constraint is regulatory and its input data
is missing, the predicate FAILS. The alternative — passing because there was
nothing to compare against — is how a non-compliant price reaches production.

**Guardrails never invent a verdict.** ``NOT_APPLICABLE`` is a distinct status
from ``PASSED``. An audit record must distinguish "checked, fine" from "nothing
to check", because only the first is evidence of compliance.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import ClassVar

from prismprice import config
from prismprice.governance.schemas import (
    PROTECTED_ATTRIBUTE_TOKENS,
    CandidateEvaluation,
    DegradationRung,
    GuardrailCode,
    GuardrailResult,
    GuardrailStatus,
    LadderEvaluation,
    PriceRequest,
)

__all__ = [
    "AbsoluteFloorGuardrail",
    "BaseGuardrail",
    "ChangeFrequencyGuardrail",
    "CompetitiveCeilingGuardrail",
    "EUOmnibusAnchorGuardrail",
    "FairnessGuardrail",
    "GuardrailEngine",
    "InventoryGuardGuardrail",
    "LadderComplianceGuardrail",
    "MarginFloorGuardrail",
    "MovementCapGuardrail",
    "at_least",
    "at_most",
    "default_guardrails",
]


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def at_least(value: float, limit: float) -> bool:
    """``value >= limit``, forgiving only float representation error.

    ``10.00 * 1.15`` is ``11.499999999999998``; a candidate of exactly ``11.50``
    must satisfy that floor. Nothing wider than that is forgiven.
    """
    return value >= limit or math.isclose(
        value, limit, rel_tol=config.FLOAT_REL_TOL, abs_tol=config.FLOAT_ABS_TOL
    )


def at_most(value: float, limit: float) -> bool:
    """``value <= limit``, forgiving only float representation error."""
    return value <= limit or math.isclose(
        value, limit, rel_tol=config.FLOAT_REL_TOL, abs_tol=config.FLOAT_ABS_TOL
    )


def _as_utc(moment: datetime) -> datetime:
    """Treat a naive timestamp as UTC so comparisons never raise."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseGuardrail(ABC):
    """A single hard constraint.

    Subclasses declare ``code`` and ``name`` as class attributes and implement
    :meth:`evaluate`. Implementations must be pure: same request, same verdict,
    no I/O, no clock reads beyond ``request.as_of``.
    """

    code: ClassVar[GuardrailCode]
    name: ClassVar[str]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if ABC in cls.__bases__:
            return
        for attr in ("code", "name"):
            if not hasattr(cls, attr):
                raise TypeError(f"{cls.__name__} must declare a class-level '{attr}'")

    @abstractmethod
    def evaluate(self, candidate_price: float, request: PriceRequest) -> GuardrailResult:
        """Return the verdict for *candidate_price* under *request*."""

    # -- result builders ---------------------------------------------------
    #
    # Shared so that slack keeps one sign convention everywhere: >= 0 feasible,
    # negative is the size of the violation, for both floors and ceilings.

    def _floor(
        self, observed: float, limit: float, unit: str = "", what: str = "Price"
    ) -> GuardrailResult:
        passed = at_least(observed, limit)
        return GuardrailResult(
            reason_code=self.code,
            constraint_name=self.name,
            status=GuardrailStatus.PASSED if passed else GuardrailStatus.FAILED,
            observed=observed,
            limit=limit,
            slack=observed - limit,
            detail=(
                f"{what} {observed:,.4f}{unit} {'meets' if passed else 'is below'} "
                f"floor {limit:,.4f}{unit}"
            ),
        )

    def _ceiling(
        self, observed: float, limit: float, unit: str = "", what: str = "Price"
    ) -> GuardrailResult:
        passed = at_most(observed, limit)
        return GuardrailResult(
            reason_code=self.code,
            constraint_name=self.name,
            status=GuardrailStatus.PASSED if passed else GuardrailStatus.FAILED,
            observed=observed,
            limit=limit,
            slack=limit - observed,
            detail=(
                f"{what} {observed:,.4f}{unit} {'is within' if passed else 'exceeds'} "
                f"ceiling {limit:,.4f}{unit}"
            ),
        )

    def _not_applicable(self, detail: str) -> GuardrailResult:
        return GuardrailResult(
            reason_code=self.code,
            constraint_name=self.name,
            status=GuardrailStatus.NOT_APPLICABLE,
            detail=detail,
        )

    def _verdict(
        self,
        passed: bool,
        detail: str,
        observed: float | None = None,
        limit: float | None = None,
        slack: float | None = None,
    ) -> GuardrailResult:
        return GuardrailResult(
            reason_code=self.code,
            constraint_name=self.name,
            status=GuardrailStatus.PASSED if passed else GuardrailStatus.FAILED,
            observed=observed,
            limit=limit,
            slack=slack,
            detail=detail,
        )


# ---------------------------------------------------------------------------
# PP-G001 .. PP-G009
# ---------------------------------------------------------------------------


class AbsoluteFloorGuardrail(BaseGuardrail):
    """PP-G001: never knowingly price below fully-loaded unit cost."""

    code = GuardrailCode.ABSOLUTE_FLOOR
    name = "Absolute Floor"

    def evaluate(self, candidate_price: float, request: PriceRequest) -> GuardrailResult:
        return self._floor(candidate_price, request.unit_cost)


class MarginFloorGuardrail(BaseGuardrail):
    """PP-G002: satisfy the category minimum margin over cost."""

    code = GuardrailCode.MARGIN_FLOOR
    name = "Margin Floor"

    def evaluate(self, candidate_price: float, request: PriceRequest) -> GuardrailResult:
        return self._floor(candidate_price, request.unit_cost * (1.0 + request.margin_floor_pct))


class EUOmnibusAnchorGuardrail(BaseGuardrail):
    """PP-G003: EU Omnibus Directive — a promotional price may not exceed the
    lowest price published in the trailing 30 days.

    Fails closed. A promotion with no 30-day price history is a data failure, and
    a data failure on a regulatory constraint is not a pass: publishing the price
    anyway is precisely the outcome the Directive penalises.
    """

    code = GuardrailCode.EU_OMNIBUS
    name = "EU Omnibus Anchor"

    def evaluate(self, candidate_price: float, request: PriceRequest) -> GuardrailResult:
        if not request.is_promo:
            return self._not_applicable(
                "Not a promotional price; EU Omnibus reference anchor does not apply"
            )

        if request.min_price_last_30d is None:
            return self._verdict(
                passed=False,
                detail=(
                    "Promotional price requested but the 30-day minimum price anchor "
                    "is missing. Failing closed: EU Omnibus compliance cannot be "
                    "demonstrated without the reference price."
                ),
                observed=candidate_price,
            )

        return self._ceiling(candidate_price, request.min_price_last_30d, what="Promo price")


class CompetitiveCeilingGuardrail(BaseGuardrail):
    """PP-G004: cap the price relative to the observed competitor price."""

    code = GuardrailCode.COMPETITIVE_CEILING
    name = "Competitive Ceiling"

    def evaluate(self, candidate_price: float, request: PriceRequest) -> GuardrailResult:
        if request.competitor_price is None:
            return self._not_applicable("No competitor price observed")
        ceiling = request.competitor_price * request.competitor_ceiling_multiplier
        return self._ceiling(candidate_price, ceiling)


class MovementCapGuardrail(BaseGuardrail):
    """PP-G005: bound the relative move from the current published price."""

    code = GuardrailCode.MOVEMENT_CAP
    name = "Movement Cap"

    def evaluate(self, candidate_price: float, request: PriceRequest) -> GuardrailResult:
        # current_price is constrained gt=0 by the schema, so this cannot divide by zero.
        rel_change = abs(candidate_price - request.current_price) / request.current_price
        return self._ceiling(
            rel_change, request.movement_cap_pct, unit="", what="Relative price change"
        )


class ChangeFrequencyGuardrail(BaseGuardrail):
    """PP-G006: at most N price changes per rolling window.

    Scores the *prospective* count — the change this decision would make — rather
    than the historical one, since the historical count is by construction always
    within budget.
    """

    code = GuardrailCode.CHANGE_FREQUENCY
    name = "Change Frequency"

    def evaluate(self, candidate_price: float, request: PriceRequest) -> GuardrailResult:
        if request.price_changes_in_window is None:
            return self._not_applicable("Price change history not supplied for the rolling window")

        if math.isclose(
            candidate_price,
            request.current_price,
            rel_tol=config.FLOAT_REL_TOL,
            abs_tol=config.FLOAT_ABS_TOL,
        ):
            return self._verdict(
                passed=True,
                detail="Candidate equals the current price; no change is consumed",
                observed=float(request.price_changes_in_window),
                limit=float(request.max_changes_per_window),
                slack=float(request.max_changes_per_window - request.price_changes_in_window),
            )

        prospective = request.price_changes_in_window + 1
        passed = prospective <= request.max_changes_per_window
        return self._verdict(
            passed=passed,
            detail=(
                f"This change would be number {prospective} in the trailing "
                f"{request.change_window_days}-day window; limit is "
                f"{request.max_changes_per_window}"
            ),
            observed=float(prospective),
            limit=float(request.max_changes_per_window),
            slack=float(request.max_changes_per_window - prospective),
        )


class InventoryGuardGuardrail(BaseGuardrail):
    """PP-G007: below the minimum days-of-cover threshold, the inventory shadow
    price binds — scarce stock must not be sold at a price that ignores its
    opportunity cost.

    Effective floor is ``unit_cost + nu``. Above the cover threshold the shadow
    price is zero by construction and the constraint reduces to PP-G001.
    """

    code = GuardrailCode.INVENTORY_GUARD
    name = "Inventory Guard"

    def evaluate(self, candidate_price: float, request: PriceRequest) -> GuardrailResult:
        if request.inventory_cover_days is None:
            return self._not_applicable("Inventory cover not supplied")

        if request.inventory_cover_days >= request.min_inventory_cover_days:
            # The quantity under test here is cover, not price: the shadow price
            # is zero by construction above the threshold, so slack must measure
            # days of headroom. Reporting price-vs-cost slack on a PASSED verdict
            # would break the "slack >= 0 iff passed" convention when the price
            # happens to be below cost — PP-G001 is what catches that case.
            return self._verdict(
                passed=True,
                detail=(
                    f"Cover {request.inventory_cover_days:.1f}d at or above threshold "
                    f"{request.min_inventory_cover_days:.1f}d; shadow price does not bind"
                ),
                observed=request.inventory_cover_days,
                limit=request.min_inventory_cover_days,
                slack=request.inventory_cover_days - request.min_inventory_cover_days,
            )

        if request.inventory_shadow_price == 0.0:
            return self._verdict(
                passed=at_least(candidate_price, request.unit_cost),
                detail=(
                    f"Cover {request.inventory_cover_days:.1f}d is below threshold "
                    f"{request.min_inventory_cover_days:.1f}d but no shadow price was "
                    "supplied, so the guard reduces to the absolute floor. Supply "
                    "inventory_shadow_price for this constraint to bind."
                ),
                observed=candidate_price,
                limit=request.unit_cost,
                slack=candidate_price - request.unit_cost,
            )

        floor = request.unit_cost + request.inventory_shadow_price
        result = self._floor(candidate_price, floor)
        return GuardrailResult(
            reason_code=result.reason_code,
            constraint_name=result.constraint_name,
            status=result.status,
            observed=result.observed,
            limit=result.limit,
            slack=result.slack,
            detail=(
                f"{result.detail} (cover {request.inventory_cover_days:.1f}d below "
                f"{request.min_inventory_cover_days:.1f}d; shadow price "
                f"{request.inventory_shadow_price:,.4f} applied)"
            ),
        )


class LadderComplianceGuardrail(BaseGuardrail):
    """PP-G008: the price must land on an allowed psychological ending."""

    code = GuardrailCode.LADDER_COMPLIANCE
    name = "Ladder Compliance"

    def evaluate(self, candidate_price: float, request: PriceRequest) -> GuardrailResult:
        allowed = request.allowed_price_endings
        if not allowed:
            return self._not_applicable("No price-ending policy configured")

        cents = round(candidate_price * 100) % 100
        passed = cents in allowed
        rendered = ", ".join(f".{e:02d}" for e in sorted(allowed))
        return self._verdict(
            passed=passed,
            detail=(
                f"Price {candidate_price:,.2f} ends .{cents:02d}; "
                f"{'allowed' if passed else 'allowed endings are ' + rendered}"
            ),
            observed=float(cents),
        )


class FairnessGuardrail(BaseGuardrail):
    """PP-G009: no personalisation on identity or any proxy for one.

    This is a structural check on the decision context, not a rubber stamp. It
    fails when the request model carries a field whose name matches a protected
    attribute or a known proxy, and when the model permits unvalidated extra
    fields — either of which would let customer identity influence a price.

    What it is not: a full algorithmic-fairness audit (README §10). It proves the
    decision context is identity-free; it does not prove outcomes are equitable.
    """

    code = GuardrailCode.FAIRNESS_CHECK
    name = "Fairness Non-Discrimination"

    def evaluate(self, candidate_price: float, request: PriceRequest) -> GuardrailResult:
        model = type(request)
        offending: list[str] = []

        for field_name in model.model_fields:
            for part in field_name.lower().split("_"):
                if any(part.startswith(token) for token in PROTECTED_ATTRIBUTE_TOKENS):
                    offending.append(field_name)
                    break

        extras = getattr(request, "model_extra", None) or {}
        if extras:
            offending.extend(f"{key} (unvalidated extra)" for key in extras)

        if model.model_config.get("extra") != "forbid":
            offending.append(
                f"{model.__name__}.model_config permits extra fields "
                "(extra != 'forbid'), so identity attributes could be smuggled in"
            )

        # slack is deliberately None: fairness is a boolean structural property,
        # not a distance to a numeric threshold. Reporting 0.0 would pin
        # CandidateEvaluation.min_slack to zero on every feasible candidate and
        # destroy its usefulness as "how close did we run to the tightest limit".
        if offending:
            return self._verdict(
                passed=False,
                detail=(
                    "Decision context carries identity attributes or proxies: "
                    + "; ".join(sorted(offending))
                ),
                observed=float(len(offending)),
                limit=0.0,
            )

        return self._verdict(
            passed=True,
            detail=(
                f"{model.__name__} exposes {len(model.model_fields)} fields, none matching "
                "a protected attribute or proxy, and forbids extras; price is a pure "
                "function of SKU-level context"
            ),
            observed=0.0,
            limit=0.0,
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def default_guardrails() -> list[BaseGuardrail]:
    """The full registry, in reason-code order.

    Every member of :class:`GuardrailCode` must appear here; ``test_guardrails``
    asserts it, so a code can never again exist without an implementation.
    """
    return [
        AbsoluteFloorGuardrail(),
        MarginFloorGuardrail(),
        EUOmnibusAnchorGuardrail(),
        CompetitiveCeilingGuardrail(),
        MovementCapGuardrail(),
        ChangeFrequencyGuardrail(),
        InventoryGuardGuardrail(),
        LadderComplianceGuardrail(),
        FairnessGuardrail(),
    ]


class GuardrailEngine:
    """Runs the registry over candidate prices and reports the feasible set."""

    def __init__(self, guardrails: Sequence[BaseGuardrail] | None = None) -> None:
        self.guardrails: list[BaseGuardrail] = (
            list(guardrails) if guardrails is not None else default_guardrails()
        )

    def evaluate_candidate(
        self, candidate_price: float, request: PriceRequest
    ) -> CandidateEvaluation:
        """Evaluate one candidate against every guardrail.

        Every predicate runs even after the first failure: the audit record needs
        the complete set of binding constraints, not just the earliest one.
        """
        results = [g.evaluate(candidate_price, request) for g in self.guardrails]
        binding = [r.reason_code for r in results if r.binding]
        return CandidateEvaluation(
            price=candidate_price,
            feasible=not binding,
            results=results,
            binding_constraints=binding,
        )

    def evaluate_ladder(
        self, candidate_prices: Sequence[float], request: PriceRequest
    ) -> LadderEvaluation:
        """Filter a candidate ladder to its feasible set.

        This is the L3 -> L4 interface. When no candidate is feasible the engine
        does not raise and does not relax a constraint: it returns the current
        price with ``FALLBACK_RULE_ENGINE``, per the degradation matrix.
        """
        evaluations = [self.evaluate_candidate(p, request) for p in candidate_prices]
        feasible = [e.price for e in evaluations if e.feasible]

        counts: dict[GuardrailCode, int] = {}
        for evaluation in evaluations:
            for code in evaluation.binding_constraints:
                counts[code] = counts.get(code, 0) + 1

        if not feasible:
            return LadderEvaluation(
                evaluations=evaluations,
                feasible_prices=[],
                binding_constraint_counts=counts,
                fallback_price=request.current_price,
                degradation_rung=DegradationRung.FALLBACK_RULE_ENGINE,
            )

        return LadderEvaluation(
            evaluations=evaluations,
            feasible_prices=feasible,
            binding_constraint_counts=counts,
            fallback_price=None,
            degradation_rung=self._health_rung(request),
        )

    def _health_rung(self, request: PriceRequest) -> DegradationRung:
        """Rung 2 when the competitor feed is stale, rung 1 otherwise."""
        if request.competitor_price is None or request.competitor_observed_at is None:
            return DegradationRung.OK_OPTIMAL
        age_hours = (
            _as_utc(request.as_of) - _as_utc(request.competitor_observed_at)
        ).total_seconds() / 3600.0
        if age_hours > config.COMPETITOR_STALENESS_HOURS:
            return DegradationRung.WARN_STALE_COMPETITOR
        return DegradationRung.OK_OPTIMAL
