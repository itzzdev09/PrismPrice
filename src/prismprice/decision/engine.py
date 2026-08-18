"""
Decision engine (L3) — the request path, end to end.

Turns a :class:`~prismprice.governance.schemas.PriceRequest` into an immutable
:class:`~prismprice.governance.schemas.DecisionRecord`, by way of:

    ladder -> simulate -> score J -> guardrail filter -> pick -> record

**The order is the design.** Scoring happens first and guardrails second, so a
model bug can produce a wrong *score* but never an illegal *price*: the feasible
set is computed by L4 from the request, not by the optimiser from its own
beliefs. An engine that filtered inside the objective could be talked out of a
constraint by a large enough J, which is precisely what a guardrail exists to
prevent.

**The estimators arrive as callables, not as model objects.** ``demand_at`` and
``delta_clv_at`` are ``price -> value`` functions, so the engine has no import
on LightGBM or torch, is testable against closed-form demand curves where the
right answer is known, and cannot be tempted to reach past the interface into a
model's internals. It also means degradation is a substitution — swap the
elasticity-derived demand curve for a pooled one — rather than a branch.

**Every recommendation is reconstructible.** The seed, the policy version, the
model versions, the inputs, every candidate's score and every guardrail verdict
go into the record. Given the record and the pinned artefacts, the number can be
rebuilt months later; that is the whole point of writing one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import numpy as np

from prismprice import config
from prismprice.decision.ladder import generate_ladder
from prismprice.decision.objective import CandidateOutcome, ObjectiveWeights, score_candidate
from prismprice.governance.guardrails import GuardrailEngine
from prismprice.governance.schemas import (
    CandidateScore,
    DecisionEstimates,
    DecisionInputs,
    DecisionRecord,
    DegradationRung,
    DemandQuantiles,
    PriceRequest,
)

__all__ = [
    "DecisionEngine",
    "DecisionOutcome",
    "competitor_match_price",
    "cost_plus_price",
]

DemandAt = Callable[[float], tuple[float, float, float]]
DeltaCLVAt = Callable[[float], float]


@dataclass(frozen=True)
class DecisionOutcome:
    """The recommendation, plus the working that produced it."""

    record: DecisionRecord
    outcomes: tuple[CandidateOutcome, ...]
    ladder: tuple[float, ...]
    feasible_prices: tuple[float, ...]

    @property
    def recommended_price(self) -> float:
        return self.record.recommended_price

    @property
    def degraded(self) -> bool:
        return self.record.degradation_reason_code is not DegradationRung.OK_OPTIMAL

    def best_outcome(self) -> CandidateOutcome | None:
        """The scored outcome for the recommended price, if it was scored.

        Returns ``None`` on the fallback path, where the recommendation is the
        previous price rather than a candidate the objective chose — a caller
        that assumes a score exists would otherwise read a fallback as an
        optimum.
        """
        return next(
            (o for o in self.outcomes if abs(o.price - self.recommended_price) < 1e-9), None
        )


def cost_plus_price(unit_cost: float, margin_pct: float = 0.35) -> float:
    """Baseline: mark cost up by a fixed percentage.

    The most common real pricing policy, and the first thing a new system has to
    beat to justify itself. It ignores demand entirely, so it leaves money on
    the table wherever buyers are insensitive and overprices wherever they are
    not.
    """
    if unit_cost <= 0:
        raise ValueError(f"unit_cost must be > 0, got {unit_cost}")
    return unit_cost * (1.0 + margin_pct)


def competitor_match_price(competitor_price: float, undercut_pct: float = 0.0) -> float:
    """Baseline: track the competitor, optionally undercutting.

    The second most common policy, and the one that starts price wars: it has no
    view on demand or margin, so it will follow a rival below cost.
    """
    if competitor_price <= 0:
        raise ValueError(f"competitor_price must be > 0, got {competitor_price}")
    return competitor_price * (1.0 - undercut_pct)


@dataclass
class DecisionEngine:
    """Scores a ladder, applies guardrails, and emits an audit record.

    Args:
        weights: Objective dials.
        guardrails: Guardrail engine. Defaults to the standard nine.
        n_candidates: Ladder size before snapping.
        n_draws: Monte-Carlo draws per candidate.
        policy_version: Recorded on every decision; a record that defaults its
            own version is not reconstructible.
        seed: Base seed. Each candidate is scored with a seed derived from it
            and the candidate index, so candidates are independent yet the whole
            decision reproduces exactly.
    """

    weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    guardrails: GuardrailEngine = field(default_factory=GuardrailEngine)
    n_candidates: int = 9
    n_draws: int = config.DEFAULT_MONTE_CARLO_DRAWS
    policy_version: str = "0.1.0"
    model_versions: dict[str, str] = field(default_factory=dict)
    seed: int = config.DEFAULT_SEED

    def decide(
        self,
        request: PriceRequest,
        demand_at: DemandAt,
        delta_clv_at: DeltaCLVAt | None = None,
        units_available: float | None = None,
        elasticity: Any = None,
    ) -> DecisionOutcome:
        """Recommend a price for *request*.

        Args:
            request: The decision context. Carries no customer attribute; see
                the fairness guardrail.
            demand_at: ``price -> (p10, p50, p90)`` units.
            delta_clv_at: ``price -> delta CLV``. Omitted means the relationship
                term is zero, which is a *choice to price transactionally* and
                is recorded as such rather than treated as missing data.
            units_available: Stock cap for the simulation.
            elasticity: Optional estimate, recorded on the decision.

        Returns:
            :class:`DecisionOutcome`.
        """
        ladder = generate_ladder(
            current_price=request.current_price,
            movement_cap_pct=request.movement_cap_pct,
            n_candidates=self.n_candidates,
            allowed_endings=request.allowed_price_endings,
        )

        outcomes = tuple(
            score_candidate(
                price=price,
                unit_cost=request.unit_cost,
                demand_quantiles=demand_at(price),
                delta_clv=delta_clv_at(price) if delta_clv_at else 0.0,
                inventory_shadow_price=request.inventory_shadow_price,
                weights=self.weights,
                n_draws=self.n_draws,
                # COMMON RANDOM NUMBERS: every candidate is scored against the
                # *same* draws. This is not a shortcut, it is the difference
                # between ranking prices and ranking noise. Measured on a
                # constant-elasticity curve, Monte-Carlo error on one candidate
                # has SD ~10.8 while adjacent ladder rungs differ by only 5-10,
                # so with independent seeds the argmax was driven by sampling
                # error and disagreed with the analytic optimum. Sharing the
                # draws makes each comparison paired, so the noise largely
                # cancels in the difference and the ordering tracks the true
                # profit curve. It also makes the whole decision reproduce
                # exactly from the recorded seed.
                rng=np.random.default_rng(self.seed),
                units_available=units_available,
            )
            for price in ladder
        )

        evaluation = self.guardrails.evaluate_ladder(ladder, request)
        feasible = set(evaluation.feasible_prices)

        if evaluation.fallback_price is not None:
            recommended = evaluation.fallback_price
            rung = evaluation.degradation_rung
        else:
            scored = [o for o in outcomes if o.price in feasible]
            best = max(scored, key=lambda o: o.j_score)
            recommended = best.price
            rung = evaluation.degradation_rung

        record = self._build_record(
            request=request,
            ladder=ladder,
            outcomes=outcomes,
            feasible=feasible,
            recommended=recommended,
            rung=rung,
            evaluation=evaluation,
            demand_at=demand_at,
            delta_clv_at=delta_clv_at,
            elasticity=elasticity,
        )

        return DecisionOutcome(
            record=record,
            outcomes=outcomes,
            ladder=tuple(ladder),
            feasible_prices=tuple(sorted(feasible)),
        )

    def _build_record(
        self,
        request: PriceRequest,
        ladder: list[float],
        outcomes: tuple[CandidateOutcome, ...],
        feasible: set[float],
        recommended: float,
        rung: DegradationRung,
        evaluation: Any,
        demand_at: DemandAt,
        delta_clv_at: DeltaCLVAt | None,
        elasticity: Any,
    ) -> DecisionRecord:
        by_price = {e.price: e for e in evaluation.evaluations}

        candidates = [
            CandidateScore(
                price=outcome.price,
                j_score=outcome.j_score,
                cvar=outcome.cvar,
                cvar_alpha=outcome.cvar_alpha,
                feasible=outcome.price in feasible,
                binding_constraints=list(
                    by_price[outcome.price].binding_constraints if outcome.price in by_price else []
                ),
            )
            for outcome in outcomes
        ]

        chosen_evaluation = by_price.get(recommended)

        # The recorded quantiles are the model's own, not a summary of the
        # simulation. Reporting the simulated mean three times would make the
        # record claim a certainty the forecast never had.
        p10, p50, p90 = demand_at(recommended)
        estimates = DecisionEstimates(
            demand_quantiles=DemandQuantiles(p10=p10, p50=p50, p90=p90),
            causal_elasticity=elasticity,
            delta_clv=(delta_clv_at(recommended) if delta_clv_at else None),
        )

        # degradation_rung is deliberately not passed: DecisionRecord derives it
        # from degradation_reason_code in a `mode="before"` validator, and the
        # schema asks callers to supply it only when round-tripping a serialised
        # record. Passing it here to satisfy the type checker would put the
        # number and the code in two places and let them drift.
        return DecisionRecord(  # type: ignore[call-arg]
            decision_id=str(uuid4()),
            sku=request.sku,
            as_of=request.as_of,
            created_at=datetime.now(timezone.utc),
            policy_version=self.policy_version,
            model_versions=dict(self.model_versions),
            inputs=DecisionInputs.from_request(request),
            estimates=estimates,
            candidates=candidates,
            recommended_price=recommended,
            binding_constraints=(
                list(chosen_evaluation.binding_constraints) if chosen_evaluation else []
            ),
            guardrail_results=(list(chosen_evaluation.results) if chosen_evaluation else []),
            degradation_reason_code=rung,
            seed=self.seed,
        )
