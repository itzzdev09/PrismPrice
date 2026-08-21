"""
Circuit breakers (L7).

Implements ``docs/metrics.md`` §6, and the distinction from everything else in
this layer is the whole point: **these halt publication, they do not notify
someone.** A dashboard that goes red at 2am has not stopped anything. A breaker
that refuses to publish has.

They exist because the guardrails cannot catch this class of fault. ``PP-G002``
checks one price against one cost and is right about every one of them; nothing
in a per-candidate predicate can see that *all ten thousand prices moved down
together*, because each individual move was legal. The failures worth halting a
run for are properties of the batch, not of any decision in it.

The most important breaker is the one that should never fire
------------------------------------------------------------

:func:`no_guardrail_violations` re-checks published prices against the
constraints that were supposed to have filtered them. If the pipeline is correct
it can never trip. It exists because "impossible" is a claim about code, and the
code is what would be wrong — a filter applied to the wrong list, a rounding
step after the check, a fallback path that skipped it. The check costs almost
nothing and is the only thing standing between a logic error and a published
illegal price.

Fail closed
-----------

A breaker that cannot evaluate returns ``HALT``, not ``PASS``. Missing data at
this layer means the system does not know whether it is safe, and "we could not
check" is not evidence of safety. This is the same rule ``PP-G003`` follows for
the Omnibus anchor and for the same reason: the cost of a wrongly halted run is
a delayed price, and the cost of a wrongly published one is not recoverable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

__all__ = [
    "BreakerResult",
    "BreakerStatus",
    "aggregate_movement",
    "evaluate_breakers",
    "no_guardrail_violations",
    "one_sided_movement",
]


class BreakerStatus(str, Enum):
    """Outcome of one breaker. ``HALT`` stops publication."""

    PASS = "PASS"
    HALT = "HALT"


@dataclass(frozen=True)
class BreakerResult:
    """One circuit breaker's verdict, with the number that caused it."""

    name: str
    status: BreakerStatus
    observed: float
    limit: float
    detail: str

    @property
    def halted(self) -> bool:
        return self.status is BreakerStatus.HALT

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "observed": self.observed,
            "limit": self.limit,
            "detail": self.detail,
            "halted": self.halted,
        }


def one_sided_movement(
    previous_prices: Sequence[float],
    new_prices: Sequence[float],
    limit: float = 0.30,
    tolerance: float = 1e-9,
) -> BreakerResult:
    """Halt when too much of the catalogue moves the same way at once.

    Every individual move can be legal, well-scored and correctly guardrailed,
    and the batch can still be a catastrophe: a bad cost feed or an inverted
    sign moves everything down together. No per-candidate check can see this,
    because the property belongs to the batch.

    **The base is the whole catalogue, not the prices that moved.** An earlier
    version divided by the movers, which makes the breaker useless: in any
    balanced run about half the movers go each way, so a 30% limit fires on
    every healthy batch. Against the catalogue the threshold means what §6
    intends — a third of everything heading the same way — and a run that
    touches twelve SKUs out of ten thousand and sends all twelve down correctly
    passes, because twelve coordinated moves are not a stampede. The count of
    movers is reported so a small uniform batch is still legible.
    """
    if len(previous_prices) != len(new_prices):
        return BreakerResult(
            name="one_sided_movement",
            status=BreakerStatus.HALT,
            observed=float("nan"),
            limit=limit,
            detail=(
                f"price series do not align ({len(previous_prices)} vs {len(new_prices)}); "
                f"cannot evaluate, so halting"
            ),
        )
    if not previous_prices:
        return BreakerResult(
            name="one_sided_movement",
            status=BreakerStatus.HALT,
            observed=float("nan"),
            limit=limit,
            detail="no prices supplied; cannot evaluate, so halting",
        )

    previous = np.asarray(previous_prices, dtype=float)
    new = np.asarray(new_prices, dtype=float)
    delta = new - previous
    moved = np.abs(delta) > tolerance

    if not moved.any():
        return BreakerResult(
            name="one_sided_movement",
            status=BreakerStatus.PASS,
            observed=0.0,
            limit=limit,
            detail="no prices moved",
        )

    total = float(previous.size)
    down = float(np.sum(delta < -tolerance)) / total
    up = float(np.sum(delta > tolerance)) / total
    worst = max(down, up)
    direction = "down" if down >= up else "up"

    halted = worst > limit
    return BreakerResult(
        name="one_sided_movement",
        status=BreakerStatus.HALT if halted else BreakerStatus.PASS,
        observed=worst,
        limit=limit,
        detail=(
            f"{worst:.1%} of the catalogue went {direction} "
            f"({int(moved.sum())} of {int(previous.size)} prices moved)"
            + (f", above the {limit:.0%} limit" if halted else "")
        ),
    )


def aggregate_movement(
    previous_prices: Sequence[float],
    new_prices: Sequence[float],
    weights: Sequence[float] | None = None,
    limit: float = 0.03,
) -> BreakerResult:
    """Halt when the weighted basket moves more than *limit* in one run.

    Weighted by volume where supplied, because a 20% cut on a SKU nobody buys
    and a 20% cut on the best-seller are not the same event, and an unweighted
    mean rates them identically.
    """
    if len(previous_prices) != len(new_prices) or not previous_prices:
        return BreakerResult(
            name="aggregate_movement",
            status=BreakerStatus.HALT,
            observed=float("nan"),
            limit=limit,
            detail="price series missing or misaligned; cannot evaluate, so halting",
        )

    previous = np.asarray(previous_prices, dtype=float)
    new = np.asarray(new_prices, dtype=float)

    if weights is None:
        weight = np.ones_like(previous)
    else:
        weight = np.asarray(weights, dtype=float)
        if weight.shape != previous.shape:
            return BreakerResult(
                name="aggregate_movement",
                status=BreakerStatus.HALT,
                observed=float("nan"),
                limit=limit,
                detail="weights do not align with prices; cannot evaluate, so halting",
            )

    basket_before = float(np.sum(weight * previous))
    if basket_before <= 0:
        return BreakerResult(
            name="aggregate_movement",
            status=BreakerStatus.HALT,
            observed=float("nan"),
            limit=limit,
            detail="basket value is not positive; cannot evaluate, so halting",
        )

    basket_after = float(np.sum(weight * new))
    movement = (basket_after - basket_before) / basket_before
    halted = abs(movement) > limit

    return BreakerResult(
        name="aggregate_movement",
        status=BreakerStatus.HALT if halted else BreakerStatus.PASS,
        observed=movement,
        limit=limit,
        detail=(
            f"basket moved {movement:+.2%}"
            + (f", beyond the +/-{limit:.0%} limit" if halted else "")
        ),
    )


def no_guardrail_violations(
    prices: Sequence[float],
    unit_costs: Sequence[float],
    margin_floor_pct: float = 0.0,
) -> BreakerResult:
    """Re-check published prices against the floor that already filtered them.

    **This should never fire.** It exists because "impossible" is a claim about
    code, and the code is exactly what would be wrong: a filter applied to the
    wrong list, a rounding step after the check, a fallback that skipped it. It
    costs a comparison per price and is the last thing between a logic error and
    a published illegal price.

    A trip is an incident, not an alert: the run halts, the batch is
    quarantined, and someone finds out why the guardrail layer was bypassed.
    """
    if len(prices) != len(unit_costs) or not prices:
        return BreakerResult(
            name="no_guardrail_violations",
            status=BreakerStatus.HALT,
            observed=float("nan"),
            limit=margin_floor_pct,
            detail="prices and costs missing or misaligned; cannot verify, so halting",
        )

    price_array = np.asarray(prices, dtype=float)
    floor = np.asarray(unit_costs, dtype=float) * (1.0 + margin_floor_pct)
    violations = int(np.sum(price_array < floor - 1e-9))

    return BreakerResult(
        name="no_guardrail_violations",
        status=BreakerStatus.HALT if violations else BreakerStatus.PASS,
        observed=float(violations),
        limit=0.0,
        detail=(
            f"{violations} published price(s) below the margin floor — this should be "
            f"impossible; quarantine the batch and find out how the guardrail was bypassed"
            if violations
            else f"all {len(prices)} prices satisfy the margin floor"
        ),
    )


def evaluate_breakers(results: Sequence[BreakerResult]) -> tuple[bool, list[BreakerResult]]:
    """Combine breakers into a single publish / halt decision.

    Any single ``HALT`` stops the run. Breakers are not scored, weighted or
    voted on: each one encodes a condition under which publishing is wrong, and
    a majority of conditions being fine does not make the remaining one
    acceptable.

    Returns:
        ``(may_publish, halting_results)``. The halting results are returned
        rather than logged so the caller can put every reason in one incident
        instead of discovering them one run at a time.
    """
    halting = [result for result in results if result.halted]
    return (not halting), halting
