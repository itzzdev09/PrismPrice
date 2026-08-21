"""
Candidate price ladder (L3).

The optimiser never searches a continuous price range. It scores a short list of
prices a retailer would actually publish, and this module builds that list.

Three properties, each of which changes the answer:

**The current price is always on the ladder.** An optimiser that cannot choose
"leave it alone" will move the price every single run, because some candidate
always scores a hair above the others once noise is included. Price churn is
itself a cost — it burns the movement-cap budget, spends change-frequency
headroom, and erodes the reference price the retention model is built on — so
"no change" has to be a candidate that can win on its merits.

**Candidates are snapped to publishable endings before scoring, not after.**
Scoring 31.37 and then rounding to 31.95 means the recommendation was optimised
for a price that was never going to be published. Snapping first costs nothing
and keeps the reported J attached to the price actually taken.

**The ladder is bounded by the movement cap, not by the guardrails.** Guardrails
are a filter applied after scoring (L4's job, and deliberately separate), but
generating candidates the movement cap will certainly reject just wastes
simulation draws and makes the feasible set look emptier than it is.
"""

from __future__ import annotations

import numpy as np

from prismprice import config

__all__ = [
    "endings_for_price",
    "generate_ladder",
    "snap_to_ending",
]


def snap_to_ending(price: float, allowed_endings: tuple[int, ...] | None) -> float:
    """Round *price* to the nearest permitted ending, in whole cents.

    Args:
        price: Unrounded price.
        allowed_endings: Permitted endings in whole cents, e.g. ``(95, 99)``.
            ``None`` disables snapping and rounds to the nearest cent.

    Returns:
        The nearest price with a permitted ending. Ties go to the lower price,
        which keeps the function deterministic and errs toward the customer.
    """
    if not allowed_endings:
        return round(price, 2)

    pounds = int(np.floor(price))
    candidates: list[float] = []
    for whole in (pounds - 1, pounds, pounds + 1):
        if whole < 0:
            continue
        candidates.extend(whole + ending / 100.0 for ending in allowed_endings)

    positive = [c for c in candidates if c > 0]
    if not positive:
        return round(price, 2)
    # min() on (distance, value) breaks ties toward the lower price.
    return min(positive, key=lambda c: (abs(c - price), c))


def endings_for_price(price: float) -> tuple[int, ...]:
    """Permitted price endings for *price*, banded by magnitude.

    A single ``.95/.99`` rule is a policy that quietly refuses to price cheap
    items. At GBP 1.25 with a 15% movement cap the permitted window is
    [1.06, 1.44] and the nearest allowed endings are 0.99 and 1.95 — **neither is
    inside it**, so the feasible set is empty and the price is held. On the real
    catalogue this put 145 of 443 SKUs on degradation rung 4, with a median held
    price of GBP 1.63 against GBP 4.13 for the ones that priced. The synthetic
    generator never showed it because its base prices were GBP 12-60.

    The rule that actually has to hold is that the ending grid be finer than the
    movement cap allows the price to travel: a +/-15% window at GBP 1 is 30p
    wide, so endings 96p apart cannot land in it. Real retailers band for exactly
    this reason — pennies and 49/99 at the low end, 95/99 further up — so the
    bands here encode an existing practice rather than inventing one.

    Args:
        price: The current price, which selects the band. Chosen from the
            current price rather than per candidate, so one ladder does not mix
            two ending policies.

    Returns:
        Endings in whole cents, ascending.
    """
    if price < 1.00:
        return (9, 19, 29, 39, 49, 59, 69, 79, 89, 99)
    if price < 5.00:
        return (25, 49, 75, 99)
    return (95, 99)


def generate_ladder(
    current_price: float,
    movement_cap_pct: float = config.DEFAULT_MOVEMENT_CAP_PCT,
    n_candidates: int = 9,
    allowed_endings: tuple[int, ...] | None = config.DEFAULT_ALLOWED_PRICE_ENDINGS,
    floor: float | None = None,
    ceiling: float | None = None,
) -> list[float]:
    """Build the candidate prices to score.

    Args:
        current_price: Today's published price. Always included in the result.
        movement_cap_pct: Bound on relative movement either side.
        n_candidates: Points across the span before snapping and de-duplication;
            the returned ladder is usually shorter, because snapping collapses
            neighbours.
        allowed_endings: Psychological endings in whole cents.
        floor: Hard lower bound (e.g. cost plus shadow price). Candidates below
            it are dropped rather than clipped — clipping would pile several
            candidates onto the floor and bias the ladder toward it.
        ceiling: Hard upper bound.

    Returns:
        Ascending, de-duplicated, publishable prices.

    Raises:
        ValueError: on a non-positive price, a negative cap, or fewer than one
            candidate.
    """
    if current_price <= 0:
        raise ValueError(f"current_price must be > 0, got {current_price}")
    if movement_cap_pct < 0:
        raise ValueError(f"movement_cap_pct must be >= 0, got {movement_cap_pct}")
    if n_candidates < 1:
        raise ValueError(f"n_candidates must be >= 1, got {n_candidates}")

    low = current_price * (1.0 - movement_cap_pct)
    high = current_price * (1.0 + movement_cap_pct)

    raw = np.linspace(low, high, n_candidates) if n_candidates > 1 else np.array([current_price])
    snapped = {snap_to_ending(float(p), allowed_endings) for p in raw}

    # "Leave it alone" must be able to win, so the current price joins the
    # ladder at its own publishable value even if the grid missed it.
    snapped.add(snap_to_ending(current_price, allowed_endings))

    prices = sorted(p for p in snapped if p > 0)
    if floor is not None:
        prices = [p for p in prices if p >= floor]
    if ceiling is not None:
        prices = [p for p in prices if p <= ceiling]

    if not prices:
        # Every candidate was excluded by the bounds. Returning the snapped
        # current price keeps the contract (a non-empty ladder) and lets L4
        # reject it explicitly, which is a legible outcome; an empty list would
        # surface downstream as an unexplained crash.
        return [snap_to_ending(current_price, allowed_endings)]
    return prices
