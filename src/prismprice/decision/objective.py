"""
The objective function (L3).

Design principle #1 of this repository: *the objective is the product.* Models
are replaceable; the definition of a good price is not. It lives here, and
everything else feeds it::

    J(p) = E[q(p)] * (p - c - nu)  +  lambda * E[dCLV(p)]  -  gamma * shortfall(p)

Three terms, and each exists to stop a specific failure.

**Contribution** is the transaction. On its own it recommends the highest price
the guardrails permit whenever demand is inelastic, which is right for a
quarter and wrong for a business.

**lambda * dCLV** is the relationship. It is what makes a discount that loses
money today capable of winning, and a price rise that prints margin today
capable of losing. ``lambda`` is not fitted — it is an exchange rate between
cash now and modelled future value, solved from a stated trade in
:mod:`prismprice.provenance`.

**gamma * shortfall** is the risk. Expected profit is indifferent between a
certain 100 and a coin flip of 0 and 200, and a pricing system that publishes
across a whole category is not: the tail is correlated across SKUs, so the
portfolio does not diversify it away.

Why simulate rather than multiply
---------------------------------

The naive implementation is ``E[q] * (p - c)``, and it is wrong wherever the
margin depends on the draw. ``E[f(q)] != f(E[q])`` unless ``f`` is linear, and
the moment inventory caps a sale, a stockout truncates it, or the risk term
looks at a tail, ``f`` stops being linear. So demand is **sampled**, profit is
computed per draw, and the collapse to a single number happens once — which is
also the only way to get a CVaR at all, since a point forecast has no tail.

Sampling from three quantiles
-----------------------------

L2 hands over ``p10 / p50 / p90``, and the gap between them is deliberately
asymmetric: the demand model conformalises each tail separately because a
symmetric interval was mis-placed even when its width was right. A single
lognormal fitted to the spread would throw that away and quietly re-impose
symmetry on the one layer that worked to remove it. :func:`sample_demand`
therefore fits a **two-piece lognormal** — one sigma below the median, another
above — so the asymmetry L2 measured survives into the decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from prismprice import config

__all__ = [
    "CandidateOutcome",
    "ObjectiveWeights",
    "sample_demand",
    "score_candidate",
]

#: z at the 10th/90th percentile of the standard normal. The half-distance from
#: the median, which is what converts a quantile gap into a lognormal sigma.
_Z_90 = 1.2815515655446004


@dataclass(frozen=True)
class ObjectiveWeights:
    """The dials that define a good price.

    Defaults come from :mod:`prismprice.config`, where each is a ``POLICY``
    parameter carrying the trade it encodes and a sensitivity bracket. They are
    repo defaults, not any operator's elicited preferences.
    """

    clv_weight_lambda: float = config.DEFAULT_CLV_WEIGHT_LAMBDA
    cvar_weight_gamma: float = config.DEFAULT_CVAR_WEIGHT_GAMMA
    cvar_alpha: float = config.DEFAULT_CVAR_ALPHA

    def __post_init__(self) -> None:
        if self.clv_weight_lambda < 0:
            raise ValueError(
                f"lambda must be >= 0, got {self.clv_weight_lambda}; a negative weight "
                f"would reward destroying customer value"
            )
        if self.cvar_weight_gamma < 0:
            raise ValueError(
                f"gamma must be >= 0, got {self.cvar_weight_gamma}; a negative weight "
                f"would reward downside risk"
            )
        if not 0.0 < self.cvar_alpha < 1.0:
            raise ValueError(f"cvar_alpha must be in (0, 1), got {self.cvar_alpha}")


@dataclass(frozen=True)
class CandidateOutcome:
    """Everything the decision record needs about one candidate price."""

    price: float
    expected_units: float
    expected_profit: float
    delta_clv: float
    cvar: float
    """Mean profit in the worst ``cvar_alpha`` of draws. Reported in absolute
    terms, because a decision record has to be readable next to the expectation."""
    tail_shortfall: float
    """``expected_profit - cvar``, always >= 0. This is what gamma prices: the
    penalty must scale with how far the tail sits below the mean, not with the
    tail's absolute level, or a profitable SKU would be penalised for being big."""
    j_score: float
    n_draws: int
    cvar_alpha: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "price": self.price,
            "expected_units": self.expected_units,
            "expected_profit": self.expected_profit,
            "delta_clv": self.delta_clv,
            "cvar": self.cvar,
            "tail_shortfall": self.tail_shortfall,
            "j_score": self.j_score,
            "cvar_alpha": self.cvar_alpha,
        }


def sample_demand(
    p10: float,
    p50: float,
    p90: float,
    n_draws: int = config.DEFAULT_MONTE_CARLO_DRAWS,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Draw demand from a two-piece lognormal matched to three quantiles.

    Below the median the spread is set by ``p50/p10``; above it by ``p90/p50``.
    Fitting one sigma to the whole interval would re-impose the symmetry the
    demand model's per-tail conformalisation exists to avoid.

    Lognormal rather than normal because demand is multiplicative and
    non-negative: a normal fitted to a wide interval puts mass below zero, and
    clipping that at zero silently shifts the mean upward.

    Args:
        p10: 10th percentile of units. Values <= 0 are floored just above zero,
            since a lognormal cannot represent them and a zero-demand quantile
            is common for slow movers.
        p50: Median units. Must be > 0.
        p90: 90th percentile of units.
        n_draws: Number of samples.
        rng: Seeded generator. Reproducibility is a stated guarantee, so this
            should be passed explicitly by anything that logs a decision.

    Returns:
        ``n_draws`` non-negative demand samples.

    Raises:
        ValueError: if the median is not positive or the quantiles are out of
            order — a crossed interval means the upstream model is broken, and
            silently sorting it would hide that.
    """
    if p50 <= 0:
        raise ValueError(f"p50 must be > 0 to sample a lognormal, got {p50}")
    if not (p10 <= p50 <= p90):
        raise ValueError(
            f"quantiles must be ordered, got p10={p10}, p50={p50}, p90={p90}. "
            f"A crossed interval is an upstream defect, not something to sort away."
        )
    if n_draws < 1:
        raise ValueError(f"n_draws must be >= 1, got {n_draws}")

    generator = rng or np.random.default_rng(config.DEFAULT_SEED)
    floor = max(p10, 1e-9)

    log_median = np.log(p50)
    sigma_low = max((log_median - np.log(floor)) / _Z_90, 1e-9)
    sigma_high = max((np.log(max(p90, p50)) - log_median) / _Z_90, 1e-9)

    z = generator.standard_normal(n_draws)
    sigma = np.where(z < 0.0, sigma_low, sigma_high)
    draws: NDArray[np.float64] = np.exp(log_median + z * sigma)
    return draws


def score_candidate(
    price: float,
    unit_cost: float,
    demand_quantiles: tuple[float, float, float],
    delta_clv: float = 0.0,
    inventory_shadow_price: float = 0.0,
    weights: ObjectiveWeights | None = None,
    n_draws: int = config.DEFAULT_MONTE_CARLO_DRAWS,
    rng: np.random.Generator | None = None,
    units_available: float | None = None,
) -> CandidateOutcome:
    """Evaluate ``J(p)`` for one candidate price.

    Args:
        price: Candidate.
        unit_cost: Fully loaded unit cost.
        demand_quantiles: ``(p10, p50, p90)`` units at *price*.
        delta_clv: Change in discounted future margin from pricing here, from
            the retention layer. Negative for a price rise.
        inventory_shadow_price: ``nu``, the opportunity cost of selling a unit
            of scarce stock. Enters the margin, so scarcity raises the effective
            cost rather than being bolted on as a constraint.
        weights: Objective dials.
        n_draws: Monte-Carlo draws.
        rng: Seeded generator.
        units_available: Stock cap. When supplied, sales are
            ``min(demand, stock)`` per draw — the non-linearity that makes
            simulation necessary rather than decorative.

    Returns:
        :class:`CandidateOutcome`.

    Raises:
        ValueError: on a non-positive price.
    """
    if price <= 0:
        raise ValueError(f"price must be > 0, got {price}")

    dials = weights or ObjectiveWeights()
    p10, p50, p90 = demand_quantiles
    demand = sample_demand(p10, p50, p90, n_draws=n_draws, rng=rng)

    sold = demand if units_available is None else np.minimum(demand, units_available)
    unit_margin = price - unit_cost - inventory_shadow_price
    profit = sold * unit_margin

    expected_profit = float(np.mean(profit))
    cvar = _conditional_value_at_risk(profit, dials.cvar_alpha)
    shortfall = max(expected_profit - cvar, 0.0)

    j_score = (
        expected_profit + dials.clv_weight_lambda * delta_clv - dials.cvar_weight_gamma * shortfall
    )

    return CandidateOutcome(
        price=price,
        expected_units=float(np.mean(sold)),
        expected_profit=expected_profit,
        delta_clv=delta_clv,
        cvar=cvar,
        tail_shortfall=shortfall,
        j_score=float(j_score),
        n_draws=n_draws,
        cvar_alpha=dials.cvar_alpha,
    )


def _conditional_value_at_risk(profit: NDArray[np.float64], alpha: float) -> float:
    """Mean profit in the worst *alpha* fraction of draws.

    Uses the empirical tail rather than a fitted distribution: the whole reason
    to simulate is that the profit distribution is not analytic once inventory
    caps bind, and fitting a parametric tail to the output would undo that.

    The tail is taken as at least one draw, so a small ``alpha`` on a small
    sample degrades to the worst observed outcome rather than to ``nan``.
    """
    n_tail = max(1, int(np.floor(alpha * profit.shape[0])))
    worst = np.partition(profit, n_tail - 1)[:n_tail]
    return float(np.mean(worst))
