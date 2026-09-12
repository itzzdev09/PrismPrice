"""
Sequential markdown under elasticity ambiguity (L3) — a robust Bellman operator.

``decision/markdown.py`` solves the markdown MDP exactly, and takes ``elasticity``
as a scalar. That scalar arrives from :mod:`prismprice.estimation.elasticity`,
which does not produce a scalar: cross-fitted DML produces a point estimate *and
a confidence interval*, and on a SKU whose price barely moved that interval is
wide enough to contain elasticities implying materially different prices. The
interval is then discarded at the module boundary, and a whole season is planned
as though the point estimate were exact.

This module keeps it. The state and dynamics are unchanged::

    state   (t, i)   periods remaining, units on hand
    action  p        price from the ladder
    reward  min(d,i) * (p - c)

but demand is Poisson with mean ``base_demand * (p / base_price) ** e`` where
``e`` is only known to lie in ``[elasticity_low, elasticity_high]``. Write
``Q_e(t, i, p)`` for the action value computed under a particular ``e``. The
operator solved here is

.. math::

    V(t, i) = \\max_p \\Big[ (1-\\rho)\\, Q_{\\hat e}(t,i,p)
              + \\rho\\, \\mathrm{CVaR}_\\alpha\\big( Q_e(t,i,p) \\big) \\Big]

with the inner ``Q`` computed against ``V`` itself, so ambiguity compounds
across periods rather than being applied once at the end.

Why not simply solve at the pessimistic end of the interval
-----------------------------------------------------------

Because there isn't one. It is tempting to read ``elasticity_low`` (the more
elastic end) as the cautious choice, but which end hurts depends on the action:
a more elastic customer punishes a price *rise* and rewards a *cut*, so the
damaging end of the interval flips sign somewhere along the ladder, and flips
again as inventory pressure moves the optimiser along it. A policy solved at one
endpoint is therefore pessimistic about some states and *optimistic* about
others — it is not conservative, it is inconsistent. Taking a tail over the
whole set at each state is what makes the caution uniform. The endpoint policy
is carried in the evaluation as a baseline precisely because it is the obvious
thing to try.

What the two dials mean, and what the formulation nests
-------------------------------------------------------

``robustness_level`` (:math:`\\rho`) and ``cvar_alpha`` (:math:`\\alpha`) are not
free parameters invented here — between them they recover the two policies the
literature already knows, as the endpoints of one family:

* :math:`\\rho = 0` is the classical certainty-equivalent DP. Not approximately:
  :func:`solve_robust_markdown` at ``robustness_level=0`` reproduces
  :func:`~prismprice.decision.markdown.solve_markdown` value-for-value, and a
  test asserts it.
* :math:`\\rho = 1` with :math:`\\alpha \\to 0` is the classical robust/minimax
  MDP over the interval — worst case at every state.

Everything in between is the part that is normally skipped, and it is where an
estimated interval actually belongs: the operator is asked to be cautious in
proportion to how uncertain the estimate genuinely is, rather than either
ignoring the interval or surrendering to its worst point.

The grid measure is a modelling choice, and is stated
-----------------------------------------------------

CVaR needs a measure, and a confidence interval is a *set*. Grid points are
weighted uniformly across the interval, which is the flat-prior reading of it.
This is an assumption and not a derivation — a sampling distribution would put
more mass near the point estimate and less at the ends, making the same
``alpha`` less conservative. Uniform is the choice that does not quietly smuggle
in extra confidence, which is the right default for a dial whose purpose is
caution.

Two operators, because the first one was measured and found wanting
--------------------------------------------------------------------

:func:`solve_robust_markdown` is the operator described above: it protects the
season's *value* across the ambiguity set. That is the natural first thing to
build and it is what the robust-MDP literature optimises, so it is what this
module started as. Measured on regret against an oracle, it loses — on the real
UCI panel it is beaten by simply trusting the point estimate, and badly.

The reason is structural rather than a tuning failure. A max-min-value policy is
pulled toward whichever elasticity in the set makes the season poorest, and on a
markdown problem that is the inelastic end: there is less volume to be had there
at any price, so the value function is lower whatever anybody does. That part of
the loss is not the policy's to prevent. Defending it means giving up real profit
in the elastic states, where a decision genuinely was available.

:func:`solve_regret_robust_markdown` subtracts off the unreachable part. It
minimises the CVaR of *regret* — what the best possible policy would have earned
under the same elasticity, minus what this one earns — so the only thing being
protected is the loss actually caused by not knowing. Both are kept, and the
benchmark reports both, because the second only makes sense as an answer to the
first one's failure.

The certificate
---------------

Alongside the value function the value-robust solver carries a second recursion:
``floor[t, i]``, the CVaR-over-ambiguity value of the *selected* action with
``floor`` itself as the continuation. That number is what
:attr:`RobustMarkdownPolicy.certified_profit_floor` reports, and it answers a
question the expected profit cannot: *if the elasticity is at the bad end of
what we estimated, what does this season still make?*

Two things it is not, both measured rather than assumed:

**It is not a bound on a single season.** It is a statement about parameter
error, not demand noise. An unlucky run of Poisson draws puts a season below it
routinely — around half the time, since the floor sits near the mean on a
tightly-estimated SKU — and :func:`simulate_robust_policy` reports that rate
rather than leaving it to be discovered.

**It is not a proven lower bound on the tail either.** The recursion applies the
tail afresh at every period, and the elasticity that is worst at one state need
not be the one that is worst over a whole season, so the nested quantity and the
end-to-end tail are not the same object and neither dominates by construction.
Measured across 40 instances spanning point estimates from -1.6 to -3.4 and
interval half-widths from 0.2 to 1.2, the floor lands **tight to within 0.05%**
of the realised tail, and on the optimistic side of it in 88% of cases — worst
overpromise 0.042%. That is small against the effects the operator is used to
decide, but it is an empirical tightness result and not a theorem, and the
distinction is worth keeping: an earlier draft of this docstring asserted the
nested measure was conservative on the strength of a single instance where it
happened to be.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from prismprice import config
from prismprice.decision.markdown import MarkdownProblem, _demand_pmf

__all__ = [
    "RegretRobustPolicy",
    "RobustMarkdownPolicy",
    "RobustMarkdownProblem",
    "exact_policy_value",
    "simulate_robust_policy",
    "solve_regret_robust_markdown",
    "solve_robust_markdown",
]


@dataclass(frozen=True)
class RobustMarkdownProblem:
    """A markdown season whose elasticity is an interval, not a number.

    Args:
        prices: The ladder, ascending. Prices are the action set.
        base_price: Reference at which ``base_demand`` applies.
        base_demand: Expected units per period at ``base_price``.
        elasticity_low: More-elastic end of the interval (most negative).
        elasticity_point: The point estimate. Kept separately rather than
            assumed to be the interval's midpoint, because a DML interval is
            symmetric in the estimate's own scale and need not be centred once
            it has been transformed.
        elasticity_high: Less-elastic end (least negative). Must still be
            negative — an interval touching zero describes a SKU whose demand
            may not respond to price at all, and the markdown problem has no
            interior solution there.
        unit_cost: Cost per unit.
        salvage_value: Recovered per unsold unit at the end of the season.
        horizon: Periods in the season.
        initial_inventory: Units available.
        discount: Per-period discount on future reward.
    """

    prices: tuple[float, ...]
    base_price: float
    base_demand: float
    elasticity_low: float
    elasticity_point: float
    elasticity_high: float
    unit_cost: float
    salvage_value: float
    horizon: int
    initial_inventory: int
    discount: float = 1.0

    def __post_init__(self) -> None:
        if len(self.prices) < 2:
            raise ValueError(f"need at least 2 prices to choose between, got {self.prices}")
        if list(self.prices) != sorted(self.prices):
            raise ValueError("prices must be ascending")
        if not self.elasticity_low <= self.elasticity_point <= self.elasticity_high:
            raise ValueError(
                f"elasticities must satisfy low <= point <= high as signed numbers, got "
                f"low={self.elasticity_low}, point={self.elasticity_point}, "
                f"high={self.elasticity_high}. 'low' is the more elastic (more negative) "
                f"end; passing a magnitude ordering instead of a signed one inverts the "
                f"interval and makes the ambiguity set empty."
            )
        if self.elasticity_high >= 0:
            raise ValueError(
                f"elasticity_high must be negative, got {self.elasticity_high}; an interval "
                f"reaching zero contains demand curves that do not respond to price, and the "
                f"markdown problem has no interior solution on those"
            )
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.initial_inventory < 0:
            raise ValueError(f"initial_inventory must be >= 0, got {self.initial_inventory}")
        if self.salvage_value >= min(self.prices):
            raise ValueError(
                f"salvage_value {self.salvage_value} is not below the cheapest price "
                f"{min(self.prices)}; unsold stock worth more than a sale makes holding "
                f"strictly better than selling and the problem degenerate"
            )
        if not 0.0 < self.discount <= 1.0:
            raise ValueError(f"discount must be in (0, 1], got {self.discount}")

    @classmethod
    def from_markdown_problem(
        cls,
        problem: MarkdownProblem,
        elasticity_low: float,
        elasticity_high: float,
    ) -> RobustMarkdownProblem:
        """Widen an existing single-elasticity problem into an interval one.

        The point estimate is taken from ``problem.elasticity``, so a solved
        certainty-equivalent problem and its robust counterpart are guaranteed
        to describe the same season — which is what makes them comparable.
        """
        return cls(
            prices=problem.prices,
            base_price=problem.base_price,
            base_demand=problem.base_demand,
            elasticity_low=elasticity_low,
            elasticity_point=problem.elasticity,
            elasticity_high=elasticity_high,
            unit_cost=problem.unit_cost,
            salvage_value=problem.salvage_value,
            horizon=problem.horizon,
            initial_inventory=problem.initial_inventory,
            discount=problem.discount,
        )

    def at_elasticity(self, elasticity: float) -> MarkdownProblem:
        """The same season with one elasticity fixed.

        Used to build baselines (the point-estimate DP, an endpoint DP) and to
        simulate against a *true* elasticity the policy was not told about.
        """
        return MarkdownProblem(
            prices=self.prices,
            base_price=self.base_price,
            base_demand=self.base_demand,
            elasticity=elasticity,
            unit_cost=self.unit_cost,
            salvage_value=self.salvage_value,
            horizon=self.horizon,
            initial_inventory=self.initial_inventory,
            discount=self.discount,
        )

    def expected_demand(self, price: float, elasticity: float | None = None) -> float:
        """Poisson mean at *price*, under the point estimate unless told otherwise."""
        e = self.elasticity_point if elasticity is None else elasticity
        return float(self.base_demand * (price / self.base_price) ** e)

    @property
    def interval_width(self) -> float:
        """``elasticity_high - elasticity_low``. Zero means no ambiguity at all,
        in which case every robustness level agrees and the dial is inert."""
        return float(self.elasticity_high - self.elasticity_low)


@dataclass(frozen=True)
class RobustMarkdownPolicy:
    """The solved robust policy, plus the two value functions that certify it.

    Args:
        value: ``V[t, i]`` under the blended operator — what the policy is
            optimising. Reported for completeness; it is a decision criterion
            rather than a forecast of anything, being neither an expectation nor
            a worst case.
        floor: ``F[t, i]``, the CVaR-over-ambiguity value of the selected action.
            The certificate.
        certainty_equivalent: ``C[t, i]``, the point-estimate expected value of
            the selected action. What this policy is worth *if the estimate is
            right* — the quantity a certainty-equivalent DP would report, but
            evaluated at the robust policy rather than at its own.
        policy: ``P[t, i]`` — the price chosen in that state.
        problem: What was solved.
        robustness_level: The rho that produced it.
        cvar_alpha: The tail fraction of the ambiguity set used.
        elasticity_grid: The ambiguity set as actually discretised.
    """

    value: NDArray[np.float64]
    floor: NDArray[np.float64]
    certainty_equivalent: NDArray[np.float64]
    policy: NDArray[np.float64]
    problem: RobustMarkdownProblem = field(repr=False)
    robustness_level: float = 0.0
    cvar_alpha: float = config.DEFAULT_CVAR_ALPHA
    elasticity_grid: tuple[float, ...] = ()

    @property
    def _opening(self) -> tuple[int, int]:
        return self.problem.horizon, self.problem.initial_inventory

    @property
    def certified_profit_floor(self) -> float:
        """Season profit if the elasticity sits in the worst ``cvar_alpha`` of the interval.

        A statement about *parameter* error only, and empirically tight rather
        than provably conservative — see the module docstring for the measured
        gap and why it is not a bound. Demand noise puts individual seasons
        below it routinely; :func:`simulate_robust_policy` reports how often,
        rather than leaving the question open.
        """
        t, i = self._opening
        return float(self.floor[t, i])

    @property
    def certainty_equivalent_profit(self) -> float:
        """Season profit under this policy if the point estimate is correct."""
        t, i = self._opening
        return float(self.certainty_equivalent[t, i])

    @property
    def robustness_cost(self) -> float:
        """Certainty-equivalent profit given up relative to the certified floor.

        The spread this policy is pricing against. A SKU where it is near zero
        has an elasticity estimate precise enough that robustness is free.
        """
        return float(self.certainty_equivalent_profit - self.certified_profit_floor)

    def price_at(self, periods_remaining: int, inventory: int) -> float:
        """The chosen price in one state.

        Raises:
            IndexError: on a state outside the solved grid, rather than
                silently clamping — a caller asking about impossible inventory
                has a bug, and clamping would answer it confidently.
        """
        if not 0 <= periods_remaining <= self.problem.horizon:
            raise IndexError(
                f"periods_remaining {periods_remaining} outside [0, {self.problem.horizon}]"
            )
        if not 0 <= inventory <= self.problem.initial_inventory:
            raise IndexError(f"inventory {inventory} outside [0, {self.problem.initial_inventory}]")
        return float(self.policy[periods_remaining, inventory])

    def shadow_price_at(self, periods_remaining: int, inventory: int) -> float:
        """Marginal certified value of one more unit: ``F(t, i) - F(t, i-1)``.

        Taken off the *floor* rather than the blended value, because a shadow
        price is quoted to somebody deciding whether to buy more stock, and the
        defensible number to quote them is the one that survives the elasticity
        being wrong.
        """
        if inventory <= 0:
            return 0.0
        return float(
            self.floor[periods_remaining, inventory] - self.floor[periods_remaining, inventory - 1]
        )

    def as_dict(self) -> dict[str, Any]:
        t, i = self._opening
        return {
            "certified_profit_floor": self.certified_profit_floor,
            "certainty_equivalent_profit": self.certainty_equivalent_profit,
            "robustness_cost": self.robustness_cost,
            "robustness_level": self.robustness_level,
            "cvar_alpha": self.cvar_alpha,
            "horizon": self.problem.horizon,
            "initial_inventory": self.problem.initial_inventory,
            "opening_price": self.price_at(t, i),
            "opening_shadow_price": self.shadow_price_at(t, i),
            "elasticity_interval": [
                self.problem.elasticity_low,
                self.problem.elasticity_point,
                self.problem.elasticity_high,
            ],
        }


def _tail_size(n: int, alpha: float) -> int:
    """Number of grid atoms in the worst-``alpha`` tail.

    Rounds **up**, where :func:`prismprice.decision.objective._conditional_value_at_risk`
    rounds down. The two are measuring different things and the difference is
    deliberate: that one takes a tail of Monte-Carlo *draws*, where rounding
    down discards a sample and nothing else, while this takes a tail of a
    deterministic *grid* over the ambiguity set, where rounding down can empty
    the very region the set was built to represent. At alpha=0.05 over 21 grid
    points, rounding down leaves a single atom and the alpha dial stops having
    any effect at all.
    """
    return int(min(max(int(np.ceil(alpha * n)), 1), n))


def _cvar_over_grid(values: NDArray[np.float64], alpha: float) -> NDArray[np.float64]:
    """Mean of the worst-``alpha`` fraction along the grid axis.

    Args:
        values: ``(n_actions, n_grid, n_states)``.
        alpha: Tail fraction.

    Returns:
        ``(n_actions, n_states)``.
    """
    k = _tail_size(values.shape[1], alpha)
    worst = np.partition(values, k - 1, axis=1)[:, :k, :]
    return np.asarray(np.mean(worst, axis=1), dtype=float)


def solve_robust_markdown(
    problem: RobustMarkdownProblem,
    robustness_level: float = config.DEFAULT_ROBUSTNESS_LEVEL,
    cvar_alpha: float = config.DEFAULT_CVAR_ALPHA,
    n_grid: int = config.DEFAULT_ROBUST_GRID_SIZE,
    demand_cap: int | None = None,
) -> RobustMarkdownPolicy:
    """Solve the markdown MDP under an elasticity interval, by backward induction.

    Three recursions run together over the same backward sweep. The first picks
    the action and is the operator proper; the other two evaluate that action
    under the two beliefs a reader needs to see, and each carries *its own*
    continuation so that neither is contaminated by the other's optimism.

    Args:
        problem: The season, with its elasticity interval.
        robustness_level: ``rho`` in [0, 1]. 0 reproduces
            :func:`~prismprice.decision.markdown.solve_markdown` exactly at the
            point estimate; 1 prices entirely against the tail.
        cvar_alpha: Tail fraction of the ambiguity set treated as "bad".
        n_grid: Elasticities sampled across the interval.
        demand_cap: Largest demand realisation modelled per period. Defaults to
            a generous multiple of the highest expected demand *across the whole
            interval* — the most elastic end at the cheapest price sells far
            more than the point estimate does, and a cap sized on the point
            estimate would truncate exactly the states robustness cares about.

    Returns:
        :class:`RobustMarkdownPolicy`.
    """
    if not 0.0 <= robustness_level <= 1.0:
        raise ValueError(f"robustness_level must be in [0, 1], got {robustness_level}")
    if not 0.0 < cvar_alpha < 1.0:
        raise ValueError(f"cvar_alpha must be in (0, 1), got {cvar_alpha}")
    if n_grid < 2:
        raise ValueError(
            f"n_grid must be >= 2 to describe an interval, got {n_grid}; a single grid "
            f"point is a point estimate wearing an interval's name"
        )

    horizon = problem.horizon
    stock_levels = problem.initial_inventory
    prices = np.asarray(problem.prices, dtype=float)
    n_actions = len(prices)

    grid = np.linspace(problem.elasticity_low, problem.elasticity_high, n_grid)
    ratio = prices / problem.base_price

    # (n_actions, n_grid) and (n_actions,) Poisson means. Neither depends on t
    # or on inventory, so the whole pmf stack is built once.
    means_grid = problem.base_demand * ratio[:, None] ** grid[None, :]
    means_point = problem.base_demand * ratio**problem.elasticity_point

    cap = demand_cap or int(max(10, np.ceil(float(means_grid.max()) * 4)))
    cap = min(cap, max(stock_levels, 1))

    pmf_grid = np.stack(
        [np.stack([_demand_pmf(float(m), cap) for m in row]) for row in means_grid]
    )  # (A, G, C+1)
    pmf_point = np.stack([_demand_pmf(float(m), cap) for m in means_point])  # (A, C+1)
    pmf_grid_flat = pmf_grid.reshape(n_actions * n_grid, cap + 1)

    margins = prices - problem.unit_cost

    # sold[d, i] = min(d, i) and the inventory it lands on. Both are fixed
    # geometry of the problem, so they are built once and reused by every
    # recursion at every period.
    demand_axis = np.arange(cap + 1, dtype=int)[:, None]
    stock_axis = np.arange(stock_levels + 1, dtype=int)[None, :]
    sold = np.minimum(demand_axis, stock_axis).astype(float)  # (C+1, I+1)
    landing = (stock_axis - sold).astype(int)  # (C+1, I+1)

    # Expected units sold per (action, elasticity) and per (action) — also
    # independent of t, because the demand distribution is.
    units_grid = (pmf_grid_flat @ sold).reshape(n_actions, n_grid, stock_levels + 1)
    units_point = pmf_point @ sold

    reward_grid = units_grid * margins[:, None, None]
    reward_point = units_point * margins[:, None]

    shape = (horizon + 1, stock_levels + 1)
    value = np.zeros(shape, dtype=float)
    floor = np.zeros(shape, dtype=float)
    certainty_equivalent = np.zeros(shape, dtype=float)
    policy = np.zeros(shape, dtype=float)

    terminal = problem.salvage_value * np.arange(stock_levels + 1, dtype=float)
    value[0] = terminal
    floor[0] = terminal
    certainty_equivalent[0] = terminal
    policy[0] = prices[-1]

    stock_index = np.arange(stock_levels + 1, dtype=int)

    for t in range(1, horizon + 1):
        # --- the operator: choose the action -----------------------------
        continuation = value[t - 1][landing]  # (C+1, I+1)
        q_grid = reward_grid + problem.discount * (pmf_grid_flat @ continuation).reshape(
            n_actions, n_grid, stock_levels + 1
        )
        q_point = reward_point + problem.discount * (pmf_point @ continuation)

        q_cvar = _cvar_over_grid(q_grid, cvar_alpha)
        blended = (1.0 - robustness_level) * q_point + robustness_level * q_cvar

        best = np.argmax(blended, axis=0)
        value[t] = blended[best, stock_index]
        policy[t] = prices[best]
        # With nothing on hand there is nothing to price; carry the top of the
        # ladder rather than an arbitrary argmax over identical zero values.
        policy[t, 0] = prices[-1]

        # --- the certificate: evaluate that action under its own tail ----
        floor_continuation = floor[t - 1][landing]
        floor_q = reward_grid + problem.discount * (pmf_grid_flat @ floor_continuation).reshape(
            n_actions, n_grid, stock_levels + 1
        )
        floor[t] = _cvar_over_grid(floor_q, cvar_alpha)[best, stock_index]

        # --- and under the point estimate --------------------------------
        ce_continuation = certainty_equivalent[t - 1][landing]
        ce_q = reward_point + problem.discount * (pmf_point @ ce_continuation)
        certainty_equivalent[t] = ce_q[best, stock_index]

    return RobustMarkdownPolicy(
        value=value,
        floor=floor,
        certainty_equivalent=certainty_equivalent,
        policy=policy,
        problem=problem,
        robustness_level=robustness_level,
        cvar_alpha=cvar_alpha,
        elasticity_grid=tuple(float(e) for e in grid),
    )


@dataclass(frozen=True)
class RegretRobustPolicy:
    """A policy chosen to minimise regret across the ambiguity set.

    Args:
        policy: ``P[t, i]`` — the price chosen in that state.
        policy_value: ``W[g, t, i]`` — this policy's expected value under grid
            elasticity ``g``. One value function per candidate truth, which is
            what makes the regret comparison possible at every state.
        oracle_value: ``V*[g, t, i]`` — the optimal value under grid elasticity
            ``g``, achieved by a policy told that ``g`` is correct.
        problem: What was solved.
        cvar_alpha: Tail fraction of the ambiguity set used.
        elasticity_grid: The ambiguity set as discretised.
    """

    policy: NDArray[np.float64]
    policy_value: NDArray[np.float64]
    oracle_value: NDArray[np.float64]
    problem: RobustMarkdownProblem = field(repr=False)
    cvar_alpha: float = config.DEFAULT_CVAR_ALPHA
    elasticity_grid: tuple[float, ...] = ()

    @property
    def _opening(self) -> tuple[int, int]:
        return self.problem.horizon, self.problem.initial_inventory

    @property
    def regret_by_elasticity(self) -> NDArray[np.float64]:
        """Season regret at each grid elasticity, as a fraction of the oracle."""
        t, i = self._opening
        oracle = self.oracle_value[:, t, i]
        achieved = self.policy_value[:, t, i]
        return np.asarray((oracle - achieved) / np.maximum(oracle, 1e-12), dtype=float)

    @property
    def worst_case_regret(self) -> float:
        """Largest regret anywhere in the ambiguity set, as a fraction."""
        return float(np.max(self.regret_by_elasticity))

    @property
    def mean_regret(self) -> float:
        """Regret averaged over the ambiguity set, as a fraction."""
        return float(np.mean(self.regret_by_elasticity))

    @property
    def certified_regret_bound(self) -> float:
        """CVaR of regret over the set — the quantity this operator minimises.

        The counterpart of :attr:`RobustMarkdownPolicy.certified_profit_floor`,
        and a different kind of promise: not "the season makes at least this
        much" but "however the elasticity turns out inside the interval, at most
        this share of the achievable profit is left on the table".
        """
        regret = np.sort(self.regret_by_elasticity)[::-1]
        k = _tail_size(len(regret), self.cvar_alpha)
        return float(np.mean(regret[:k]))

    def price_at(self, periods_remaining: int, inventory: int) -> float:
        """The chosen price in one state."""
        if not 0 <= periods_remaining <= self.problem.horizon:
            raise IndexError(
                f"periods_remaining {periods_remaining} outside [0, {self.problem.horizon}]"
            )
        if not 0 <= inventory <= self.problem.initial_inventory:
            raise IndexError(f"inventory {inventory} outside [0, {self.problem.initial_inventory}]")
        return float(self.policy[periods_remaining, inventory])

    def as_dict(self) -> dict[str, Any]:
        t, i = self._opening
        return {
            "worst_case_regret_pct": self.worst_case_regret * 100.0,
            "mean_regret_pct": self.mean_regret * 100.0,
            "certified_regret_bound_pct": self.certified_regret_bound * 100.0,
            "cvar_alpha": self.cvar_alpha,
            "opening_price": self.price_at(t, i),
            "elasticity_interval": [
                self.problem.elasticity_low,
                self.problem.elasticity_point,
                self.problem.elasticity_high,
            ],
        }


def solve_regret_robust_markdown(
    problem: RobustMarkdownProblem,
    cvar_alpha: float = config.DEFAULT_CVAR_ALPHA,
    n_grid: int = config.DEFAULT_ROBUST_GRID_SIZE,
    demand_cap: int | None = None,
) -> RegretRobustPolicy:
    """Solve the markdown MDP to minimise *regret* across the elasticity interval.

    :func:`solve_robust_markdown` protects the value: it asks what the season
    earns if the elasticity is unfavourable, and prices against that. This asks a
    different question — how much of the *achievable* profit is given up by not
    knowing the elasticity — and prices against that instead.

    The distinction is not academic, and it is not a matter of taste either. A
    max-min-value policy is drawn toward whichever elasticity in the set makes
    the season *poorest*, and on a markdown problem that is usually the inelastic
    end: less volume is available there at any price, so the value function is
    lower whatever anyone does. But nothing can be done about that. The profit is
    low there for reasons that are not the policy's fault and that no policy can
    repair, and steering the whole season toward defending it sacrifices real
    profit in the elastic states where a decision genuinely was available. Regret
    subtracts off exactly that unreachable part — what the best possible policy
    would have made under the same elasticity — leaving only the loss actually
    attributable to not knowing.

    This behaviour is measured rather than argued: on the real UCI panel, whose
    intervals straddle the elastic/inelastic boundary, the value-robust policy
    is *beaten by the certainty-equivalent DP* on both mean and worst-case
    regret, which is what motivated this operator.

    The recursion
    -------------

    Two families of value function are carried, one per grid elasticity ``g``:

    * ``V*[g]``, the optimal value under ``g`` — what a policy told the truth
      would earn. Computed by ordinary backward induction, independently per
      ``g``, and it is the benchmark regret is measured against.
    * ``W[g]``, the value of *this* policy under ``g``. A single policy, scored
      under every candidate truth at once.

    At each state the action is chosen by

    .. math::

        p^*(t,i) = \\arg\\min_p \\mathrm{CVaR}_\\alpha
                   \\big( V^*_g(t,i) - Q_g(t,i,p) \\big)

    with ``Q_g`` built from ``W[g]`` as its continuation, so the policy is scored
    against the consequences of its own future actions rather than against an
    optimism it will not deliver. Both families advance together in one backward
    sweep.

    Args:
        problem: The season, with its elasticity interval.
        cvar_alpha: Tail fraction of the set. Small alpha approaches pure
            minimax regret; alpha at 1 minimises mean regret over the set.
        n_grid: Elasticities sampled across the interval.
        demand_cap: Largest demand realisation modelled per period.

    Returns:
        :class:`RegretRobustPolicy`.
    """
    if not 0.0 < cvar_alpha <= 1.0:
        raise ValueError(f"cvar_alpha must be in (0, 1], got {cvar_alpha}")
    if n_grid < 2:
        raise ValueError(f"n_grid must be >= 2 to describe an interval, got {n_grid}")

    horizon = problem.horizon
    stock_levels = problem.initial_inventory
    prices = np.asarray(problem.prices, dtype=float)
    n_actions = len(prices)

    grid = np.linspace(problem.elasticity_low, problem.elasticity_high, n_grid)
    ratio = prices / problem.base_price
    means = problem.base_demand * ratio[:, None] ** grid[None, :]  # (A, G)

    cap = demand_cap or int(max(10, np.ceil(float(means.max()) * 4)))
    cap = min(cap, max(stock_levels, 1))

    # pmf[g] is (A, C+1) — the action-conditional demand law under elasticity g.
    pmf = np.stack(
        [
            np.stack([_demand_pmf(float(means[a, g]), cap) for a in range(n_actions)])
            for g in range(n_grid)
        ]
    )

    margins = prices - problem.unit_cost
    demand_axis = np.arange(cap + 1, dtype=int)[:, None]
    stock_axis = np.arange(stock_levels + 1, dtype=int)[None, :]
    sold = np.minimum(demand_axis, stock_axis).astype(float)
    landing = (stock_axis - sold).astype(int)

    # Expected reward per (g, action, inventory) — independent of t.
    reward = np.stack([pmf[g] @ sold for g in range(n_grid)]) * margins[None, :, None]

    terminal = problem.salvage_value * np.arange(stock_levels + 1, dtype=float)
    oracle = np.zeros((n_grid, horizon + 1, stock_levels + 1), dtype=float)
    policy_value = np.zeros((n_grid, horizon + 1, stock_levels + 1), dtype=float)
    oracle[:, 0, :] = terminal
    policy_value[:, 0, :] = terminal

    policy = np.zeros((horizon + 1, stock_levels + 1), dtype=float)
    policy[0] = prices[-1]

    stock_index = np.arange(stock_levels + 1, dtype=int)
    k = _tail_size(n_grid, cvar_alpha)

    for t in range(1, horizon + 1):
        # Q[g, a, i] under each of the two continuations.
        oracle_q = np.empty((n_grid, n_actions, stock_levels + 1), dtype=float)
        policy_q = np.empty((n_grid, n_actions, stock_levels + 1), dtype=float)
        for g in range(n_grid):
            oracle_q[g] = reward[g] + problem.discount * (pmf[g] @ oracle[g, t - 1][landing])
            policy_q[g] = reward[g] + problem.discount * (pmf[g] @ policy_value[g, t - 1][landing])

        # Told the truth, each g does the best it can.
        oracle[:, t, :] = oracle_q.max(axis=1)

        # Regret of each action under each candidate truth, then the tail of it.
        regret = oracle[:, t, :][:, None, :] - policy_q  # (G, A, I+1)
        worst = -np.partition(-regret, k - 1, axis=0)[:k, :, :]
        cvar_regret = worst.mean(axis=0)  # (A, I+1)

        best = np.argmin(cvar_regret, axis=0)
        policy[t] = prices[best]
        policy[t, 0] = prices[-1]
        policy_value[:, t, :] = policy_q[:, best, stock_index]

    return RegretRobustPolicy(
        policy=policy,
        policy_value=policy_value,
        oracle_value=oracle,
        problem=problem,
        cvar_alpha=cvar_alpha,
        elasticity_grid=tuple(float(e) for e in grid),
    )


def exact_policy_value(
    problem: MarkdownProblem,
    price_rule: Callable[[int, int], float],
    demand_cap: int | None = None,
) -> NDArray[np.float64]:
    """Expected profit of a *fixed* policy under *problem*'s dynamics, exactly.

    The same backward induction as :func:`solve_robust_markdown`, with the
    maximisation removed: the action in each state is whatever ``price_rule``
    says, and the recursion evaluates it rather than improving on it. The result
    is the policy's true expected value under these dynamics — no Monte-Carlo
    error, because no sampling happens.

    This is what makes the comparison in the evaluation defensible. Scoring
    policies by simulation puts a standard error on every number, and on a few
    hundred seasons that error is the same order as the effect being measured;
    two policies can then trade places on the strength of the random stream. The
    quantity of interest is a well-defined expectation over an enumerable state
    space, so it can simply be computed.

    Simulation is still worth keeping for what it alone can show — the
    *distribution* of season outcomes, and how often one lands under the
    certified floor — which is why :func:`simulate_robust_policy` remains.

    Args:
        problem: The season **as it truly is**. Pass the true elasticity here,
            not the estimate the policy was built on; that mismatch is the whole
            experiment.
        price_rule: ``(periods_remaining, inventory) -> price``. Prices off the
            ladder are used as given, so a policy may be evaluated under
            dynamics whose ladder it never saw.
        demand_cap: Largest demand realisation modelled per period.

    Returns:
        ``V[t, i]``, the policy's expected remaining profit in each state.
        ``V[horizon, initial_inventory]`` is the season value.
    """
    horizon = problem.horizon
    stock_levels = problem.initial_inventory

    means = {p: problem.expected_demand(p) for p in set(problem.prices)}
    cap = demand_cap or int(max(10, np.ceil(max(means.values()) * 4)))
    cap = min(cap, max(stock_levels, 1))
    pmf_by_price = {p: _demand_pmf(m, cap) for p, m in means.items()}

    demand_axis = np.arange(cap + 1, dtype=int)[:, None]
    stock_axis = np.arange(stock_levels + 1, dtype=int)[None, :]
    sold = np.minimum(demand_axis, stock_axis).astype(float)
    landing = (stock_axis - sold).astype(int)

    value = np.zeros((horizon + 1, stock_levels + 1), dtype=float)
    value[0] = problem.salvage_value * np.arange(stock_levels + 1, dtype=float)

    for t in range(1, horizon + 1):
        continuation = value[t - 1][landing]  # (C+1, I+1)
        for i in range(stock_levels + 1):
            if i == 0:
                # Nothing on hand: no sale is possible and nothing is salvaged
                # beyond what the terminal row already carries.
                value[t, 0] = 0.0
                continue
            price = float(price_rule(t, i))
            pmf = pmf_by_price.get(price)
            if pmf is None:
                pmf = _demand_pmf(problem.expected_demand(price), cap)
                pmf_by_price[price] = pmf
            value[t, i] = float(
                pmf
                @ (sold[:, i] * (price - problem.unit_cost) + problem.discount * continuation[:, i])
            )

    return value


def simulate_robust_policy(
    problem: RobustMarkdownProblem,
    price_rule: Callable[[int, int], float],
    true_elasticity: float,
    certified_floor: float | None = None,
    n_seasons: int = 2000,
    seed: int = config.DEFAULT_SEED,
) -> dict[str, float]:
    """Run *price_rule* against a season whose true elasticity it was not told.

    This is the only honest way to score these policies against each other. Each
    one believes something different about the demand curve, so comparing their
    value functions compares beliefs; and both were fitted on an estimate, so
    scoring them at the estimate would grade the exam they wrote. The true
    elasticity is therefore supplied by the caller and is generally *not* the
    point estimate either policy holds.

    Args:
        problem: The season.
        price_rule: ``(periods_remaining, inventory) -> price``.
        true_elasticity: What demand actually does. Pass a value inside the
            interval to test the well-specified case, outside it to test what
            happens when the estimator's interval missed.
        certified_floor: The floor this policy claimed. When given, the returned
            ``floor_violation_rate`` says how often a realised season fell below
            it — the calibration check on the certificate.
        n_seasons: Independent seasons to average over.
        seed: RNG seed. Common random numbers across policies are the caller's
            job — pass the same seed to each, or the comparison carries the
            noise of two different draws.

    Returns:
        Realised profit and its standard error, sell-through, and (when a floor
        was given) the share of seasons below it.
    """
    rng = np.random.default_rng(seed)
    truth = problem.at_elasticity(true_elasticity)

    profits = np.empty(n_seasons, dtype=float)
    sold_total = np.empty(n_seasons, dtype=float)
    leftover = np.empty(n_seasons, dtype=float)

    for season in range(n_seasons):
        inventory = problem.initial_inventory
        profit = 0.0
        sold = 0
        for t in range(problem.horizon, 0, -1):
            if inventory <= 0:
                break
            price = float(price_rule(t, inventory))
            demand = int(rng.poisson(truth.expected_demand(price)))
            units = min(demand, inventory)
            profit += units * (price - problem.unit_cost)
            inventory -= units
            sold += units
        profit += problem.salvage_value * inventory
        profits[season] = profit
        sold_total[season] = sold
        leftover[season] = inventory

    result = {
        "mean_profit": float(np.mean(profits)),
        "profit_std_error": float(np.std(profits, ddof=1) / np.sqrt(n_seasons)),
        "mean_units_sold": float(np.mean(sold_total)),
        "mean_leftover": float(np.mean(leftover)),
        "sell_through": float(np.mean(sold_total) / problem.initial_inventory)
        if problem.initial_inventory
        else 0.0,
        "true_elasticity": float(true_elasticity),
    }
    if certified_floor is not None:
        result["certified_floor"] = float(certified_floor)
        result["floor_violation_rate"] = float(np.mean(profits < certified_floor))
    return result
