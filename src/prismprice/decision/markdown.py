"""
Sequential markdown pricing (L3) — finite-horizon dynamic programming.

Everywhere else in this system a price is chosen for *today*. That is the right
frame when stock replenishes: sell a mug too cheaply and another mug arrives.
It is the wrong frame for a finite quantity sold over a finite season — seasonal
stock, a discontinued line, an end-of-life run. There, selling a unit today
consumes the option to sell it tomorrow, and a price is a decision about
inventory as much as about margin.

That makes it a genuine Markov decision process rather than a repeated one-shot
choice::

    state   (t, i)   days remaining, units on hand
    action  p        price from the ladder
    reward  min(d,i) * (p - c)
    next    i - min(d,i)

and it is solved by backward induction on the Bellman equation::

    V(0, i) = salvage * i
    V(t, i) = max_p  E_d[ min(d,i)(p - c) + V(t-1, i - min(d,i)) ]

Why exact DP rather than a learned policy
------------------------------------------

This is the module where reinforcement learning would be the obvious tool, and
it deliberately does not use one. The state space here is *enumerable* — a few
hundred inventory levels by a few hundred days — so the Bellman equation can be
solved exactly, and an exact optimum dominates any approximation of it. Fitting
a policy network would add sampling error, training variance and an
uninterpretable decision rule, and buy nothing back.

Function approximation earns its place when the state stops being enumerable:
several thousand SKUs sharing a shelf-space constraint, cross-price effects
inside the category, or a competitor whose price is part of the state. Until
then, the honest engineering answer is that the tabular solution is not a
simplification of RL — it *is* the answer RL would be approximating.

The one thing worth taking from the RL framing is the discipline: the value
function is the object of interest, and the policy is a by-product of it. This
module exposes both, because ``V(t, i)`` is what tells a merchandiser the
opportunity cost of the stock they are holding — which is the same quantity the
one-shot objective calls the inventory shadow price ``nu``, computed properly
instead of assumed.

Two behaviours fall out of the solution rather than being coded
---------------------------------------------------------------

**Prices fall as the season runs out with stock left.** Nothing marks down
explicitly; the continuation value of a unit simply collapses as the days to
sell it disappear, so the optimiser stops protecting margin.

**Prices rise when stock is scarce relative to time.** The same arithmetic, run
the other way. A rule-based markdown ladder has to be told this; the DP works it
out, and both directions are asserted in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from prismprice import config

__all__ = [
    "MarkdownPolicy",
    "MarkdownProblem",
    "simulate_policy",
    "solve_markdown",
]


@dataclass(frozen=True)
class MarkdownProblem:
    """A finite quantity to sell over a finite season.

    Args:
        prices: The ladder, ascending. Prices are the action set.
        base_price: Reference at which ``base_demand`` applies.
        base_demand: Expected units per period at ``base_price``.
        elasticity: Own-price elasticity. Demand at ``p`` is Poisson with mean
            ``base_demand * (p / base_price) ** elasticity``.
        unit_cost: Cost per unit.
        salvage_value: Recovered per unsold unit at the end of the season. This
            is what makes the problem interesting: a salvage value equal to the
            price would make holding costless and the optimal policy trivial.
        horizon: Periods in the season.
        initial_inventory: Units available.
        discount: Per-period discount on future reward.
    """

    prices: tuple[float, ...]
    base_price: float
    base_demand: float
    elasticity: float
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
        if self.elasticity >= 0:
            raise ValueError(
                f"elasticity must be negative, got {self.elasticity}; a non-negative "
                f"elasticity means demand rises with price and the markdown problem "
                f"has no interior solution"
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

    def expected_demand(self, price: float) -> float:
        """Poisson mean at *price*."""
        return float(self.base_demand * (price / self.base_price) ** self.elasticity)


@dataclass(frozen=True)
class MarkdownPolicy:
    """The solved value function and the policy that falls out of it.

    Args:
        value: ``V[t, i]`` — expected remaining profit with ``t`` periods left
            and ``i`` units on hand.
        policy: ``P[t, i]`` — the optimal price in that state.
        problem: What was solved.
    """

    value: NDArray[np.float64]
    policy: NDArray[np.float64]
    problem: MarkdownProblem = field(repr=False)

    @property
    def expected_profit(self) -> float:
        """Expected profit over the whole season from the initial state."""
        return float(self.value[self.problem.horizon, self.problem.initial_inventory])

    def price_at(self, periods_remaining: int, inventory: int) -> float:
        """The optimal price in one state.

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
        """Marginal value of one more unit: ``V(t, i) - V(t, i-1)``.

        This is the inventory shadow price ``nu`` that the one-shot objective
        takes as an input and treats as an assumption. Here it is *derived* —
        the opportunity cost of selling a unit now is exactly what that unit
        would have earned later, and the value function knows that number.

        Returns 0.0 at zero inventory, where there is no unit to be marginal about.
        """
        if inventory <= 0:
            return 0.0
        return float(
            self.value[periods_remaining, inventory] - self.value[periods_remaining, inventory - 1]
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "expected_profit": self.expected_profit,
            "horizon": self.problem.horizon,
            "initial_inventory": self.problem.initial_inventory,
            "opening_price": self.price_at(self.problem.horizon, self.problem.initial_inventory),
            "opening_shadow_price": self.shadow_price_at(
                self.problem.horizon, self.problem.initial_inventory
            ),
        }


def _demand_pmf(mean: float, max_units: int) -> NDArray[np.float64]:
    """Poisson pmf over ``0..max_units``, with the tail folded into the last cell.

    Folding rather than truncating matters: a truncated pmf does not sum to one,
    so every expectation computed from it is quietly scaled down and the whole
    value function shrinks. The tail belongs at ``max_units`` because demand
    above the inventory on hand is indistinguishable from demand exactly equal
    to it — the extra customers leave either way.
    """
    counts = np.arange(max_units + 1, dtype=float)
    log_pmf = counts * np.log(max(mean, 1e-12)) - mean - _log_factorial(counts)
    pmf: NDArray[np.float64] = np.exp(log_pmf)
    pmf[-1] += max(1.0 - float(np.sum(pmf)), 0.0)
    return pmf


def _log_factorial(counts: NDArray[np.float64]) -> NDArray[np.float64]:
    """``log(k!)`` via the gamma function, so large counts do not overflow."""
    from math import lgamma

    return np.array([lgamma(k + 1.0) for k in counts], dtype=float)


def solve_markdown(problem: MarkdownProblem, demand_cap: int | None = None) -> MarkdownPolicy:
    """Solve the markdown MDP exactly by backward induction.

    Args:
        problem: The season to solve.
        demand_cap: Largest demand realisation modelled per period. Defaults to
            a generous multiple of the highest expected demand; the Poisson tail
            beyond it is folded in rather than dropped.

    Returns:
        :class:`MarkdownPolicy` carrying both the value function and the policy.
    """
    inventory_levels = problem.initial_inventory
    horizon = problem.horizon

    means = np.array([problem.expected_demand(p) for p in problem.prices], dtype=float)
    cap = demand_cap or int(max(10, np.ceil(means.max() * 4)))
    cap = min(cap, max(inventory_levels, 1))

    pmfs = np.stack([_demand_pmf(mean, cap) for mean in means])
    margins = np.array(problem.prices, dtype=float) - problem.unit_cost

    value = np.zeros((horizon + 1, inventory_levels + 1), dtype=float)
    policy = np.zeros((horizon + 1, inventory_levels + 1), dtype=float)

    # Terminal condition: whatever is left is salvaged.
    value[0, :] = problem.salvage_value * np.arange(inventory_levels + 1, dtype=float)
    policy[0, :] = problem.prices[-1]

    stock = np.arange(inventory_levels + 1, dtype=int)

    for t in range(1, horizon + 1):
        continuation = value[t - 1]
        # candidate[a, i] = expected value of taking action a in state (t, i)
        candidate = np.empty((len(problem.prices), inventory_levels + 1), dtype=float)

        for action, pmf in enumerate(pmfs):
            expected = np.zeros(inventory_levels + 1, dtype=float)
            for demand in range(cap + 1):
                probability = pmf[demand]
                if probability <= 0.0:
                    continue
                sold = np.minimum(demand, stock)
                remaining = stock - sold
                expected += probability * (
                    sold * margins[action] + problem.discount * continuation[remaining]
                )
            candidate[action] = expected

        best = np.argmax(candidate, axis=0)
        value[t] = candidate[best, stock]
        policy[t] = np.array(problem.prices, dtype=float)[best]
        # With nothing on hand there is nothing to price; carry the top of the
        # ladder rather than an arbitrary argmax over identical zero values.
        policy[t, 0] = problem.prices[-1]

    return MarkdownPolicy(value=value, policy=policy, problem=problem)


def simulate_policy(
    problem: MarkdownProblem,
    price_rule: Any,
    n_seasons: int = 2000,
    seed: int = config.DEFAULT_SEED,
) -> dict[str, float]:
    """Run *price_rule* through simulated seasons and report realised profit.

    Scoring policies against each other in simulation rather than by comparing
    their own value functions: a policy's value function is what it *believes*,
    and two policies that believe different things cannot be compared on belief.

    Args:
        problem: The season.
        price_rule: ``(periods_remaining, inventory) -> price``.
        n_seasons: Independent seasons to average over.
        seed: RNG seed. Common random numbers across policies are the caller's
            job — pass the same seed to each.

    Returns:
        Mean profit, mean units sold, mean leftover stock and sell-through.
    """
    rng = np.random.default_rng(seed)
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
            demand = int(rng.poisson(problem.expected_demand(price)))
            units = min(demand, inventory)
            profit += units * (price - problem.unit_cost)
            inventory -= units
            sold += units
        profit += problem.salvage_value * inventory
        profits[season] = profit
        sold_total[season] = sold
        leftover[season] = inventory

    return {
        "mean_profit": float(np.mean(profits)),
        "profit_std_error": float(np.std(profits, ddof=1) / np.sqrt(n_seasons)),
        "mean_units_sold": float(np.mean(sold_total)),
        "mean_leftover": float(np.mean(leftover)),
        "sell_through": float(np.mean(sold_total) / problem.initial_inventory)
        if problem.initial_inventory
        else 0.0,
    }
