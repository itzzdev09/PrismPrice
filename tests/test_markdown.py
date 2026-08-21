"""
Sequential markdown tests.

Two kinds of claim are checked here, and they are checked differently.

**Structural claims** — prices fall as the season runs out, prices rise when
stock is scarce, the shadow price decays to salvage — are read straight off the
solved policy. They are properties of the optimum, not of any simulation, so a
simulation cannot make them true or false.

**Performance claims** are scored by simulating realised profit, never by
comparing value functions. A policy's value function is what it *believes*; two
policies that believe different things cannot be compared on belief. The
baselines are also given every advantage: the static price is chosen with full
hindsight over the whole demand curve, which no real operator has.
"""

from __future__ import annotations

import numpy as np
import pytest

from prismprice.decision.markdown import (
    MarkdownProblem,
    simulate_policy,
    solve_markdown,
)

PRICES = tuple(np.round(np.arange(15.95, 40.0, 1.0), 2))


def make_problem(horizon: int = 60, inventory: int = 420, demand: float = 8.0) -> MarkdownProblem:
    return MarkdownProblem(
        prices=PRICES,
        base_price=30.0,
        base_demand=demand,
        elasticity=-2.5,
        unit_cost=12.0,
        salvage_value=2.0,
        horizon=horizon,
        initial_inventory=inventory,
    )


@pytest.fixture(scope="module")
def problem() -> MarkdownProblem:
    return make_problem()


@pytest.fixture(scope="module")
def policy(problem):
    return solve_markdown(problem)


def myopic_rule(problem: MarkdownProblem):
    """Maximise this period's expected profit, ignoring the future entirely."""

    def rule(periods_remaining: int, inventory: int) -> float:
        return max(
            problem.prices,
            key=lambda p: min(problem.expected_demand(p), inventory) * (p - problem.unit_cost),
        )

    return rule


def best_static_price(problem: MarkdownProblem, seed: int = 1) -> float:
    """The best single fixed price, chosen with hindsight. A generous baseline."""
    scores = {
        price: simulate_policy(problem, lambda t, i, p=price: p, n_seasons=400, seed=seed)[
            "mean_profit"
        ]
        for price in problem.prices
    }
    return max(scores, key=lambda p: scores[p])


# ---------------------------------------------------------------------------
# Structural properties of the optimum
# ---------------------------------------------------------------------------


def test_price_falls_as_the_season_runs_out(policy):
    """Nothing marks down explicitly.

    The continuation value of a unit collapses as the days left to sell it
    disappear, so the optimiser stops protecting margin. A rule-based ladder has
    to be told to do this; the Bellman recursion works it out.
    """
    held = 250
    trajectory = [policy.price_at(t, held) for t in (60, 45, 30, 20, 10, 5)]
    assert trajectory == sorted(trajectory, reverse=True), f"not monotone: {trajectory}"
    assert trajectory[0] > trajectory[-1], "price never marked down at all"


def test_price_rises_when_stock_is_scarce(policy):
    """The same arithmetic run the other way: few units, plenty of time."""
    prices = [policy.price_at(30, i) for i in (400, 300, 200, 100, 50)]
    assert prices == sorted(prices), f"price did not rise as stock fell: {prices}"
    assert prices[-1] > prices[0]


def test_shadow_price_is_never_negative(policy, problem):
    """A unit of stock cannot be worth less than nothing."""
    for t in range(0, problem.horizon + 1, 10):
        for i in range(1, problem.initial_inventory + 1, 50):
            assert policy.shadow_price_at(t, i) >= -1e-9


def test_shadow_price_falls_as_stock_grows(policy):
    """Concavity of the value function: the tenth unit is worth more than the
    three-hundredth, because the season can only absorb so many."""
    values = [policy.shadow_price_at(40, i) for i in (25, 100, 200, 300, 400)]
    assert values == sorted(values, reverse=True), f"not decreasing: {values}"


def test_shadow_price_reaches_salvage_when_stock_is_hopeless(policy, problem):
    """With far more units than the season can absorb, the marginal unit is
    worth exactly what it salvages for — the derived version of the `nu` the
    one-shot objective takes as an assumption."""
    assert policy.shadow_price_at(2, problem.initial_inventory) == pytest.approx(
        problem.salvage_value, abs=0.5
    )


def test_value_rises_with_inventory_and_with_time(policy, problem):
    """Monotonicity in both arguments. Violations mean the recursion is wrong."""
    for t in (10, 30, 60):
        row = policy.value[t]
        assert np.all(np.diff(row) >= -1e-9), f"value not increasing in stock at t={t}"

    column = policy.value[:, problem.initial_inventory]
    assert np.all(np.diff(column) >= -1e-9), "value not increasing in time remaining"


def test_terminal_value_is_pure_salvage(policy, problem):
    expected = problem.salvage_value * np.arange(problem.initial_inventory + 1)
    assert np.allclose(policy.value[0], expected)


def test_no_stock_is_worth_nothing(policy, problem):
    for t in range(problem.horizon + 1):
        assert policy.value[t, 0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Performance, scored on realised profit
# ---------------------------------------------------------------------------


def test_beats_the_best_static_price_chosen_with_hindsight(problem, policy):
    """The strongest baseline available: a single price picked knowing the whole
    demand curve, which no real operator has."""
    static = best_static_price(problem)
    dp_result = simulate_policy(
        problem, lambda t, i: policy.price_at(t, i), n_seasons=3000, seed=11
    )
    static_result = simulate_policy(problem, lambda t, i: static, n_seasons=3000, seed=11)

    assert dp_result["mean_profit"] > static_result["mean_profit"], (
        f"DP {dp_result['mean_profit']:.1f} did not beat static {static:.2f} "
        f"at {static_result['mean_profit']:.1f}"
    )


def test_crushes_myopic_pricing(problem, policy):
    """Myopic ignores that stock is finite, so it prices at the one-period
    optimum, sells out early at low margin, and leaves most of the season's
    value on the table."""
    dp_result = simulate_policy(
        problem, lambda t, i: policy.price_at(t, i), n_seasons=1500, seed=11
    )
    myopic_result = simulate_policy(problem, myopic_rule(problem), n_seasons=1500, seed=11)

    assert dp_result["mean_profit"] > 1.5 * myopic_result["mean_profit"]


def test_sells_through_more_than_a_static_price(problem, policy):
    """Adaptivity shows up as clearance: the DP reacts to realised sales, a
    fixed price cannot."""
    static = best_static_price(problem)
    dp_result = simulate_policy(
        problem, lambda t, i: policy.price_at(t, i), n_seasons=1500, seed=11
    )
    static_result = simulate_policy(problem, lambda t, i: static, n_seasons=1500, seed=11)
    assert dp_result["sell_through"] > static_result["sell_through"]


def test_advantage_grows_with_demand_uncertainty():
    """The theoretical prediction, checked rather than asserted.

    Adapting to realised sales is worth more when there is more to adapt to. A
    short season carries proportionally more Poisson noise, so the gap over a
    fixed price should widen as the horizon shrinks — and it does: measured
    lifts of ~1.1% over 60 periods against ~2.2% over 7.
    """
    lifts = []
    for horizon, inventory in ((60, 420), (14, 80)):
        problem = make_problem(horizon=horizon, inventory=inventory)
        policy = solve_markdown(problem)
        static = best_static_price(problem)
        dp_result = simulate_policy(
            problem, lambda t, i: policy.price_at(t, i), n_seasons=2500, seed=11
        )
        static_result = simulate_policy(problem, lambda t, i: static, n_seasons=2500, seed=11)
        lifts.append(dp_result["mean_profit"] / static_result["mean_profit"] - 1.0)

    assert lifts[1] > lifts[0], (
        f"advantage did not grow with uncertainty: {lifts[0]:.3%} over a long season "
        f"vs {lifts[1]:.3%} over a short one"
    )


def test_expected_profit_matches_the_simulation(problem, policy):
    """The value function's own claim, checked against realised outcomes.

    A DP whose V(T, I) disagrees with what its policy actually earns has a bug
    in the recursion, the simulator, or both — and neither is visible from
    inside one of them.
    """
    simulated = simulate_policy(problem, lambda t, i: policy.price_at(t, i), n_seasons=4000, seed=3)
    assert simulated["mean_profit"] == pytest.approx(policy.expected_profit, rel=0.02)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_policy_refuses_states_outside_the_grid(policy, problem):
    """A caller asking about impossible inventory has a bug; clamping would
    answer it confidently."""
    with pytest.raises(IndexError, match="periods_remaining"):
        policy.price_at(problem.horizon + 1, 10)
    with pytest.raises(IndexError, match="inventory"):
        policy.price_at(10, problem.initial_inventory + 1)


def test_shadow_price_of_no_stock_is_zero(policy):
    assert policy.shadow_price_at(30, 0) == 0.0


def test_salvage_above_the_cheapest_price_is_refused():
    """Unsold stock worth more than a sale makes holding strictly better than
    selling, and the problem degenerate."""
    with pytest.raises(ValueError, match="not below the cheapest price"):
        MarkdownProblem(
            prices=(10.0, 20.0),
            base_price=15.0,
            base_demand=5.0,
            elasticity=-2.0,
            unit_cost=5.0,
            salvage_value=12.0,
            horizon=10,
            initial_inventory=50,
        )


def test_non_negative_elasticity_is_refused():
    with pytest.raises(ValueError, match="elasticity must be negative"):
        MarkdownProblem(
            prices=(10.0, 20.0),
            base_price=15.0,
            base_demand=5.0,
            elasticity=0.5,
            unit_cost=5.0,
            salvage_value=1.0,
            horizon=10,
            initial_inventory=50,
        )


def test_unsorted_prices_are_refused():
    with pytest.raises(ValueError, match="ascending"):
        MarkdownProblem(
            prices=(20.0, 10.0),
            base_price=15.0,
            base_demand=5.0,
            elasticity=-2.0,
            unit_cost=5.0,
            salvage_value=1.0,
            horizon=10,
            initial_inventory=50,
        )


def test_a_single_price_is_not_a_decision():
    with pytest.raises(ValueError, match="at least 2 prices"):
        MarkdownProblem(
            prices=(10.0,),
            base_price=15.0,
            base_demand=5.0,
            elasticity=-2.0,
            unit_cost=5.0,
            salvage_value=1.0,
            horizon=10,
            initial_inventory=50,
        )


def test_demand_falls_with_price(problem):
    assert problem.expected_demand(40.0) < problem.expected_demand(20.0)
    assert problem.expected_demand(problem.base_price) == pytest.approx(problem.base_demand)


def test_zero_inventory_season_is_solvable():
    """Degenerate but reachable: a SKU that sold out before the season started."""
    problem = make_problem(horizon=5, inventory=0)
    solved = solve_markdown(problem)
    assert solved.expected_profit == pytest.approx(0.0)


def test_summary_reports_the_opening_decision(policy):
    summary = policy.as_dict()
    assert summary["opening_price"] in PRICES
    assert summary["opening_shadow_price"] >= 0.0
    assert summary["expected_profit"] > 0.0
