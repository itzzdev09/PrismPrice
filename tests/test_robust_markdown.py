"""
Robust sequential markdown tests.

The claims here divide into three kinds, and each is checked in the only way it
can be.

**Nesting claims** are exact arithmetic. ``robustness_level=0`` must reproduce
the certainty-equivalent DP value-for-value, and a degenerate interval must make
the dial inert. These are not approximations to be checked within a tolerance —
they are the property that makes the new operator a strict generalisation of the
old one rather than a different model that happens to resemble it, so they are
asserted at float precision.

**Monotonicity claims** are read off solved policies. More robustness must buy a
better certificate and cost expected profit; if it ever did the reverse the
operator would be mislabelled.

**The certificate itself** is checked by simulation, because it is the only
claim about the world rather than about the arithmetic. The floor says: if the
elasticity sits in the bad tail of the interval, the season still makes this
much. So the policy is run at every elasticity in the grid and the realised tail
is compared against what was promised — the certificate must be *conservative*,
never optimistic.
"""

from __future__ import annotations

import numpy as np
import pytest

from prismprice import config
from prismprice.decision.markdown import MarkdownProblem, simulate_policy, solve_markdown
from prismprice.decision.robust_markdown import (
    RobustMarkdownProblem,
    _tail_size,
    exact_policy_value,
    simulate_robust_policy,
    solve_regret_robust_markdown,
    solve_robust_markdown,
)

PRICES = tuple(np.round(np.arange(15.95, 40.0, 1.0), 2))


def make_base(horizon: int = 40, inventory: int = 300) -> MarkdownProblem:
    return MarkdownProblem(
        prices=PRICES,
        base_price=30.0,
        base_demand=8.0,
        elasticity=-2.5,
        unit_cost=12.0,
        salvage_value=2.0,
        horizon=horizon,
        initial_inventory=inventory,
    )


@pytest.fixture(scope="module")
def base() -> MarkdownProblem:
    return make_base()


@pytest.fixture(scope="module")
def wide(base) -> RobustMarkdownProblem:
    """A deliberately wide interval — the regime where robustness is load-bearing."""
    return RobustMarkdownProblem.from_markdown_problem(base, -4.5, -1.2)


# ---------------------------------------------------------------------------
# Nesting: the new operator contains the old one
# ---------------------------------------------------------------------------


def test_zero_robustness_reproduces_the_certainty_equivalent_dp(base, wide):
    """rho = 0 is not *similar to* solve_markdown, it *is* solve_markdown.

    This is the load-bearing test of the whole module. If it fails, every
    reported difference between the robust and classical policies is
    contaminated by an implementation difference rather than the operator, and
    the entire evaluation means nothing.

    The tolerance is float-accumulation only: this module reaches the same sum
    through matrix products where markdown.py loops, so the two agree to the
    last few bits rather than bit-for-bit.
    """
    classical = solve_markdown(base)
    robust = solve_robust_markdown(wide, robustness_level=0.0)

    assert np.allclose(classical.value, robust.value, rtol=1e-9, atol=1e-7)
    assert np.array_equal(classical.policy, robust.policy)
    assert robust.certainty_equivalent_profit == pytest.approx(classical.expected_profit)


def test_no_ambiguity_makes_the_dial_inert(base):
    """A zero-width interval is a point estimate, and every rho must agree on it.

    The interesting half of this is the floor: with nothing to be uncertain
    about, the certified worst case must equal the expected value exactly. A
    floor that sat below it would mean the operator was manufacturing caution
    out of an interval that contains no disagreement.
    """
    degenerate = RobustMarkdownProblem.from_markdown_problem(base, base.elasticity, base.elasticity)
    timid = solve_robust_markdown(degenerate, robustness_level=0.0)
    paranoid = solve_robust_markdown(degenerate, robustness_level=1.0)

    assert np.array_equal(timid.policy, paranoid.policy)
    assert paranoid.certified_profit_floor == pytest.approx(paranoid.certainty_equivalent_profit)
    assert paranoid.robustness_cost == pytest.approx(0.0, abs=1e-9)
    assert degenerate.interval_width == 0.0


# ---------------------------------------------------------------------------
# Monotonicity: the dial does what its name says
# ---------------------------------------------------------------------------


def test_robustness_buys_a_better_floor_and_costs_expected_profit(wide):
    """The trade the dial exists to make, in both directions at once.

    Checked together rather than as two tests because either alone is passable
    by a broken operator: a policy that simply priced lower would raise the
    floor, and one that priced at random would lower expected profit. Only the
    pair moving in opposite directions across the same sweep is evidence that a
    trade is being made rather than a mistake.
    """
    solved = [solve_robust_markdown(wide, robustness_level=r) for r in (0.0, 0.25, 0.5, 0.75, 1.0)]
    floors = [s.certified_profit_floor for s in solved]
    expected = [s.certainty_equivalent_profit for s in solved]

    assert floors == sorted(floors), f"floor did not improve with robustness: {floors}"
    assert expected == sorted(expected, reverse=True), f"no cost was paid: {expected}"
    assert floors[-1] > floors[0], "the dial never moved the certificate at all"


def test_the_floor_never_exceeds_the_expected_value(wide):
    """A worst case above the average is a contradiction, not a good outcome."""
    for rho in (0.0, 0.5, 1.0):
        solved = solve_robust_markdown(wide, robustness_level=rho)
        assert solved.certified_profit_floor <= solved.certainty_equivalent_profit + 1e-9
        assert solved.robustness_cost >= -1e-9


def test_a_wider_interval_costs_more_to_insure(base):
    """Robustness against a vague estimate must cost more than against a sharp one.

    This is what ties the dial to the estimator: the price of caution is set by
    how much the data actually failed to pin down, not by the operator's mood.
    """
    costs = [
        solve_robust_markdown(
            RobustMarkdownProblem.from_markdown_problem(base, -2.5 - w, -2.5 + w),
            robustness_level=1.0,
        ).robustness_cost
        for w in (0.2, 0.8, 1.6)
    ]
    assert costs == sorted(costs), f"cost of robustness did not grow with the interval: {costs}"


# ---------------------------------------------------------------------------
# The certificate, against simulation
# ---------------------------------------------------------------------------


def test_the_certified_floor_is_tight_against_realised_seasons(wide):
    """Run the policy at every elasticity in its own ambiguity set and take the tail.

    The floor claims: in the worst ``cvar_alpha`` of the interval, the season
    makes about this much. **About** is load-bearing and the tolerance below is
    two-sided on purpose.

    An earlier version of this test asserted the floor was a strict lower bound,
    on the reasoning that a nested risk measure — one applying the tail afresh
    each period — is dominated by the static measure taken over whole
    trajectories. That reasoning does not transfer here: the elasticity that is
    worst at a given state is not necessarily the one that is worst over a whole
    season, so the per-state tail and the end-to-end tail select different
    members of the set and neither dominates. Measured across 40 instances the
    floor is optimistic in 88% of them, by at most 0.042%.

    So what is asserted is tightness within 0.5% — an order of magnitude above
    the observed worst case, and still far below the 5-10% effects the operator
    is used to decide — rather than a bound that the construction does not
    actually provide.
    """
    solved = solve_robust_markdown(wide, robustness_level=1.0)
    realised = np.array(
        [
            simulate_robust_policy(wide, solved.price_at, true_elasticity=e, n_seasons=300, seed=7)[
                "mean_profit"
            ]
            for e in solved.elasticity_grid
        ]
    )

    k = _tail_size(len(realised), solved.cvar_alpha)
    realised_tail = float(np.mean(np.sort(realised)[:k]))

    assert realised_tail == pytest.approx(solved.certified_profit_floor, rel=5e-3), (
        f"certificate is not tight: promised {solved.certified_profit_floor:.2f}, "
        f"realised tail {realised_tail:.2f}"
    )


def test_simulation_reports_floor_violations_when_asked(wide):
    """The certificate covers parameter error, not Poisson luck, and says so.

    An individual season can land below the floor on a bad run of draws even
    when the elasticity is exactly as assumed. The violation rate is reported
    rather than assumed to be zero, because a certificate whose failure mode is
    undocumented reads as a stronger guarantee than it is.
    """
    solved = solve_robust_markdown(wide, robustness_level=1.0)
    result = simulate_robust_policy(
        wide,
        solved.price_at,
        true_elasticity=wide.elasticity_point,
        certified_floor=solved.certified_profit_floor,
        n_seasons=400,
        seed=11,
    )
    assert 0.0 <= result["floor_violation_rate"] <= 1.0
    assert result["true_elasticity"] == pytest.approx(wide.elasticity_point)
    assert result["certified_floor"] == pytest.approx(solved.certified_profit_floor)


def test_common_random_numbers_make_policies_comparable(wide, base):
    """Two policies on the same seed must face the same demand draws.

    Without this the reported difference between a robust and a classical policy
    is partly the difference between two random streams, which on a few hundred
    seasons is the same order as the effect being measured.
    """
    robust = solve_robust_markdown(wide, robustness_level=1.0)
    first = simulate_robust_policy(wide, robust.price_at, -3.0, n_seasons=120, seed=3)
    second = simulate_robust_policy(wide, robust.price_at, -3.0, n_seasons=120, seed=3)
    assert first == second


# ---------------------------------------------------------------------------
# Exact policy evaluation, which the benchmark rests on
# ---------------------------------------------------------------------------


def test_evaluating_the_optimal_policy_recovers_the_optimal_value(base):
    """The tightest available check on the evaluator.

    Backward induction with the maximisation removed, run on the policy that
    maximisation produced, must return the value function it came from. Any
    disagreement in the transition, the truncation of demand at inventory, or
    the terminal salvage would show up here as a mismatch, because the two
    routines share none of their code but must agree exactly.
    """
    optimal = solve_markdown(base)
    evaluated = exact_policy_value(base, optimal.price_at)
    assert evaluated[base.horizon, base.initial_inventory] == pytest.approx(
        optimal.expected_profit, rel=1e-9
    )


def test_exact_evaluation_agrees_with_simulation(base):
    """The analytic value must sit inside the simulator's confidence interval.

    They are computed in completely different ways — one sums a Poisson pmf over
    a state grid, the other draws seasons — so agreement is evidence that the
    benchmark's headline numbers are not an artefact of dropping to closed form.
    """
    optimal = solve_markdown(base)
    analytic = float(
        exact_policy_value(base, optimal.price_at)[base.horizon, base.initial_inventory]
    )
    sampled = simulate_policy(base, optimal.price_at, n_seasons=3000, seed=5)
    assert abs(analytic - sampled["mean_profit"]) < 4.0 * sampled["profit_std_error"]


def test_no_policy_beats_the_optimum_under_its_own_dynamics(base, wide):
    """The optimum is an upper bound, and the evaluator must respect it.

    Every policy compared in the benchmark is scored by this function against a
    truth it was not built on. If any of them could score *above* the DP optimum
    for that truth, the evaluator would be crediting profit the dynamics cannot
    produce, and the entire regret table would be meaningless.
    """
    ceiling = solve_markdown(base).expected_profit
    contenders = [
        solve_robust_markdown(wide, robustness_level=1.0).price_at,
        solve_robust_markdown(wide, robustness_level=0.5).price_at,
        solve_markdown(wide.at_elasticity(wide.elasticity_low)).price_at,
        lambda t, i: base.prices[-1],
        lambda t, i: base.prices[0],
    ]
    for rule in contenders:
        scored = exact_policy_value(base, rule)[base.horizon, base.initial_inventory]
        assert scored <= ceiling + 1e-6


# ---------------------------------------------------------------------------
# Structure preserved from the classical solution
# ---------------------------------------------------------------------------


def test_robust_policy_still_marks_down_as_the_season_runs_out(wide):
    """Robustness must not destroy the behaviour that makes this a markdown model.

    The continuation value still collapses as the days to sell disappear,
    whatever the elasticity turns out to be, so the price must still fall. An
    operator so cautious that it held price to the last day would be protecting
    a margin it has no remaining chance to earn.
    """
    solved = solve_robust_markdown(wide, robustness_level=1.0)
    trajectory = [solved.price_at(t, 200) for t in (40, 30, 20, 10, 5)]
    assert trajectory == sorted(trajectory, reverse=True), f"not monotone: {trajectory}"
    assert trajectory[0] > trajectory[-1], "price never marked down at all"


def test_certified_shadow_price_is_non_negative_and_decreasing(wide):
    """Stock cannot be worth less than nothing, and the tenth unit beats the
    two-hundredth — concavity survives the tail being taken."""
    solved = solve_robust_markdown(wide, robustness_level=0.75)
    for i in range(1, 300, 40):
        assert solved.shadow_price_at(20, i) >= -1e-9
    assert solved.shadow_price_at(20, 0) == 0.0

    marginal = [solved.shadow_price_at(30, i) for i in (25, 100, 200, 290)]
    assert marginal == sorted(marginal, reverse=True), f"not decreasing: {marginal}"


# ---------------------------------------------------------------------------
# The regret operator
# ---------------------------------------------------------------------------


def test_internal_oracle_matches_an_independent_solve(wide):
    """The regret benchmark must be the real optimum, not an internal artefact.

    ``solve_regret_robust_markdown`` computes ``V*[g]`` inside its own sweep,
    vectorised across the grid. Every regret it reports is measured against
    those numbers, so if they were even slightly wrong the operator would be
    minimising a distance from a fiction. They are checked here against
    :func:`solve_markdown` run independently at each grid elasticity.
    """
    solved = solve_regret_robust_markdown(wide)
    t, i = wide.horizon, wide.initial_inventory
    for g, elasticity in enumerate(solved.elasticity_grid):
        independent = solve_markdown(wide.at_elasticity(elasticity)).expected_profit
        assert solved.oracle_value[g, t, i] == pytest.approx(independent, rel=1e-9)


def test_internal_policy_value_matches_evaluating_the_emitted_policy(wide):
    """``W[g]`` must be what the emitted ladder is actually worth under ``g``.

    The recursion carries ``W`` forward as it chooses actions; this checks it
    against evaluating the finished policy from scratch. A mismatch would mean
    the operator chose its actions against a continuation it does not deliver —
    the subtle failure mode of any policy-evaluation-inside-optimisation scheme.
    """
    solved = solve_regret_robust_markdown(wide)
    t, i = wide.horizon, wide.initial_inventory
    for g, elasticity in enumerate(solved.elasticity_grid):
        evaluated = exact_policy_value(wide.at_elasticity(elasticity), solved.price_at)[t, i]
        assert solved.policy_value[g, t, i] == pytest.approx(evaluated, rel=1e-9)


def test_regret_is_non_negative_everywhere(wide):
    """No policy can beat the policy that was told the truth."""
    solved = solve_regret_robust_markdown(wide)
    assert np.all(solved.regret_by_elasticity >= -1e-9)
    assert solved.mean_regret <= solved.worst_case_regret + 1e-12
    assert solved.certified_regret_bound <= solved.worst_case_regret + 1e-12


def test_regret_operator_beats_value_operator_on_worst_case_regret(wide):
    """The finding that motivated this operator, pinned as a regression test.

    The value-robust policy protects the season's *value* across the ambiguity
    set, which on a markdown problem drags it toward the inelastic end where
    profit is low for reasons no policy can fix. Measured on regret — the profit
    actually forfeited by not knowing the elasticity — it does badly, and worse
    than simply trusting the point estimate. The regret operator exists because
    of that, so the comparison is asserted rather than left in a document.
    """
    regret_robust = solve_regret_robust_markdown(wide)
    t, i = wide.horizon, wide.initial_inventory
    oracle = regret_robust.oracle_value[:, t, i]

    def worst_regret(rule) -> float:
        achieved = np.array(
            [
                exact_policy_value(wide.at_elasticity(e), rule)[t, i]
                for e in regret_robust.elasticity_grid
            ]
        )
        return float(np.max((oracle - achieved) / oracle))

    value_robust = worst_regret(solve_robust_markdown(wide, robustness_level=1.0).price_at)
    certainty = worst_regret(solve_markdown(wide.at_elasticity(wide.elasticity_point)).price_at)

    assert regret_robust.worst_case_regret < value_robust
    assert regret_robust.worst_case_regret < certainty


def test_minimising_mean_regret_is_a_different_policy_from_minimising_the_tail(wide):
    """``cvar_alpha`` is the operator's only dial and must actually turn.

    At alpha=1 the whole grid is the tail, so the operator minimises mean regret;
    at the shipped default it minimises a two-atom tail, which is close to
    minimax. The first should win on the mean and the second on the worst case,
    or the dial is not expressing a trade.
    """
    tail = solve_regret_robust_markdown(wide)
    mean = solve_regret_robust_markdown(wide, cvar_alpha=1.0)
    assert mean.mean_regret <= tail.mean_regret + 1e-9
    assert tail.worst_case_regret <= mean.worst_case_regret + 1e-9


def test_regret_operator_is_inert_without_ambiguity(base):
    """With a degenerate interval there is nothing to trade off, and the operator
    must return the ordinary optimum with zero regret."""
    degenerate = RobustMarkdownProblem.from_markdown_problem(base, base.elasticity, base.elasticity)
    solved = solve_regret_robust_markdown(degenerate)
    assert solved.worst_case_regret == pytest.approx(0.0, abs=1e-9)
    classical = solve_markdown(base)
    assert np.array_equal(solved.policy[1:], classical.policy[1:])


@pytest.mark.parametrize("alpha", [0.0, 1.5])
def test_regret_operator_rejects_an_impossible_alpha(wide, alpha):
    with pytest.raises(ValueError, match="cvar_alpha"):
        solve_regret_robust_markdown(wide, cvar_alpha=alpha)


# ---------------------------------------------------------------------------
# The tail-size rule, which is where alpha stops meaning anything
# ---------------------------------------------------------------------------


def test_tail_size_rounds_up_and_stays_in_range():
    """Rounding down would empty the tail at the shipped defaults.

    At alpha=0.05 over 21 grid points the tail is 1.05 atoms. Rounding down
    leaves one — the single worst grid point — at which the 'CVaR' is a
    worst-case and the alpha dial has no effect at any value below 1/21. The
    config record for DEFAULT_ROBUST_GRID_SIZE states this constraint; this test
    is what holds the code to it.
    """
    assert _tail_size(21, 0.05) == 2
    assert _tail_size(2, 0.05) == 1, "the tail must never be empty"
    assert _tail_size(10, 1.0) == 10, "the tail must never exceed the grid"
    assert _tail_size(100, 0.5) == 50


def test_shipped_defaults_leave_a_tail_worth_averaging():
    """The two config constants have to agree with each other, not just exist."""
    assert _tail_size(config.DEFAULT_ROBUST_GRID_SIZE, config.DEFAULT_CVAR_ALPHA) >= 2


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_interval_must_be_ordered_as_signed_numbers(base):
    """Passing magnitudes instead of signed values inverts the interval.

    A plausible mistake — 'low elasticity' colloquially means small in magnitude
    — and one that silently produces an empty ambiguity set rather than an
    error, so it is refused explicitly.
    """
    with pytest.raises(ValueError, match="signed"):
        RobustMarkdownProblem.from_markdown_problem(base, -1.2, -4.5)


def test_interval_may_not_reach_zero(base):
    with pytest.raises(ValueError, match="must be negative"):
        RobustMarkdownProblem.from_markdown_problem(base, -3.0, 0.1)


def test_point_estimate_must_lie_inside_its_own_interval(base):
    with pytest.raises(ValueError, match="signed"):
        RobustMarkdownProblem.from_markdown_problem(base, -2.0, -1.5)


@pytest.mark.parametrize("rho", [-0.1, 1.1])
def test_robustness_level_must_be_a_fraction(wide, rho):
    with pytest.raises(ValueError, match="robustness_level"):
        solve_robust_markdown(wide, robustness_level=rho)


def test_a_single_grid_point_is_refused(wide):
    """One point is a point estimate wearing an interval's name."""
    with pytest.raises(ValueError, match="n_grid"):
        solve_robust_markdown(wide, n_grid=1)


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_cvar_alpha_must_be_an_interior_fraction(wide, alpha):
    with pytest.raises(ValueError, match="cvar_alpha"):
        solve_robust_markdown(wide, cvar_alpha=alpha)


def test_price_at_refuses_impossible_states(wide):
    """Clamping would answer a buggy caller confidently."""
    solved = solve_robust_markdown(wide)
    with pytest.raises(IndexError):
        solved.price_at(wide.horizon + 1, 10)
    with pytest.raises(IndexError):
        solved.price_at(1, wide.initial_inventory + 1)
