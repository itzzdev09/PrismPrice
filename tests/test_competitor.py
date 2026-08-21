"""
Competitor reaction and price-war tests.

Two claims are checked, and they need different evidence.

**The estimator recovers a known reaction.** The competitor is simulated with a
chosen ``beta`` and ``gamma``, so the fitted coefficients can be scored against
arithmetic. Averaged over independent histories, because a single fit lands
anywhere in a band roughly two standard errors wide.

**The stability analysis calls price wars correctly.** That is a claim about the
dynamics rather than about a fit, so it is checked by playing both policies
forward and looking at where prices end up — including the case that matters
most, where a *mathematically stable* system still walks to the floor because
its fixed point is infeasible.
"""

from __future__ import annotations

import numpy as np
import pytest

from prismprice.estimation.competitor import (
    ReactionFunction,
    estimate_reaction,
    find_equilibrium,
    simulate_price_path,
)

TRUE_BETA = 0.45
TRUE_GAMMA = 0.40
TRUE_INTERCEPT = 6.0


def simulate_history(
    seed: int,
    n: int = 400,
    common_shock: np.ndarray | None = None,
    shock_loading: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """A history where the competitor's reaction is known by construction."""
    rng = np.random.default_rng(seed)
    ours = np.empty(n)
    theirs = np.empty(n)
    ours[0], theirs[0] = 30.0, 29.0
    shock = common_shock if common_shock is not None else np.zeros(n)

    for t in range(1, n):
        ours[t] = 28.0 + 0.15 * theirs[t - 1] + shock_loading * shock[t] + rng.normal(0, 0.8)
        theirs[t] = (
            TRUE_INTERCEPT
            + TRUE_BETA * ours[t - 1]
            + TRUE_GAMMA * theirs[t - 1]
            + shock_loading * shock[t]
            + rng.normal(0, 0.5)
        )
    return ours, theirs


def reaction(beta: float, gamma: float, intercept: float = 2.0) -> ReactionFunction:
    return ReactionFunction(
        intercept=intercept,
        reaction=beta,
        persistence=gamma,
        reaction_std_error=0.02,
        r_squared=0.8,
        n_observations=300,
    )


# ---------------------------------------------------------------------------
# Recovering a known reaction
# ---------------------------------------------------------------------------


def test_recovers_the_known_reaction_coefficient():
    """Averaged over histories, because one fit proves nothing either way."""
    betas = [estimate_reaction(*simulate_history(seed)).reaction for seed in range(15)]
    assert float(np.mean(betas)) == pytest.approx(TRUE_BETA, abs=0.05)


def test_recovers_the_known_persistence():
    gammas = [estimate_reaction(*simulate_history(seed)).persistence for seed in range(15)]
    assert float(np.mean(gammas)) == pytest.approx(TRUE_GAMMA, abs=0.05)


def test_reported_standard_error_matches_the_actual_spread():
    """An interval that disagrees with the estimator's real variability is
    worse than none, because it will be believed."""
    fits = [estimate_reaction(*simulate_history(seed)) for seed in range(20)]
    spread = float(np.std([f.reaction for f in fits], ddof=1))
    reported = float(np.mean([f.reaction_std_error for f in fits]))
    assert 0.5 < reported / spread < 2.0


def test_long_run_response_exceeds_the_immediate_one():
    """A competitor who follows 45% now but holds 40% of their own price ends up
    following 75%. Judging war risk on the headline number would call a
    divergent pair stable."""
    fitted = reaction(beta=0.45, gamma=0.40)
    assert fitted.long_run_response == pytest.approx(0.75)
    assert fitted.long_run_response > fitted.reaction


def test_a_perfectly_sticky_competitor_never_settles():
    assert reaction(beta=0.5, gamma=1.0).long_run_response == float("inf")


def test_no_reaction_is_reported_as_insignificant():
    """A competitor who ignores us must not be modelled as reacting."""
    rng = np.random.default_rng(1)
    n = 400
    ours = 30.0 + rng.normal(0, 2.0, n)
    theirs = 28.0 + rng.normal(0, 2.0, n)
    assert not estimate_reaction(ours, theirs).is_significant


def test_a_real_reaction_is_reported_as_significant():
    assert estimate_reaction(*simulate_history(3)).is_significant


# ---------------------------------------------------------------------------
# The identification problem
# ---------------------------------------------------------------------------


def test_a_shared_shock_inflates_the_estimated_reaction():
    """The same confounding elasticity has, in a different costume.

    Two firms facing the same cost index move together without either reacting
    to the other, and the bias runs *upward* — toward declaring a price war that
    is not happening.
    """
    rng = np.random.default_rng(4)
    shock = np.cumsum(rng.normal(0, 1.0, 400))

    clean = [estimate_reaction(*simulate_history(s)).reaction for s in range(8)]
    confounded = [
        estimate_reaction(*simulate_history(s, common_shock=shock, shock_loading=0.5)).reaction
        for s in range(8)
    ]
    assert float(np.mean(confounded)) > float(np.mean(clean)), (
        "a shared shock should look like reaction; if it does not, the test panel "
        "stopped being confounded"
    )


def test_controlling_for_the_shared_shock_removes_the_inflation():
    """What `controls` is for, measured rather than asserted."""
    rng = np.random.default_rng(4)
    shock = np.cumsum(rng.normal(0, 1.0, 400))

    uncontrolled, controlled = [], []
    for seed in range(8):
        ours, theirs = simulate_history(seed, common_shock=shock, shock_loading=0.5)
        uncontrolled.append(estimate_reaction(ours, theirs).reaction)
        controlled.append(estimate_reaction(ours, theirs, controls={"cost_index": shock}).reaction)

    assert abs(float(np.mean(controlled)) - TRUE_BETA) < abs(
        float(np.mean(uncontrolled)) - TRUE_BETA
    )


def test_controls_are_recorded_on_the_fit():
    ours, theirs = simulate_history(1)
    fitted = estimate_reaction(ours, theirs, controls={"cost_index": np.zeros(400)})
    assert fitted.control_names == ("cost_index",)


# ---------------------------------------------------------------------------
# Stability and price wars
# ---------------------------------------------------------------------------


def test_ignoring_the_competitor_is_always_stable():
    """Slope zero means our price does not depend on theirs, so nothing can
    amplify."""
    analysis = find_equilibrium(30.0, 0.0, reaction(0.95, 0.0), price_floor=12.0)
    assert analysis.spiral_coefficient == pytest.approx(0.0)
    assert analysis.is_equilibrium
    assert analysis.war_risk == "STABLE"


def test_mutual_matching_walks_prices_to_the_floor():
    """The price war, played forward.

    Both sides undercut by 2. Neither ever chooses a low price; the pair simply
    has no feasible resting point above the floor.
    """
    aggressive = reaction(beta=0.95, gamma=0.0, intercept=-2.0)
    ours, theirs = simulate_price_path(
        lambda t: t - 2.0, aggressive, 30.0, 30.0, periods=20, price_floor=12.0
    )
    assert ours[-1] == pytest.approx(12.0)
    assert theirs[-1] == pytest.approx(12.0)
    assert ours[3] < ours[0], "prices did not fall at all"


def test_a_stable_system_can_still_be_a_price_war():
    """The case the spiral coefficient alone misses, and the reason
    `is_equilibrium` needs three conditions rather than one.

    With b = 1.0 and d = 0.95 the product is below one, so the dynamics damp and
    `is_stable` is True — but the fixed point they damp toward sits at roughly
    -80, far below any price anyone can charge. The pair converges to the
    guardrail floor, and only `settled_at_bound` distinguishes that from having
    settled on an equilibrium.
    """
    analysis = find_equilibrium(
        our_intercept=-2.0,
        our_slope=1.0,
        their_reaction=reaction(beta=0.95, gamma=0.0, intercept=-2.0),
        price_floor=12.0,
    )
    assert analysis.is_stable, "this configuration is mathematically stable"
    assert analysis.settled_at_bound, "yet it comes to rest on the floor"
    assert not analysis.is_equilibrium, "resting on a constraint is not an equilibrium"
    assert analysis.our_price == pytest.approx(12.0)


def test_divergent_reactions_are_flagged():
    analysis = find_equilibrium(
        our_intercept=0.0,
        our_slope=1.5,
        their_reaction=reaction(beta=0.9, gamma=0.2),
        price_floor=12.0,
        price_ceiling=60.0,
    )
    assert abs(analysis.spiral_coefficient) >= 1.0
    assert not analysis.is_stable
    assert analysis.war_risk == "DIVERGENT"


def test_divergence_is_bounded_rather_than_running_to_infinity():
    """An unbounded run reports a resting price of 1e11, which means nothing.
    Real prices are bounded by the margin floor and the competitive ceiling."""
    analysis = find_equilibrium(
        our_intercept=0.0,
        our_slope=1.5,
        their_reaction=reaction(beta=0.9, gamma=0.2),
        price_floor=12.0,
        price_ceiling=60.0,
    )
    assert 12.0 <= analysis.our_price <= 60.0
    assert analysis.settled_at_bound


@pytest.mark.parametrize(
    ("slope", "beta", "expected"),
    [
        (0.0, 0.9, "STABLE"),
        (0.3, 0.9, "STABLE"),
        (0.7, 0.9, "DAMPED"),
        (1.0, 0.9, "FRAGILE"),
        (1.5, 0.9, "DIVERGENT"),
    ],
)
def test_war_risk_bands(slope, beta, expected):
    """Banded because the action differs by band: below 0.5 nothing changes,
    above 1.0 the matching rule has to be abandoned, and between them a human
    should be looking."""
    analysis = find_equilibrium(10.0, slope, reaction(beta, 0.0), price_floor=1.0)
    assert analysis.war_risk == expected


def test_a_stable_pair_settles_at_its_fixed_point():
    """Closed form: p* = (a + b*c) / (1 - b*d), checked against the iteration."""
    a, b = 10.0, 0.5
    their = reaction(beta=0.4, gamma=0.0, intercept=8.0)
    analysis = find_equilibrium(a, b, their, price_floor=0.0, price_ceiling=1e6)

    expected = (a + b * their.intercept) / (1.0 - b * their.reaction)
    assert analysis.our_price == pytest.approx(expected, rel=1e-6)
    assert analysis.is_equilibrium


def test_equilibrium_serialises_for_the_decision_record():
    analysis = find_equilibrium(10.0, 0.5, reaction(0.4, 0.0), price_floor=1.0)
    record = analysis.as_dict()
    assert set(record) >= {"spiral_coefficient", "war_risk", "is_equilibrium", "settled_at_bound"}


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_mismatched_series_are_refused():
    with pytest.raises(ValueError, match="same length"):
        estimate_reaction(np.zeros(20), np.zeros(19))


def test_too_short_a_history_is_refused():
    """A coefficient from five observations is noise with a standard error."""
    with pytest.raises(ValueError, match="at least 10 observations"):
        estimate_reaction(np.zeros(5), np.zeros(5))


def test_a_control_of_the_wrong_length_is_refused():
    ours, theirs = simulate_history(1, n=50)
    with pytest.raises(ValueError, match="must match the price series length"):
        estimate_reaction(ours, theirs, controls={"bad": np.zeros(10)})


def test_simulation_requires_at_least_one_period():
    with pytest.raises(ValueError, match="periods must be >= 1"):
        simulate_price_path(lambda t: t, reaction(0.5, 0.0), 30.0, 30.0, periods=0)


def test_reaction_serialises_with_its_long_run_response():
    record = estimate_reaction(*simulate_history(1)).as_dict()
    assert "long_run_response" in record
    assert "is_significant" in record
