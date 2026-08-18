"""
Decision layer tests.

The phase-5 gate is that the engine beats cost-plus and competitor-match on
realised profit. That is checkable here in a way it never is in production,
because the synthetic demand curve is known: realised profit can be computed
exactly rather than estimated, so "better" means better and not "scored itself
higher".

The other load-bearing test is
:func:`test_engine_finds_the_analytic_optimum`. A constant-elasticity curve has
a closed-form profit maximum, so the optimiser can be checked against arithmetic
instead of against its own output. It is the test that caught the common-random-
numbers defect described below.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from prismprice.decision import (
    DecisionEngine,
    ObjectiveWeights,
    competitor_match_price,
    cost_plus_price,
    generate_ladder,
    sample_demand,
    score_candidate,
    snap_to_ending,
)
from prismprice.governance.schemas import DegradationRung, PriceRequest

AS_OF = datetime(2026, 8, 17, 2, tzinfo=timezone.utc)
ELASTICITY = -1.8
BASE_PRICE = 30.0
BASE_UNITS = 100.0


def true_demand(price: float, elasticity: float = ELASTICITY) -> float:
    """Constant-elasticity demand. Known, so the optimum is arithmetic."""
    return BASE_UNITS * (price / BASE_PRICE) ** elasticity


def demand_at(price: float) -> tuple[float, float, float]:
    q = true_demand(price)
    return (q * 0.7, q, q * 1.4)


def analytic_profit(price: float, unit_cost: float) -> float:
    return true_demand(price) * (price - unit_cost)


def make_request(**overrides) -> PriceRequest:
    payload = {
        "sku": "SKU-001",
        "as_of": AS_OF,
        "current_price": BASE_PRICE,
        "unit_cost": 15.0,
    }
    payload.update(overrides)
    return PriceRequest(**payload)


# ---------------------------------------------------------------------------
# The phase-5 gate
# ---------------------------------------------------------------------------


def test_engine_beats_cost_plus_and_competitor_match():
    """Phase-5 gate, scored on realised profit under the true demand curve.

    Both baselines are given every advantage: they price without a movement cap
    and pay no simulation error. The engine still has to win, because scoring
    itself higher than a baseline it also scored would prove nothing.
    """
    unit_cost = 15.0
    competitor = 27.50

    request = make_request(unit_cost=unit_cost, competitor_price=competitor)
    engine = DecisionEngine()
    recommended = engine.decide(request, demand_at=demand_at).recommended_price

    engine_profit = analytic_profit(recommended, unit_cost)
    cost_plus_profit = analytic_profit(cost_plus_price(unit_cost), unit_cost)
    match_profit = analytic_profit(competitor_match_price(competitor), unit_cost)

    assert engine_profit > cost_plus_profit, (
        f"engine {engine_profit:.2f} at {recommended} did not beat cost-plus "
        f"{cost_plus_profit:.2f} at {cost_plus_price(unit_cost):.2f}"
    )
    assert engine_profit > match_profit, (
        f"engine {engine_profit:.2f} did not beat competitor-match {match_profit:.2f}"
    )


def test_engine_beats_baselines_across_a_category():
    """Phase-5 gate: category contribution beats cost-plus and competitor-match.

    Two things about the shape of this test are deliberate, and both were forced
    by watching it fail honestly.

    **It runs over successive decisions, not one.** The movement cap bounds each
    move to +/-30%, so at elasticity -3.0 the profit-maximising price of 18.00 is
    simply unreachable from 30.00 in a single step while unconstrained cost-plus
    sits at 16.20. Scoring one capped step against an uncapped baseline measures
    the cap, not the optimiser.

    **It is scored on the category total, not on every SKU.** A baseline can be
    accidentally excellent on one item: at elasticity -1.8 the true optimum is
    27.00, the competitor happens to sit at 27.50, and the .95/.99 ending rule
    puts the engine at 27.95 — a loss of 0.06%. Demanding a win on every SKU
    would mean demanding that no fixed price ever lands near an optimum by
    chance, which is a claim about luck rather than about the optimiser. The
    guarantee that matters commercially is the category, plus a floor on how
    badly any single SKU may lose.
    """
    unit_cost = 12.0
    competitor = 27.50
    elasticities = [-1.2, -1.5, -1.8, -2.2, -2.6, -3.0]

    engine_total = cost_plus_total = match_total = 0.0
    worst_relative_loss = 0.0

    for elasticity in elasticities:

        def curve(price: float, e: float = elasticity) -> tuple[float, float, float]:
            q = true_demand(price, e)
            return (q * 0.7, q, q * 1.4)

        def realised(price: float, e: float = elasticity) -> float:
            return true_demand(price, e) * (price - unit_cost)

        price = BASE_PRICE
        for _ in range(8):
            request = make_request(unit_cost=unit_cost, current_price=price, movement_cap_pct=0.30)
            price = DecisionEngine().decide(request, demand_at=curve).recommended_price

        engine_profit = realised(price)
        best_baseline = max(
            realised(cost_plus_price(unit_cost)), realised(competitor_match_price(competitor))
        )
        engine_total += engine_profit
        cost_plus_total += realised(cost_plus_price(unit_cost))
        match_total += realised(competitor_match_price(competitor))
        worst_relative_loss = max(
            worst_relative_loss, (best_baseline - engine_profit) / best_baseline
        )

    assert engine_total > cost_plus_total, (
        f"category contribution {engine_total:.0f} did not beat cost-plus {cost_plus_total:.0f}"
    )
    assert engine_total > match_total, (
        f"category contribution {engine_total:.0f} did not beat competitor-match {match_total:.0f}"
    )
    assert worst_relative_loss < 0.01, (
        f"lost {worst_relative_loss:.2%} to a baseline on one SKU; a near-tie from the "
        f"price-ending rule is acceptable, a real loss is not"
    )


def test_repeated_decisions_converge_and_then_hold():
    """The movement cap makes pricing a walk, so it must terminate.

    A policy that keeps moving forever burns its change-frequency budget and
    destroys the reference price the retention model depends on. Reaching the
    optimum and then choosing "no change" is what makes the current price's
    presence on the ladder load-bearing.
    """
    unit_cost = 12.0
    price = BASE_PRICE
    trajectory = []
    for _ in range(12):
        request = make_request(unit_cost=unit_cost, current_price=price, movement_cap_pct=0.30)
        price = DecisionEngine().decide(request, demand_at=demand_at).recommended_price
        trajectory.append(price)

    assert trajectory[-1] == trajectory[-2] == trajectory[-3], (
        f"price never settled: {trajectory[-5:]}"
    )


def test_engine_finds_the_analytic_optimum():
    """The test that caught the common-random-numbers defect.

    Scoring each candidate against independent draws gave Monte-Carlo error of
    SD ~10.8 per candidate while adjacent ladder rungs differ by only 5-10, so
    the argmax was chosen by sampling noise and disagreed with arithmetic.
    """
    unit_cost = 15.0
    request = make_request(unit_cost=unit_cost)
    outcome = DecisionEngine().decide(request, demand_at=demand_at)

    best_on_ladder = max(outcome.ladder, key=lambda p: analytic_profit(p, unit_cost))
    assert outcome.recommended_price == pytest.approx(best_on_ladder), (
        f"recommended {outcome.recommended_price}, arithmetic says {best_on_ladder}"
    )


def test_scores_are_ordered_like_the_true_profit_curve():
    """Common random numbers should make the J curve track the real one."""
    outcome = DecisionEngine().decide(make_request(), demand_at=demand_at)
    scored = sorted(outcome.outcomes, key=lambda o: o.price)
    simulated = [o.expected_profit for o in scored]
    analytic = [analytic_profit(o.price, 15.0) for o in scored]
    correlation = float(np.corrcoef(simulated, analytic)[0, 1])
    assert correlation > 0.98, f"simulated profit tracks truth at only r={correlation:.3f}"


# ---------------------------------------------------------------------------
# Ladder
# ---------------------------------------------------------------------------


def test_ladder_always_contains_the_current_price():
    """An optimiser that cannot choose 'leave it alone' moves the price forever."""
    ladder = generate_ladder(29.99, movement_cap_pct=0.15)
    assert 29.99 in ladder


def test_ladder_respects_the_movement_cap():
    ladder = generate_ladder(30.0, movement_cap_pct=0.10)
    assert min(ladder) >= 30.0 * 0.90 - 1.0
    assert max(ladder) <= 30.0 * 1.10 + 1.0


def test_ladder_prices_are_publishable():
    for price in generate_ladder(30.0, allowed_endings=(95, 99)):
        assert round(price * 100) % 100 in (95, 99), f"{price} is not a permitted ending"


def test_ladder_is_sorted_and_unique():
    ladder = generate_ladder(30.0, n_candidates=15)
    assert ladder == sorted(ladder)
    assert len(ladder) == len(set(ladder))


def test_ladder_honours_a_floor():
    ladder = generate_ladder(30.0, movement_cap_pct=0.30, floor=28.0)
    assert all(p >= 28.0 for p in ladder)


def test_ladder_is_never_empty_even_when_bounds_exclude_everything():
    """An empty ladder would surface downstream as an unexplained crash; a
    single infeasible candidate gets an explicit guardrail rejection instead."""
    ladder = generate_ladder(30.0, movement_cap_pct=0.05, floor=1000.0)
    assert len(ladder) == 1


def test_snapping_picks_the_nearest_permitted_ending():
    assert snap_to_ending(31.37, (95, 99)) == pytest.approx(30.99)
    assert snap_to_ending(31.80, (95, 99)) == pytest.approx(31.95)
    assert snap_to_ending(31.97, (95, 99)) == pytest.approx(31.95)


def test_snapping_can_be_disabled():
    assert snap_to_ending(31.374, None) == pytest.approx(31.37)


def test_ladder_rejects_impossible_inputs():
    with pytest.raises(ValueError, match="current_price must be > 0"):
        generate_ladder(0.0)
    with pytest.raises(ValueError, match="movement_cap_pct must be >= 0"):
        generate_ladder(30.0, movement_cap_pct=-0.1)


# ---------------------------------------------------------------------------
# Sampling and the objective
# ---------------------------------------------------------------------------


def test_samples_reproduce_the_quantiles_they_were_built_from():
    draws = sample_demand(70.0, 100.0, 140.0, n_draws=200_000, rng=np.random.default_rng(0))
    assert np.quantile(draws, 0.10) == pytest.approx(70.0, rel=0.03)
    assert np.quantile(draws, 0.50) == pytest.approx(100.0, rel=0.03)
    assert np.quantile(draws, 0.90) == pytest.approx(140.0, rel=0.03)


def test_sampling_preserves_asymmetry():
    """L2 conformalises each tail separately; a single sigma would undo that."""
    draws = sample_demand(90.0, 100.0, 200.0, n_draws=200_000, rng=np.random.default_rng(0))
    lower_gap = 100.0 - float(np.quantile(draws, 0.10))
    upper_gap = float(np.quantile(draws, 0.90)) - 100.0
    assert upper_gap > 5 * lower_gap, "the asymmetric interval collapsed to a symmetric one"


def test_samples_are_never_negative():
    draws = sample_demand(0.0, 5.0, 40.0, n_draws=10_000, rng=np.random.default_rng(1))
    assert np.all(draws > 0.0)


def test_crossed_quantiles_are_refused():
    """An out-of-order interval is an upstream defect; sorting it hides that."""
    with pytest.raises(ValueError, match="quantiles must be ordered"):
        sample_demand(140.0, 100.0, 70.0, n_draws=10)


def test_zero_median_is_refused():
    with pytest.raises(ValueError, match="p50 must be > 0"):
        sample_demand(0.0, 0.0, 1.0, n_draws=10)


def test_cvar_sits_below_the_expectation():
    outcome = score_candidate(30.0, 15.0, (70.0, 100.0, 140.0), rng=np.random.default_rng(0))
    assert outcome.cvar < outcome.expected_profit
    assert outcome.tail_shortfall > 0


def test_lambda_moves_the_score_by_the_stated_trade():
    """lambda is an exchange rate, so its effect must be exactly linear."""
    common = dict(price=30.0, unit_cost=15.0, demand_quantiles=(70.0, 100.0, 140.0))
    neutral = score_candidate(
        **common,
        delta_clv=-50.0,
        weights=ObjectiveWeights(clv_weight_lambda=0.0, cvar_weight_gamma=0.0),
        rng=np.random.default_rng(0),
    )
    weighted = score_candidate(
        **common,
        delta_clv=-50.0,
        weights=ObjectiveWeights(clv_weight_lambda=0.4, cvar_weight_gamma=0.0),
        rng=np.random.default_rng(0),
    )
    assert weighted.j_score == pytest.approx(neutral.j_score + 0.4 * -50.0)


def test_gamma_penalises_only_the_shortfall():
    common = dict(price=30.0, unit_cost=15.0, demand_quantiles=(70.0, 100.0, 140.0))
    risk_neutral = score_candidate(
        **common, weights=ObjectiveWeights(cvar_weight_gamma=0.0), rng=np.random.default_rng(0)
    )
    risk_averse = score_candidate(
        **common, weights=ObjectiveWeights(cvar_weight_gamma=0.5), rng=np.random.default_rng(0)
    )
    assert risk_averse.j_score == pytest.approx(
        risk_neutral.j_score - 0.5 * risk_neutral.tail_shortfall
    )
    assert risk_averse.j_score < risk_neutral.j_score


def test_inventory_cap_makes_expectation_non_linear():
    """The reason the objective simulates instead of multiplying.

    With a stock cap, E[min(q, stock)] < min(E[q], stock): the upside is
    truncated but the downside is not, so a point forecast overstates sales.
    """
    uncapped = score_candidate(30.0, 15.0, (70.0, 100.0, 140.0), rng=np.random.default_rng(0))
    capped = score_candidate(
        30.0, 15.0, (70.0, 100.0, 140.0), units_available=100.0, rng=np.random.default_rng(0)
    )
    assert capped.expected_units < uncapped.expected_units
    assert capped.expected_units < 100.0


def test_shadow_price_raises_the_effective_cost():
    """Scarcity enters the margin, not a side constraint."""
    plentiful = score_candidate(30.0, 15.0, (70.0, 100.0, 140.0), rng=np.random.default_rng(0))
    scarce = score_candidate(
        30.0, 15.0, (70.0, 100.0, 140.0), inventory_shadow_price=5.0, rng=np.random.default_rng(0)
    )
    assert scarce.expected_profit < plentiful.expected_profit


def test_negative_weights_are_refused():
    with pytest.raises(ValueError, match="lambda must be >= 0"):
        ObjectiveWeights(clv_weight_lambda=-0.1)
    with pytest.raises(ValueError, match="gamma must be >= 0"):
        ObjectiveWeights(cvar_weight_gamma=-0.1)
    with pytest.raises(ValueError, match="cvar_alpha must be in"):
        ObjectiveWeights(cvar_alpha=1.5)


# ---------------------------------------------------------------------------
# Guardrails bound the optimiser, not the other way round
# ---------------------------------------------------------------------------


def test_recommendation_is_always_in_the_feasible_set():
    request = make_request(unit_cost=28.0, margin_floor_pct=0.05)
    outcome = DecisionEngine().decide(request, demand_at=demand_at)
    if outcome.feasible_prices:
        assert outcome.recommended_price in outcome.feasible_prices


def test_no_price_below_cost_can_be_recommended():
    """A model bug may produce a wrong score; it must not produce an illegal price."""
    request = make_request(unit_cost=29.0)
    outcome = DecisionEngine().decide(request, demand_at=demand_at)
    assert outcome.recommended_price >= 29.0 or outcome.degraded


def test_empty_feasible_set_degrades_to_the_previous_price():
    """Rung 4: no candidate is legal, so the engine holds rather than relaxing
    a constraint or raising."""
    request = make_request(unit_cost=40.0)
    outcome = DecisionEngine().decide(request, demand_at=demand_at)
    assert outcome.recommended_price == pytest.approx(request.current_price)
    assert outcome.record.degradation_reason_code is DegradationRung.FALLBACK_RULE_ENGINE
    assert outcome.best_outcome() is None, "a fallback must not look like an optimum"


def test_stale_competitor_feed_degrades_to_rung_two():
    request = make_request(
        competitor_price=31.0, competitor_observed_at=AS_OF - timedelta(hours=72)
    )
    outcome = DecisionEngine().decide(request, demand_at=demand_at)
    assert outcome.record.degradation_reason_code is DegradationRung.WARN_STALE_COMPETITOR


# ---------------------------------------------------------------------------
# The audit record
# ---------------------------------------------------------------------------


def test_record_carries_every_candidate_it_scored():
    outcome = DecisionEngine().decide(make_request(), demand_at=demand_at)
    assert len(outcome.record.candidates) == len(outcome.ladder)
    assert {c.price for c in outcome.record.candidates} == set(outcome.ladder)


def test_record_reports_the_models_own_quantiles():
    """Reporting the simulated mean three times would claim a certainty the
    forecast never had."""
    outcome = DecisionEngine().decide(make_request(), demand_at=demand_at)
    quantiles = outcome.record.estimates.demand_quantiles
    expected = demand_at(outcome.recommended_price)
    assert quantiles.p10 == pytest.approx(expected[0])
    assert quantiles.p90 == pytest.approx(expected[2])
    assert quantiles.p10 < quantiles.p50 < quantiles.p90


def test_record_pins_seed_and_policy_version():
    """A record that defaults its own version is not reconstructible."""
    record = (
        DecisionEngine(policy_version="1.4.2", seed=99)
        .decide(make_request(), demand_at=demand_at)
        .record
    )
    assert record.policy_version == "1.4.2"
    assert record.seed == 99


def test_record_captures_cvar_alpha_explicitly():
    """A record that assumes 5% cannot describe a run that used 1%."""
    engine = DecisionEngine(weights=ObjectiveWeights(cvar_alpha=0.01))
    record = engine.decide(make_request(), demand_at=demand_at).record
    assert all(c.cvar_alpha == pytest.approx(0.01) for c in record.candidates)


def test_decisions_are_reproducible_from_the_seed():
    first = DecisionEngine(seed=7).decide(make_request(), demand_at=demand_at)
    second = DecisionEngine(seed=7).decide(make_request(), demand_at=demand_at)
    assert first.recommended_price == second.recommended_price
    assert [o.j_score for o in first.outcomes] == [o.j_score for o in second.outcomes]


def test_delta_clv_pulls_the_recommendation_down():
    """The relationship term must be able to overrule the transaction one.

    Without it the optimiser takes the margin; with a steep enough Delta-CLV
    penalty on price rises it should not.
    """
    request = make_request()

    def clv_at(price: float) -> float:
        return -400.0 * max(price - BASE_PRICE, 0.0)

    transactional = DecisionEngine().decide(request, demand_at=demand_at).recommended_price
    relational = (
        DecisionEngine(weights=ObjectiveWeights(clv_weight_lambda=1.0))
        .decide(request, demand_at=demand_at, delta_clv_at=clv_at)
        .recommended_price
    )

    assert relational < transactional, (
        "a large Delta-CLV penalty on price rises did not restrain the optimiser"
    )


def test_omitting_delta_clv_records_it_as_absent():
    record = DecisionEngine().decide(make_request(), demand_at=demand_at).record
    assert record.estimates.delta_clv is None


def test_baselines_reject_impossible_inputs():
    with pytest.raises(ValueError, match="unit_cost must be > 0"):
        cost_plus_price(0.0)
    with pytest.raises(ValueError, match="competitor_price must be > 0"):
        competitor_match_price(0.0)
