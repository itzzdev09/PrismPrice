"""
Circuit breaker tests.

These differ from every other test in the suite in what they are protecting
against. The guardrail tests assert that no *individual* price is illegal; these
assert that a batch of individually legal prices can still be stopped. Both
failures are real and only one of them is visible per-decision.

The fail-closed tests are the ones to keep honest. A breaker that cannot
evaluate must halt, because "we could not check" is not evidence of safety — the
same rule PP-G003 follows for the Omnibus anchor, for the same reason.
"""

from __future__ import annotations

import pytest

from prismprice.observability.alerts import (
    BreakerStatus,
    aggregate_movement,
    evaluate_breakers,
    no_guardrail_violations,
    one_sided_movement,
)

# ---------------------------------------------------------------------------
# One-sided movement: the batch-level fault no guardrail can see
# ---------------------------------------------------------------------------


def test_a_normal_mixed_run_publishes():
    """A healthy run touches a small share of the catalogue, in both directions."""
    previous = [10.0] * 100
    new = [10.0] * 80 + [11.0] * 10 + [9.0] * 10
    assert one_sided_movement(previous, new).status is BreakerStatus.PASS


def test_a_wholesale_move_in_one_direction_halts():
    """Every individual price can be legal, well-scored and correctly
    guardrailed, and the batch still be a catastrophe — an inverted sign or a
    bad cost feed moves everything the same way."""
    previous = [10.0] * 100
    new = [9.0] * 100
    result = one_sided_movement(previous, new)
    assert result.halted
    assert result.observed == pytest.approx(1.0)
    assert "down" in result.detail


def test_a_few_coordinated_moves_are_not_a_stampede():
    """Twelve SKUs out of a thousand all moving down is not the batch-level
    fault this breaker is for.

    Basing the share on the *movers* instead of the catalogue would report 100%
    here and halt — and would also halt every healthy run, since about half the
    movers go each way in a balanced batch. That mistake makes the breaker fire
    constantly, which is the same as not having it.
    """
    previous = [10.0] * 1000
    new = [10.0] * 988 + [9.0] * 12
    result = one_sided_movement(previous, new)
    assert result.observed == pytest.approx(12 / 1000)
    assert not result.halted
    assert "12 of 1000 prices moved" in result.detail


def test_a_run_that_moves_nothing_passes():
    result = one_sided_movement([10.0] * 50, [10.0] * 50)
    assert result.status is BreakerStatus.PASS
    assert result.observed == 0.0


def test_an_upward_stampede_halts_too():
    """A pricing error that raises everything is as much a fault as one that
    cuts everything; only one of them is intuitive."""
    result = one_sided_movement([10.0] * 100, [12.0] * 100)
    assert result.halted
    assert "up" in result.detail


def test_movement_just_below_the_limit_publishes():
    """29% of the catalogue down is under the 30% limit and publishes."""
    previous = [10.0] * 100
    new = [9.0] * 29 + [11.0] * 29 + [10.0] * 42
    assert not one_sided_movement(previous, new, limit=0.30).halted


# ---------------------------------------------------------------------------
# Aggregate basket movement
# ---------------------------------------------------------------------------


def test_a_small_basket_move_publishes():
    assert not aggregate_movement([10.0] * 10, [10.1] * 10, limit=0.03).halted


def test_a_large_basket_move_halts():
    result = aggregate_movement([10.0] * 10, [10.6] * 10, limit=0.03)
    assert result.halted
    assert result.observed == pytest.approx(0.06)


def test_basket_movement_is_volume_weighted():
    """A 20% cut on a SKU nobody buys and the same cut on the best-seller are
    not the same event; an unweighted mean rates them identically."""
    previous = [10.0, 10.0]
    new = [8.0, 10.0]

    tiny_seller = aggregate_movement(previous, new, weights=[1.0, 1000.0], limit=0.03)
    best_seller = aggregate_movement(previous, new, weights=[1000.0, 1.0], limit=0.03)

    assert not tiny_seller.halted
    assert best_seller.halted


def test_a_downward_basket_move_halts_on_magnitude():
    assert aggregate_movement([10.0] * 5, [9.0] * 5, limit=0.03).halted


# ---------------------------------------------------------------------------
# The breaker that should never fire
# ---------------------------------------------------------------------------


def test_compliant_prices_pass_the_post_hoc_check():
    result = no_guardrail_violations([20.0, 30.0], [10.0, 15.0], margin_floor_pct=0.15)
    assert result.status is BreakerStatus.PASS


def test_a_price_below_the_floor_halts_and_reads_as_an_incident():
    """If this ever fires, the guardrail layer was bypassed — a filter applied
    to the wrong list, a rounding step after the check, a fallback that skipped
    it. The message says so rather than reporting a threshold breach."""
    result = no_guardrail_violations([20.0, 9.0], [10.0, 15.0], margin_floor_pct=0.15)
    assert result.halted
    assert result.observed == 1.0
    assert "should be impossible" in result.detail
    assert "quarantine" in result.detail


def test_exact_floor_is_not_a_violation():
    """Representation error must not manufacture an incident out of a price
    that is exactly legal."""
    assert not no_guardrail_violations([11.5], [10.0], margin_floor_pct=0.15).halted


# ---------------------------------------------------------------------------
# Fail closed
# ---------------------------------------------------------------------------


def test_misaligned_series_halt_rather_than_pass():
    """'We could not check' is not evidence of safety."""
    assert one_sided_movement([10.0, 11.0], [10.0]).halted
    assert aggregate_movement([10.0, 11.0], [10.0]).halted
    assert no_guardrail_violations([10.0, 11.0], [5.0]).halted


def test_empty_input_halts_rather_than_passing_vacuously():
    """An empty batch means the run produced nothing, which is a fault, not a
    clean bill of health."""
    assert one_sided_movement([], []).halted
    assert aggregate_movement([], []).halted
    assert no_guardrail_violations([], []).halted


def test_misaligned_weights_halt():
    assert aggregate_movement([10.0, 11.0], [10.0, 11.0], weights=[1.0]).halted


def test_a_non_positive_basket_halts():
    assert aggregate_movement([0.0, 0.0], [1.0, 1.0]).halted


# ---------------------------------------------------------------------------
# Combining breakers
# ---------------------------------------------------------------------------


def test_all_clear_publishes():
    results = [
        one_sided_movement([10.0] * 100, [10.0] * 80 + [10.1] * 10 + [9.9] * 10),
        aggregate_movement([10.0] * 100, [10.0] * 100),
        no_guardrail_violations([20.0] * 100, [10.0] * 100),
    ]
    may_publish, halting = evaluate_breakers(results)
    assert may_publish
    assert halting == []


def test_any_single_halt_stops_the_run():
    """Breakers are not voted on. Each encodes a condition under which
    publishing is wrong, and a majority being fine does not make the remaining
    one acceptable."""
    results = [
        one_sided_movement([10.0] * 100, [10.0] * 80 + [10.1] * 10 + [9.9] * 10),
        aggregate_movement([10.0] * 100, [10.0] * 100),
        no_guardrail_violations([9.0] * 100, [10.0] * 100),
    ]
    may_publish, halting = evaluate_breakers(results)
    assert not may_publish
    assert len(halting) == 1


def test_every_reason_is_returned_not_just_the_first():
    """So one incident carries all of them, instead of them being discovered
    one run at a time."""
    results = [
        one_sided_movement([10.0] * 100, [9.0] * 100),
        aggregate_movement([10.0] * 100, [9.0] * 100),
        no_guardrail_violations([9.0] * 100, [10.0] * 100),
    ]
    may_publish, halting = evaluate_breakers(results)
    assert not may_publish
    assert len(halting) == 3


def test_no_breakers_at_all_publishes():
    """An explicit empty list is a caller who ran no checks, which this function
    cannot distinguish from a healthy run — the individual breakers fail closed
    on empty data, which is where that is caught."""
    assert evaluate_breakers([])[0]


def test_result_serialises_for_an_incident_record():
    record = one_sided_movement([10.0] * 100, [9.0] * 100).as_dict()
    assert record["status"] == "HALT"
    assert record["halted"] is True
    assert record["detail"]
