"""
KPI tests.

The load-bearing tests here are the ones asserting that a metric *cannot be
gamed*. CPPC is the headline precisely because discount-led volume and margin
extraction both show up in it, so both failure modes are constructed and the
metric is required to fall in each — and gross margin percentage is shown
failing to notice one of them, which is the argument for having CPPC at all.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from prismprice.observability.kpis import (
    contribution_profit_per_customer,
    degradation_distribution,
    gross_margin_pct,
    guardrail_bind_rate,
    price_adherence,
    recommendation_churn,
)

AS_OF = datetime(2026, 8, 17, tzinfo=timezone.utc)


def transactions(
    n_customers: int = 100,
    units_each: float = 2.0,
    price: float = 30.0,
    cost: float = 15.0,
    days_ago: int = 10,
    guests: int = 0,
) -> pd.DataFrame:
    rows = [
        {
            "date": AS_OF - timedelta(days=days_ago),
            "customer_id": f"C-{i:04d}",
            "units": units_each,
            "price": price,
            "unit_cost": cost,
        }
        for i in range(n_customers)
    ]
    rows += [
        {
            "date": AS_OF - timedelta(days=days_ago),
            "customer_id": None,
            "units": units_each,
            "price": price,
            "unit_cost": cost,
        }
        for _ in range(guests)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CPPC cannot be gamed by winning one half of the trade-off
# ---------------------------------------------------------------------------


def test_cppc_is_contribution_over_distinct_customers():
    result = contribution_profit_per_customer(transactions(), AS_OF)
    assert result.value == pytest.approx(2.0 * (30.0 - 15.0))
    assert result.detail["n_customers"] == 100


def test_cppc_falls_when_discounting_buys_volume_that_does_not_pay():
    """Failure mode one: cut price, sell more, earn less per customer."""
    healthy = contribution_profit_per_customer(transactions(price=30.0, units_each=2.0), AS_OF)
    discounted = contribution_profit_per_customer(transactions(price=18.0, units_each=4.0), AS_OF)
    assert discounted.value < healthy.value


def test_cppc_falls_when_margin_extraction_drives_customers_away():
    """Failure mode two: raise price, hold margin per sale, lose the customers."""
    healthy = contribution_profit_per_customer(transactions(n_customers=100, price=30.0), AS_OF)
    extracted = contribution_profit_per_customer(transactions(n_customers=55, price=38.0), AS_OF)
    assert extracted.value > 0
    assert extracted.detail["n_customers"] < healthy.detail["n_customers"]
    # Total contribution collapsed even though per-sale margin rose.
    assert extracted.detail["contribution"] < healthy.detail["contribution"]


def test_gross_margin_misses_what_cppc_catches():
    """The argument for having a headline metric at all.

    Losing 45% of customers while raising price *improves* gross margin
    percentage — the surviving sales are richer. Only a per-customer metric
    notices that the business got smaller.
    """
    healthy = transactions(n_customers=100, price=30.0, cost=15.0)
    shrunk = transactions(n_customers=55, price=38.0, cost=15.0)

    assert gross_margin_pct(shrunk, 0.15).value > gross_margin_pct(healthy, 0.15).value
    assert (
        contribution_profit_per_customer(shrunk, AS_OF).detail["contribution"]
        < contribution_profit_per_customer(healthy, AS_OF).detail["contribution"]
    )


def test_cppc_window_excludes_older_transactions():
    """Ninety days because the repurchase cycle sits inside it."""
    recent = transactions(n_customers=50, days_ago=10)
    old = transactions(n_customers=50, days_ago=200)
    old["customer_id"] = old["customer_id"] + "-old"

    result = contribution_profit_per_customer(pd.concat([recent, old]), AS_OF, window_days=90)
    assert result.detail["n_customers"] == 50


def test_guest_checkouts_are_counted_in_profit_and_reported():
    """Their profit was earned and must count; they cannot be identified, and
    collapsing them into one customer would make the ratio meaningless."""
    result = contribution_profit_per_customer(transactions(n_customers=100, guests=40), AS_OF)
    assert result.detail["n_customers"] == 100
    assert result.detail["guest_share"] == pytest.approx(40 / 140)
    assert result.detail["contribution"] > 100 * 2.0 * 15.0


def test_an_empty_window_is_not_reported_as_zero():
    """Zero profit and no data are different events; only one is a business
    outcome."""
    result = contribution_profit_per_customer(transactions(days_ago=500), AS_OF)
    assert pd.isna(result.value)
    assert "ingestion" in result.action


def test_cppc_names_missing_columns():
    with pytest.raises(ValueError, match="unit_cost"):
        contribution_profit_per_customer(transactions().drop(columns=["unit_cost"]), AS_OF)


# ---------------------------------------------------------------------------
# Price adherence
# ---------------------------------------------------------------------------


def test_full_adherence_when_everything_is_published():
    result = price_adherence([10.0, 20.0, 30.0], [10.0, 20.0, 30.0])
    assert result.value == 1.0
    assert not result.breached


def test_adherence_breaches_when_recommendations_are_overridden():
    """A system producing numbers nobody uses has failed, and no profit metric
    can see it — the prices were never taken."""
    result = price_adherence([10.0] * 10, [10.0] * 5 + [12.0] * 5)
    assert result.value == pytest.approx(0.5)
    assert result.breached
    assert "not trusted" in result.action


def test_rounding_differences_are_not_counted_as_rejections():
    assert price_adherence([10.00], [10.001]).value == 1.0


def test_misaligned_series_are_refused():
    """A misaligned comparison would produce a plausible number from nonsense."""
    with pytest.raises(ValueError, match="must align"):
        price_adherence([1.0, 2.0], [1.0])


# ---------------------------------------------------------------------------
# Guardrail bind rate
# ---------------------------------------------------------------------------


def test_bind_rate_is_the_share_of_candidates_a_code_rejected():
    codes = [["PP-G002"], ["PP-G002"], [], ["PP-G005"]]
    rates = guardrail_bind_rate(codes)
    assert rates["PP-G002"].value == pytest.approx(0.5)
    assert rates["PP-G005"].value == pytest.approx(0.25)


def test_a_dominant_constraint_is_flagged_as_setting_the_price():
    """Not an error rate: it may be correct, but it has to be a decision
    somebody made rather than a fact nobody noticed."""
    result = guardrail_bind_rate([["PP-G002"]] * 9 + [[]])["PP-G002"]
    assert result.breached
    assert "setting the price" in result.action


def test_ladder_compliance_is_reported_but_not_alerted():
    """Binding on nearly every candidate is its expected state, and an alert
    that always fires is an alert nobody reads."""
    result = guardrail_bind_rate([["PP-G008"]] * 10)["PP-G008"]
    assert result.value == 1.0
    assert not result.breached
    assert "not alerted" in result.action


def test_bind_rate_requires_evaluations():
    with pytest.raises(ValueError, match="no candidate evaluations"):
        guardrail_bind_rate([])


# ---------------------------------------------------------------------------
# Recommendation churn
# ---------------------------------------------------------------------------


def test_churn_counts_direction_reversals():
    """Up then down. Each decision may have scored well; together they destroy
    trust, which is why this cannot be found one decision at a time."""
    prior = {"A": 10.0, "B": 10.0, "C": 10.0}
    previous = {"A": 12.0, "B": 12.0, "C": 12.0}
    current = {"A": 9.0, "B": 13.0, "C": 14.0}

    result = recommendation_churn(previous, current, prior)
    assert result.value == pytest.approx(1 / 3)
    assert result.detail["n_reversals"] == 1


def test_a_steady_walk_is_not_churn():
    """The markdown path moves the same way every period and must not alert."""
    prior = {"A": 30.0}
    previous = {"A": 28.0}
    current = {"A": 26.0}
    assert recommendation_churn(previous, current, prior).value == 0.0


def test_churn_breaches_above_the_threshold():
    prior = {f"S{i}": 10.0 for i in range(10)}
    previous = {f"S{i}": 12.0 for i in range(10)}
    current = {f"S{i}": (9.0 if i < 3 else 13.0) for i in range(10)}
    assert recommendation_churn(previous, current, prior).breached


def test_churn_requires_three_periods_of_overlap():
    with pytest.raises(ValueError, match="all three periods"):
        recommendation_churn({"A": 1.0}, {"B": 1.0}, {"C": 1.0})


# ---------------------------------------------------------------------------
# Degradation distribution
# ---------------------------------------------------------------------------


def test_rung_distribution_sums_to_one():
    results = degradation_distribution([1, 1, 2, 3, 5])
    total = sum(results[f"rung_{r}"].value for r in range(1, 6))
    assert total == pytest.approx(1.0)


def test_a_healthy_system_does_not_breach():
    assert not degradation_distribution([1] * 99 + [3])["degraded_share"].breached


def test_degraded_share_counts_rung_three_and_worse():
    """'We returned a price' and 'we returned a good price' are different
    numbers, and only the rung histogram separates them."""
    results = degradation_distribution([1] * 90 + [3] * 5 + [5] * 5)
    assert results["degraded_share"].value == pytest.approx(0.10)
    assert results["degraded_share"].breached


def test_degradation_requires_decisions():
    with pytest.raises(ValueError, match="no decisions"):
        degradation_distribution([])


def test_result_serialises_for_a_dashboard():
    record = price_adherence([1.0], [1.0]).as_dict()
    assert set(record) >= {"name", "value", "threshold", "breached", "action"}
