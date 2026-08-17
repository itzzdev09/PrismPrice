"""
Feature layer tests (L1).

The leakage tests are the reason this file exists. Everything else here is
ordinary correctness; leakage is the failure that produces an excellent backtest
and a model that does not work, and it is invisible unless tested directly.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from prismprice.data.synthetic import generate_panel
from prismprice.features.builder import (
    FeatureBuilder,
    FeatureVector,
    InsufficientHistoryError,
)


@pytest.fixture(scope="module")
def panel():
    return generate_panel(n_skus=4, n_days=400, n_customers=40, seed=13)


@pytest.fixture(scope="module")
def builder(panel):
    return FeatureBuilder(panel.daily)


@pytest.fixture(scope="module")
def as_of(panel):
    return panel.daily["date"].iloc[200].to_pydatetime()


# ---------------------------------------------------------------------------
# Leakage — the mandatory guarantee
# ---------------------------------------------------------------------------


def test_features_are_identical_when_the_future_is_appended(panel, as_of):
    """Build at T, append T..T+30, rebuild. Any difference is leakage.

    This is the test named as mandatory in docs/architecture.md §4.1.
    """
    cutoff = pd.Timestamp(as_of)
    past_only = panel.daily[panel.daily["date"] < cutoff]
    with_future = panel.daily[panel.daily["date"] < cutoff + pd.Timedelta(days=30)]
    assert len(with_future) > len(past_only), "the future rows must actually exist"

    for sku in panel.skus:
        before = FeatureBuilder(past_only).build(sku, as_of)
        after = FeatureBuilder(with_future).build(sku, as_of)
        assert before == after, f"feature values for {sku} changed when future data was added"


def test_features_ignore_the_decision_day_itself(panel, as_of):
    """`date < as_of`, never `<=`. Today's sales are not known this morning."""
    cutoff = pd.Timestamp(as_of)
    without_today = panel.daily[panel.daily["date"] != cutoff]
    full = FeatureBuilder(panel.daily).build(panel.skus[0], as_of)
    trimmed = FeatureBuilder(without_today).build(panel.skus[0], as_of)
    assert full == trimmed


def test_mutating_future_rows_cannot_change_past_features(panel, as_of):
    """A stronger form: corrupt the future outright and demand no reaction."""
    corrupted = panel.daily.copy()
    future = corrupted["date"] >= pd.Timestamp(as_of)
    corrupted.loc[future, "price"] = 9_999.0
    corrupted.loc[future, "units"] = 0.0

    baseline = FeatureBuilder(panel.daily).build(panel.skus[0], as_of)
    poisoned = FeatureBuilder(corrupted).build(panel.skus[0], as_of)
    assert baseline == poisoned


def test_training_frame_target_is_not_present_in_its_own_features(panel):
    """The one place features and outcomes are joined must not join them twice."""
    builder = FeatureBuilder(panel.daily)
    sku = panel.skus[0]
    frame = builder.training_frame(sku, min_history_days=120, stride=25)
    assert not frame.empty

    group = panel.daily[panel.daily["sku"] == sku].sort_values("date").reset_index(drop=True)
    for (_sku, as_of), row in frame.iterrows():
        realised = group.loc[group["date"] == pd.Timestamp(as_of), "units"]
        assert row["target_units"] == pytest.approx(float(realised.iloc[0]))
        # No feature may equal the target by construction.
        prior = group[group["date"] < pd.Timestamp(as_of)]["units"]
        assert row["units_7d"] == pytest.approx(float(prior.tail(7).mean()))


# ---------------------------------------------------------------------------
# Feature correctness
# ---------------------------------------------------------------------------


def test_trailing_windows_match_a_hand_computation(panel, as_of):
    sku = panel.skus[0]
    history = panel.daily[(panel.daily["sku"] == sku) & (panel.daily["date"] < pd.Timestamp(as_of))]
    vector = FeatureBuilder(panel.daily).build(sku, as_of)

    assert vector.units_7d == pytest.approx(float(history["units"].tail(7).mean()))
    assert vector.units_28d == pytest.approx(float(history["units"].tail(28).mean()))
    assert vector.units_91d == pytest.approx(float(history["units"].tail(91).mean()))
    assert vector.current_price == pytest.approx(float(history["price"].iloc[-1]))
    assert vector.reference_price_60d == pytest.approx(float(history["price"].tail(60).median()))


def test_omnibus_anchor_is_strictly_trailing(panel, as_of):
    """PP-G003's anchor must exclude the day being priced, or it validates itself."""
    sku = panel.skus[0]
    history = panel.daily[(panel.daily["sku"] == sku) & (panel.daily["date"] < pd.Timestamp(as_of))]
    vector = FeatureBuilder(panel.daily).build(sku, as_of)
    assert vector.min_price_last_30d == pytest.approx(float(history["price"].tail(30).min()))


def test_discount_depth_is_relative_to_the_reference_price(panel, as_of):
    vector = FeatureBuilder(panel.daily).build(panel.skus[0], as_of)
    expected = 1.0 - vector.current_price / vector.reference_price_60d
    assert vector.discount_depth == pytest.approx(expected)


def test_competitor_gap_sign_convention(panel, as_of):
    """Positive gap means we are more expensive; a sign flip here misprices."""
    vector = FeatureBuilder(panel.daily).build(panel.skus[0], as_of)
    assert vector.competitor_price is not None
    expected = vector.current_price / vector.competitor_price - 1.0
    assert vector.competitor_gap == pytest.approx(expected)


def test_zero_sales_streak_counts_back_from_the_present():
    dates = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
    daily = pd.DataFrame(
        {
            "sku": "S",
            "date": dates,
            "price": 10.0,
            "units": [5, 5, 5, 5, 5, 5, 5, 0, 0, 0],
        }
    )
    vector = FeatureBuilder(daily).build("S", datetime(2026, 1, 11, tzinfo=timezone.utc))
    assert vector.zero_sales_streak == 3


def test_demand_trend_is_seven_over_twenty_eight():
    dates = pd.date_range("2026-01-01", periods=40, freq="D", tz="UTC")
    units = [10.0] * 33 + [20.0] * 7
    daily = pd.DataFrame({"sku": "S", "date": dates, "price": 10.0, "units": units})
    vector = FeatureBuilder(daily).build("S", datetime(2026, 2, 10, tzinfo=timezone.utc))
    assert vector.demand_trend == pytest.approx(20.0 / vector.units_28d)
    assert vector.demand_trend > 1.0


def test_price_change_count_ignores_flat_days():
    dates = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
    prices = [10.0, 10.0, 12.0, 12.0, 12.0, 9.0, 9.0, 9.0, 9.0, 9.0]
    daily = pd.DataFrame({"sku": "S", "date": dates, "price": prices, "units": 1.0})
    vector = FeatureBuilder(daily).build("S", datetime(2026, 1, 11, tzinfo=timezone.utc))
    assert vector.price_changes_28d == 2


# ---------------------------------------------------------------------------
# Staleness and cold start
# ---------------------------------------------------------------------------


def test_stale_history_is_flagged_rather_than_silently_used(panel):
    """Rung 2 exists because the alternative is a confident number from old data."""
    sku = panel.skus[0]
    last = panel.daily["date"].max().to_pydatetime()
    vector = FeatureBuilder(panel.daily).build(sku, last + timedelta(days=10))
    assert vector.is_stale
    assert vector.days_since_last_observation >= 10


def test_fresh_history_is_not_flagged(panel, as_of):
    assert not FeatureBuilder(panel.daily).build(panel.skus[0], as_of).is_stale


def test_unknown_sku_raises_the_cold_start_signal(panel, as_of):
    with pytest.raises(InsufficientHistoryError) as excinfo:
        FeatureBuilder(panel.daily).build("SKU-DOES-NOT-EXIST", as_of)
    assert excinfo.value.sku == "SKU-DOES-NOT-EXIST"


def test_as_of_before_all_history_raises_rather_than_returning_zeros(panel):
    """Returning a zero-filled vector would price a product on nothing at all."""
    first = panel.daily["date"].min().to_pydatetime()
    with pytest.raises(InsufficientHistoryError):
        FeatureBuilder(panel.daily).build(panel.skus[0], first)


def test_build_many_skips_unusable_skus_without_failing(panel, as_of):
    vectors = FeatureBuilder(panel.daily).build_many(
        [(panel.skus[0], as_of), ("NOPE", as_of), (panel.skus[1], as_of)]
    )
    assert [v.sku for v in vectors] == [panel.skus[0], panel.skus[1]]


# ---------------------------------------------------------------------------
# Contracts and shape
# ---------------------------------------------------------------------------


def test_missing_required_column_is_rejected_at_construction():
    with pytest.raises(ValueError, match="missing required columns"):
        FeatureBuilder(pd.DataFrame({"sku": ["S"], "date": [pd.Timestamp("2026-01-01")]}))


def test_feature_vector_is_immutable(panel, as_of):
    vector = FeatureBuilder(panel.daily).build(panel.skus[0], as_of)
    with pytest.raises(ValidationError):
        vector.units_7d = 1.0


def test_to_series_drops_identity_and_keeps_only_numbers(panel, as_of):
    series = FeatureBuilder(panel.daily).build(panel.skus[0], as_of).to_series()
    assert "sku" not in series.index
    assert "as_of" not in series.index
    assert series.dtype == np.float64


def test_build_matrix_is_indexed_by_sku_and_time(panel, as_of):
    matrix = FeatureBuilder(panel.daily).build_matrix([(s, as_of) for s in panel.skus])
    assert list(matrix.index.names) == ["sku", "as_of"]
    assert len(matrix) == len(panel.skus)
    assert matrix.notna().all().all()


def test_naive_timestamps_are_accepted_and_treated_as_utc(panel):
    naive = panel.daily["date"].iloc[200].to_pydatetime().replace(tzinfo=None)
    aware = naive.replace(tzinfo=timezone.utc)
    builder = FeatureBuilder(panel.daily)
    assert builder.build(panel.skus[0], naive) == builder.build(panel.skus[0], aware)


def test_builder_can_read_from_the_uncensored_series(panel):
    """Trailing demand should reflect demand, not supply, once un-censoring ran."""
    as_of = panel.daily["date"].iloc[300].to_pydatetime()
    raw = FeatureBuilder(panel.daily, demand_column="units").build(panel.skus[0], as_of)
    latent = FeatureBuilder(panel.daily, demand_column="units_uncensored").build(
        panel.skus[0], as_of
    )
    assert latent.units_91d >= raw.units_91d


def test_feature_vector_field_set_is_stable():
    """Downstream models bind to these names; renaming one is a breaking change."""
    expected = {
        "sku",
        "as_of",
        "units_7d",
        "units_28d",
        "units_91d",
        "demand_trend",
        "demand_volatility",
        "zero_sales_streak",
        "current_price",
        "reference_price_60d",
        "discount_depth",
        "min_price_last_30d",
        "price_changes_28d",
        "week_of_year",
        "day_of_week",
        "is_weekend",
        "holiday_proximity",
        "inventory_on_hand",
        "inventory_cover_days",
        "inventory_shadow_price",
        "competitor_price",
        "competitor_gap",
        "competitor_age_days",
        "observations_used",
        "days_since_last_observation",
        "is_stale",
    }
    assert set(FeatureVector.model_fields) == expected
