"""
Quantile demand model tests (L2).

The gate from the roadmap is calibration, not accuracy: "are p10/p90 honest?"
An accurate median with a dishonest interval is worse than the reverse here,
because L3 samples the interval and the CVaR penalty is computed from it.

These tests set ``PRISMPRICE_ALLOW_CPU`` because neither CI runners nor the
current dev machine has a CUDA-enabled LightGBM/torch build. The GPU contract
itself is tested separately, without the escape hatch.
"""

import os
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from prismprice import compute
from prismprice.compute import GPUInfo, GPUUnavailableError
from prismprice.data.synthetic import generate_panel
from prismprice.estimation.demand import (
    DemandForecast,
    QuantileDemandModel,
    calibration_report,
    rolling_origin_backtest,
)
from prismprice.features.builder import FeatureBuilder


@pytest.fixture(autouse=True)
def _allow_cpu(monkeypatch):
    monkeypatch.setenv("PRISMPRICE_ALLOW_CPU", "1")


@pytest.fixture(scope="module")
def frame():
    panel = generate_panel(n_skus=5, n_days=520, n_customers=20, seed=4)
    builder = FeatureBuilder(panel.daily)
    return pd.concat([builder.training_frame(sku, min_history_days=91) for sku in panel.skus])


@pytest.fixture(scope="module")
def split(frame):
    as_of = frame.index.get_level_values("as_of")
    boundary = as_of.unique().sort_values()[int(0.7 * as_of.nunique())]
    return frame[as_of <= boundary], frame[as_of > boundary]


@pytest.fixture(scope="module")
def fitted(split):
    # Set directly rather than via monkeypatch: this fixture is module-scoped and
    # monkeypatch is function-scoped, so the autouse fixture above is not active
    # while it runs. Restored on teardown.
    previous = os.environ.get("PRISMPRICE_ALLOW_CPU")
    os.environ["PRISMPRICE_ALLOW_CPU"] = "1"
    try:
        yield QuantileDemandModel().fit(split[0])
    finally:
        if previous is None:
            os.environ.pop("PRISMPRICE_ALLOW_CPU", None)
        else:
            os.environ["PRISMPRICE_ALLOW_CPU"] = previous


# ---------------------------------------------------------------------------
# Calibration — the roadmap gate
# ---------------------------------------------------------------------------


def test_backtest_interval_coverage_is_within_three_points(frame):
    """The specified gate: rolling-origin p10/p90 coverage within +/-3pp."""
    pooled, folds = rolling_origin_backtest(frame, n_folds=3)
    assert len(folds) >= 2
    assert pooled.interval_calibrated(3.0), pooled.summary()


def test_backtest_tail_placement_is_within_five_points(frame):
    """Looser, and reported separately — see CalibrationReport.tails_calibrated."""
    pooled, _ = rolling_origin_backtest(frame, n_folds=3)
    assert pooled.tails_calibrated(5.0), pooled.summary()


def test_conformalisation_materially_improves_coverage(split):
    """Raw booster quantiles are too narrow; this is why CQR is in the pipeline."""
    train, test = split
    raw = QuantileDemandModel(conformal_fraction=0.0).fit(train)
    conformal = QuantileDemandModel(conformal_fraction=0.25).fit(train)

    raw_report = calibration_report(test["target_units"], raw.predict_quantiles(test))
    cqr_report = calibration_report(test["target_units"], conformal.predict_quantiles(test))

    assert raw_report.coverage_80 < 0.75, "the raw model was supposed to be under-covered"
    assert cqr_report.coverage_80 > raw_report.coverage_80


def test_conformal_radii_are_recorded_for_audit(fitted):
    assert fitted.conformal_lower != 0.0 or fitted.conformal_upper != 0.0


def test_conformal_is_skipped_when_there_is_too_little_history(split):
    """Below the split threshold the model must not calibrate on a handful of rows."""
    train, _ = split
    tiny = train.head(60)
    model = QuantileDemandModel().fit(tiny)
    assert model.conformal_lower == 0.0
    assert model.conformal_upper == 0.0


# ---------------------------------------------------------------------------
# Distributional sanity
# ---------------------------------------------------------------------------


def test_quantiles_never_cross(fitted, split):
    """Separate boosters per quantile are uncoupled and can cross on a row."""
    predicted = fitted.predict_quantiles(split[1])
    assert (predicted["p10"] <= predicted["p50"] + 1e-9).all()
    assert (predicted["p50"] <= predicted["p90"] + 1e-9).all()


def test_predictions_are_never_negative(fitted, split):
    predicted = fitted.predict_quantiles(split[1])
    assert (predicted.to_numpy() >= 0).all()


def test_median_beats_a_seasonal_naive_baseline(fitted, split):
    """Calibration is the gate, but a model worse than 'yesterday' is not useful."""
    _, test = split
    predicted = fitted.predict_quantiles(test)
    truth = test["target_units"].to_numpy(dtype=float)

    model_wape = np.sum(np.abs(truth - predicted["p50"])) / np.sum(np.abs(truth))
    baseline_wape = np.sum(np.abs(truth - test["units_7d"])) / np.sum(np.abs(truth))
    assert model_wape < baseline_wape, f"model {model_wape:.3f} vs 7d-mean {baseline_wape:.3f}"


def test_interval_widens_where_demand_is_more_uncertain(fitted, split):
    predicted = fitted.predict_quantiles(split[1])
    spread = predicted["p90"] - predicted["p10"]
    assert (spread >= 0).all()
    assert spread.std() > 0, "a constant-width interval is not carrying uncertainty"


# ---------------------------------------------------------------------------
# Monotonicity in price
# ---------------------------------------------------------------------------


def test_ladder_demand_is_non_increasing_in_price(fitted, split):
    """An optimiser handed an upward-sloping demand curve will walk up it."""
    context = split[1].iloc[0].drop(["target_units"])
    prices = np.linspace(
        context["reference_price_60d"] * 0.6, context["reference_price_60d"] * 1.6, 12
    )
    medians = [f.p50 for f in fitted.forecast_at_prices(context, prices)]
    assert all(a >= b - 1e-9 for a, b in pairwise(medians))


def test_every_quantile_is_projected_not_just_the_median(fitted, split):
    context = split[1].iloc[0].drop(["target_units"])
    prices = np.linspace(
        context["reference_price_60d"] * 0.5, context["reference_price_60d"] * 1.8, 15
    )
    forecasts = fitted.forecast_at_prices(context, prices)
    for attribute in ("p10", "p50", "p90"):
        series = [getattr(f, attribute) for f in forecasts]
        assert all(a >= b - 1e-9 for a, b in pairwise(series)), attribute


def test_projection_preserves_the_input_price_order(fitted, split):
    """Prices arrive unsorted from a ladder; the mapping back must be exact."""
    context = split[1].iloc[0].drop(["target_units"])
    prices = [40.0, 20.0, 60.0, 30.0]
    forecasts = fitted.forecast_at_prices(context, prices)
    assert [f.price for f in forecasts] == prices
    by_price = {f.price: f.p50 for f in forecasts}
    ordered = [by_price[p] for p in sorted(prices)]
    assert all(a >= b - 1e-9 for a, b in pairwise(ordered))


def test_monotonicity_can_be_switched_off_for_diagnosis(fitted, split):
    context = split[1].iloc[0].drop(["target_units"])
    prices = np.linspace(15.0, 80.0, 10)
    raw = fitted.forecast_at_prices(context, prices, enforce_monotonicity=False)
    projected = fitted.forecast_at_prices(context, prices, enforce_monotonicity=True)
    assert all(p.p50 <= r.p50 + 1e-9 for p, r in zip(projected, raw, strict=True))


def test_empty_ladder_returns_no_forecasts(fitted, split):
    assert fitted.forecast_at_prices(split[1].iloc[0].drop(["target_units"]), []) == []


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------


def test_design_derives_the_relative_price_terms(frame):
    design = QuantileDemandModel.design(frame.head(20))
    assert {"log_candidate_price", "price_vs_reference", "price_vs_competitor"} <= set(
        design.columns
    )
    expected = frame["target_price"].head(20) / frame["reference_price_60d"].head(20)
    np.testing.assert_allclose(design["price_vs_reference"], expected, rtol=1e-9)


def test_design_excludes_the_target_and_pipeline_provenance(frame):
    """observations_used describes our data pipeline, not the product."""
    design = QuantileDemandModel.design(frame.head(20))
    for column in (
        "target_units",
        "target_price",
        "observations_used",
        "days_since_last_observation",
    ):
        assert column not in design.columns


def test_design_requires_a_candidate_price():
    with pytest.raises(ValueError, match="target_price"):
        QuantileDemandModel.design(pd.DataFrame({"reference_price_60d": [10.0]}))


# ---------------------------------------------------------------------------
# Errors and contracts
# ---------------------------------------------------------------------------


def test_prediction_before_fitting_raises():
    with pytest.raises(RuntimeError, match="must be fitted"):
        QuantileDemandModel().predict_quantiles(pd.DataFrame())


def test_missing_target_column_is_rejected(frame):
    with pytest.raises(ValueError, match="target_units"):
        QuantileDemandModel().fit(frame.drop(columns=["target_units"]))


def test_tiny_training_frame_is_rejected(frame):
    with pytest.raises(ValueError, match="training rows"):
        QuantileDemandModel().fit(frame.head(5))


def test_backtest_needs_a_time_index(frame):
    with pytest.raises(ValueError, match="as_of"):
        rolling_origin_backtest(frame.reset_index(drop=True))


def test_fitting_is_deterministic(split):
    train, test = split
    first = QuantileDemandModel().fit(train).predict_quantiles(test)
    second = QuantileDemandModel().fit(train).predict_quantiles(test)
    pd.testing.assert_frame_equal(first, second)


def test_forecast_exposes_its_spread():
    forecast = DemandForecast(price=10.0, p10=5.0, p50=8.0, p90=14.0)
    assert forecast.spread == pytest.approx(9.0)
    assert forecast.as_dict()["p50"] == 8.0


# ---------------------------------------------------------------------------
# GPU contract
# ---------------------------------------------------------------------------


def test_training_refuses_to_run_on_cpu_without_the_escape_hatch(monkeypatch, split):
    """LightGBM warns and trains on CPU when its GPU build is missing; we don't."""
    monkeypatch.delenv("PRISMPRICE_ALLOW_CPU", raising=False)
    monkeypatch.setattr(compute, "gpu_report", lambda: GPUInfo(available=False, reason="simulated"))
    with pytest.raises(GPUUnavailableError, match=r"estimation\.demand"):
        QuantileDemandModel().fit(split[0])
