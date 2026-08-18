"""
Retention and Delta-CLV tests.

The phase-4 gate is recovery of the known repurchase sensitivity ``theta`` from
the synthetic generator. That is a stronger claim than the roadmap's "cohort
curve MAE below threshold": a model can track cohort curves well while getting
the *price* coefficient wrong, and the price coefficient is the only part of
this layer the decision engine consumes.

A note on reading these numbers. On a single panel the estimate lands anywhere
in a band roughly 0.12 wide, so any single-seed assertion is a coin toss dressed
as a test. The gates below therefore average over independent panels, and
:func:`test_theta_is_unbiased_across_panels` is the one that actually
establishes the estimator is right.
"""

from __future__ import annotations

import numpy as np
import pytest

from prismprice.compute import GPUInfo, GPUUnavailableError
from prismprice.data.synthetic import generate_panel
from prismprice.estimation.retention import (
    CLVEstimate,
    CoxSurvivalModel,
    clv_sensitivity,
    price_shock,
)

SEEDS = (11, 23, 37)
TRUE_THETA = -1.2


def _survival_frame(seed: int, with_frailty: bool = False):
    panel = generate_panel(n_skus=5, n_days=420, n_customers=500, seed=seed)
    frame = panel.purchases.copy()
    if with_frailty:
        frame = frame.merge(panel.customers[["customer_id", "frailty"]], on="customer_id")
    frame["price_shock"] = frame["price_ratio"] - 1.0
    return panel, frame


@pytest.fixture(scope="module")
def fitted_models():
    out = []
    for seed in SEEDS:
        panel, frame = _survival_frame(seed)
        out.append((panel, CoxSurvivalModel(max_epochs=250).fit(frame)))
    return out


@pytest.fixture(scope="module")
def model(fitted_models):
    return fitted_models[0][1]


# ---------------------------------------------------------------------------
# The phase-4 gate
# ---------------------------------------------------------------------------


def test_theta_is_unbiased_across_panels(fitted_models):
    """Phase-4 gate: recovers the generator's known price sensitivity.

    Averaged over independent panels, because the per-panel spread is wide
    enough that a single draw proves nothing either way.
    """
    estimates = [m.fit_result.coefficient_for("price_shock") for _, m in fitted_models]
    bias = float(np.mean(estimates)) - TRUE_THETA
    assert abs(bias) < 0.20, f"mean theta {np.mean(estimates):+.4f} vs true {TRUE_THETA}"


def test_theta_has_the_right_sign_on_every_panel(fitted_models):
    """Paying above reference must lengthen the gap to the next purchase.

    A positive theta would tell the objective that raising prices improves
    retention, and lambda would then reward exactly the wrong move.
    """
    for panel, fitted in fitted_models:
        theta = fitted.fit_result.coefficient_for("price_shock")
        assert theta < 0, f"seed {panel.truth.seed}: theta {theta:+.4f} is not negative"


def test_interval_covers_truth_on_most_panels(fitted_models):
    covered = sum(
        1
        for _, m in fitted_models
        if m.fit_result.confidence_interval("price_shock")[0]
        <= TRUE_THETA
        <= m.fit_result.confidence_interval("price_shock")[1]
    )
    assert covered >= len(fitted_models) - 1, f"only {covered}/{len(fitted_models)} intervals cover"


def test_recovers_both_coefficients_when_frailty_is_observed():
    """With the unobserved heterogeneity supplied, both coefficients land.

    This is the test that separates "the estimator is wrong" from "the data
    withholds something": the generator's frailty has a true coefficient of 1.0,
    and recovering it alongside theta shows the partial likelihood, the tie
    handling and the information matrix are all correct.
    """
    _, frame = _survival_frame(11, with_frailty=True)
    fitted = CoxSurvivalModel(feature_columns=("price_shock", "frailty"), max_epochs=250).fit(frame)
    assert fitted.fit_result.coefficient_for("frailty") == pytest.approx(1.0, abs=0.3)
    assert fitted.fit_result.coefficient_for("price_shock") == pytest.approx(TRUE_THETA, abs=0.35)


# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------


def test_price_shock_is_zero_at_the_reference():
    assert price_shock(10.0, 10.0) == pytest.approx(0.0)
    assert price_shock(12.0, 10.0) == pytest.approx(0.2)
    assert price_shock(8.0, 10.0) == pytest.approx(-0.2)


def test_censored_rows_are_kept(fitted_models):
    """A customer who has not returned yet still survived this long.

    Dropping censored rows biases every curve toward the customers who churned
    fastest, which would make retention look worse than it is and inflate the
    apparent value of a discount.
    """
    _, fitted = fitted_models[0]
    assert fitted.fit_result.n_censored > 0
    assert fitted.fit_result.n_events > fitted.fit_result.n_censored


def test_ties_are_grouped_into_one_risk_set():
    """Gaps are whole days here, so ties are the common case, not an edge case.

    Each tied row must see the whole tied group in its risk set; missing them
    shrinks every denominator slightly and biases the coefficient.
    """
    durations = np.array([9.0, 7.0, 7.0, 7.0, 3.0])
    last = CoxSurvivalModel._last_index_of_tied_group(durations)
    assert last.tolist() == [0, 3, 3, 3, 4]


def test_all_events_at_one_time_is_a_single_risk_set():
    last = CoxSurvivalModel._last_index_of_tied_group(np.array([5.0, 5.0, 5.0]))
    assert last.tolist() == [2, 2, 2]


def test_a_frame_with_no_events_is_refused():
    """The partial likelihood is defined by the order of events; with none, the
    fit is unidentified and returning a number would be fabrication."""
    _, frame = _survival_frame(11)
    frame = frame.assign(repurchased=False)
    with pytest.raises(ValueError, match="no observed repurchases"):
        CoxSurvivalModel(max_epochs=20).fit(frame)


def test_missing_columns_are_named():
    _, frame = _survival_frame(11)
    with pytest.raises(ValueError, match="price_shock"):
        CoxSurvivalModel(max_epochs=20).fit(frame.drop(columns=["price_shock"]))


def test_deep_risk_score_refuses_to_report_a_coefficient():
    """An MLP has no single log-hazard ratio, and inventing one is how a
    decision record ends up quoting a number that does not exist."""
    _, frame = _survival_frame(11)
    fitted = CoxSurvivalModel(hidden_sizes=(8,), max_epochs=40).fit(frame)
    assert not fitted.fit_result.is_linear
    with pytest.raises(LookupError, match="no coefficient"):
        fitted.fit_result.coefficient_for("price_shock")


def test_clustered_and_naive_errors_are_both_available():
    """Recurrent events break the independence the partial likelihood assumes.

    On this generator the residual within-customer correlation is small once the
    price shock is conditioned on, so the two agree closely — which is the
    correct result, not a broken correction. The estimator is here for data
    where it is not small.
    """
    _, frame = _survival_frame(11)
    clustered = CoxSurvivalModel(max_epochs=250).fit(frame).fit_result.standard_errors[0]
    naive = (
        CoxSurvivalModel(cluster_column=None, max_epochs=250)
        .fit(frame)
        .fit_result.standard_errors[0]
    )
    assert np.isfinite(clustered) and np.isfinite(naive)
    assert 0.5 < clustered / naive < 2.0


# ---------------------------------------------------------------------------
# Survival curves and Delta-CLV
# ---------------------------------------------------------------------------


def test_survival_is_a_decreasing_probability(model):
    days = np.array([10.0, 30.0, 60.0, 120.0, 240.0])
    curve = model.survival(np.array([[0.0]]), days)[0]
    assert np.all((curve >= 0.0) & (curve <= 1.0))
    assert np.all(np.diff(curve) <= 1e-12), "survival must be non-increasing in time"


def test_a_price_rise_delays_the_next_purchase(model):
    """The whole point of the layer, and the direction is a trap.

    `survival` here is the survival function of the *inter-purchase gap*, so a
    price rise pushes it UP — the customer takes longer to come back. High
    survival is bad retention. The retention quantity that reads the intuitive
    way is `repurchase_probability`, and both directions are asserted so the
    pair can never drift into agreeing with each other.
    """
    days = np.array([30.0, 90.0, 180.0])
    above = model.survival(np.array([[0.20]]), days)[0]
    at = model.survival(np.array([[0.0]]), days)[0]
    below = model.survival(np.array([[-0.20]]), days)[0]

    assert np.all(above > at), "paying above reference must lengthen the gap"
    assert np.all(below < at), "paying below reference must shorten the gap"

    returned_above = model.repurchase_probability(np.array([[0.20]]), days)[0]
    returned_at = model.repurchase_probability(np.array([[0.0]]), days)[0]
    assert np.all(returned_above < returned_at), "a price rise must not improve retention"
    assert np.all(returned_above + above == pytest.approx(1.0))


def test_delta_clv_is_zero_at_the_reference_price(model):
    estimate = model.delta_clv(price=20.0, reference_price=20.0, unit_margin=5.0)
    assert estimate.delta_clv == pytest.approx(0.0, abs=1e-9)
    assert estimate.price_shock == pytest.approx(0.0)


def test_delta_clv_is_negative_for_a_price_rise(model):
    """The counterweight the objective needs: today's extra margin is not free."""
    assert model.delta_clv(price=24.0, reference_price=20.0, unit_margin=5.0).delta_clv < 0


def test_delta_clv_is_positive_for_a_price_cut(model):
    assert model.delta_clv(price=17.0, reference_price=20.0, unit_margin=5.0).delta_clv > 0


def test_delta_clv_grows_with_the_size_of_the_shock(model):
    small = model.delta_clv(price=21.0, reference_price=20.0, unit_margin=5.0).delta_clv
    large = model.delta_clv(price=26.0, reference_price=20.0, unit_margin=5.0).delta_clv
    assert large < small < 0


def test_delta_clv_scales_with_margin(model):
    single = model.delta_clv(price=24.0, reference_price=20.0, unit_margin=1.0).delta_clv
    double = model.delta_clv(price=24.0, reference_price=20.0, unit_margin=2.0).delta_clv
    assert double == pytest.approx(2.0 * single, rel=1e-9)


def test_a_longer_horizon_cannot_shrink_the_magnitude(model):
    short = abs(model.delta_clv(24.0, 20.0, 5.0, horizon_periods=4).delta_clv)
    long = abs(model.delta_clv(24.0, 20.0, 5.0, horizon_periods=24).delta_clv)
    assert long >= short


def test_a_higher_discount_rate_shrinks_the_magnitude(model):
    patient = abs(model.delta_clv(24.0, 20.0, 5.0, discount_rate=0.05).delta_clv)
    impatient = abs(model.delta_clv(24.0, 20.0, 5.0, discount_rate=0.30).delta_clv)
    assert impatient < patient


def test_delta_clv_rejects_impossible_inputs(model):
    with pytest.raises(ValueError, match="reference_price must be > 0"):
        model.delta_clv(price=10.0, reference_price=0.0, unit_margin=1.0)
    with pytest.raises(ValueError, match="horizon_periods must be >= 1"):
        model.delta_clv(price=10.0, reference_price=10.0, unit_margin=1.0, horizon_periods=0)


def test_unfitted_model_refuses_to_predict():
    with pytest.raises(ValueError, match="fit\\(\\) has not been called"):
        CoxSurvivalModel().delta_clv(10.0, 10.0, 1.0)


def test_sensitivity_grid_spans_horizon_and_discount(model):
    """Delta-CLV's two most arbitrary inputs are configuration, so a single
    number would imply a precision the model does not have."""
    grid = clv_sensitivity(model, price=24.0, reference_price=20.0, unit_margin=5.0)
    assert set(grid.columns) == {"horizon_periods", "discount_rate", "delta_clv"}
    assert len(grid) == 9
    assert grid["delta_clv"].nunique() > 1, "a flat grid would mean the dials do nothing"


def test_estimate_serialises_for_the_decision_record(model):
    record = model.delta_clv(24.0, 20.0, 5.0).as_dict()
    assert set(record) >= {"price", "reference_price", "delta_clv", "price_shock"}
    assert isinstance(record["delta_clv"], float)


def test_fit_serialises_with_its_device(model):
    record = model.fit_result.as_dict()
    assert record["device"] in {"cpu", "cuda", "cuda:0"}
    assert record["is_linear"] is True


# ---------------------------------------------------------------------------
# Compute contract
# ---------------------------------------------------------------------------


def test_retention_requires_a_gpu(monkeypatch):
    """A torch-backed component, so the hard GPU rule still applies here."""
    from prismprice import compute

    monkeypatch.delenv("PRISMPRICE_ALLOW_CPU", raising=False)
    monkeypatch.setattr(compute, "gpu_report", lambda: GPUInfo(available=False, reason="sim"))
    _, frame = _survival_frame(11)
    with pytest.raises(GPUUnavailableError, match=r"estimation\.retention"):
        CoxSurvivalModel(max_epochs=5).fit(frame)


def test_clv_estimate_is_frozen():
    estimate = CLVEstimate(
        price=1.0,
        reference_price=1.0,
        delta_clv=0.0,
        unit_margin=1.0,
        horizon_periods=1,
        discount_rate=0.1,
        period_days=30,
        survival_at_horizon_price=1.0,
        survival_at_horizon_reference=1.0,
    )
    with pytest.raises((AttributeError, TypeError)):
        estimate.delta_clv = 1.0  # type: ignore[misc]
