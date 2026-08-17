"""
Demand un-censoring tests (L1).

The bar is not "the model fits" — it is "the estimate is closer to the latent
truth than doing nothing", measured only on the rows where censoring occurred.
The synthetic generator retains the truth, so this is directly scorable.
"""

import numpy as np
import pandas as pd
import pytest

from prismprice.data.synthetic import generate_panel
from prismprice.features.uncensoring import TobitUncensoring, score_uncensoring


@pytest.fixture(scope="module")
def censored_panel():
    # A deliberately under-stocked panel: the default cover target produces too
    # few stockouts to score a recovery against.
    return generate_panel(n_skus=5, n_days=380, n_customers=40, seed=5, cover_target=14.0)


@pytest.fixture(scope="module")
def fitted(censored_panel):
    return TobitUncensoring().fit(censored_panel.daily)


# ---------------------------------------------------------------------------
# Recovery quality
# ---------------------------------------------------------------------------


def test_the_panel_actually_contains_censoring(censored_panel):
    assert censored_panel.daily["stockout"].mean() > 0.05


def test_uncensoring_beats_doing_nothing_by_a_wide_margin(censored_panel, fitted):
    result = fitted.transform(censored_panel.daily)
    score = score_uncensoring(result)
    assert score.censored_rows > 50
    assert score.estimated_wape < score.naive_wape
    assert score.improvement > 0.4, (
        f"only removed {score.improvement:.1%} of the censoring error "
        f"(naive {score.naive_wape:.3f} -> {score.estimated_wape:.3f})"
    )


def test_estimates_are_never_below_what_was_actually_sold(censored_panel, fitted):
    """Demand is at least the units served; an estimate below that is incoherent."""
    result = fitted.transform(censored_panel.daily)
    assert (result["units_uncensored_est"] >= result["units"] - 1e-6).all()


def test_uncensored_rows_are_passed_through_untouched(censored_panel, fitted):
    """The model fills gaps; it does not overwrite observations."""
    result = fitted.transform(censored_panel.daily)
    open_rows = result[~result["stockout"]]
    np.testing.assert_allclose(
        open_rows["units_uncensored_est"].to_numpy(),
        np.maximum(open_rows["units"].to_numpy(), 1e-3),
        rtol=1e-9,
    )


def test_estimates_move_toward_latent_truth_on_censored_rows(censored_panel, fitted):
    result = fitted.transform(censored_panel.daily)
    censored = result[result["stockout"]]
    naive_gap = (censored["units_uncensored"] - censored["units"]).abs().mean()
    model_gap = (censored["units_uncensored"] - censored["units_uncensored_est"]).abs().mean()
    assert model_gap < naive_gap


def test_the_model_converges_and_reports_a_sane_sigma(fitted):
    assert fitted.converged
    assert fitted.sigma is not None
    assert 0.0 < fitted.sigma < 2.0


# ---------------------------------------------------------------------------
# Behaviour at the edges
# ---------------------------------------------------------------------------


def test_a_panel_with_no_censoring_is_left_alone():
    panel = generate_panel(n_skus=3, n_days=300, n_customers=20, seed=8, cover_target=60.0)
    daily = panel.daily
    if daily["stockout"].any():
        pytest.skip("panel unexpectedly contains censoring")
    result = TobitUncensoring().fit_transform(daily)
    np.testing.assert_allclose(
        result["units_uncensored_est"].to_numpy(),
        np.maximum(daily["units"].to_numpy(), 1e-3),
        rtol=1e-9,
    )


def test_fully_censored_panel_is_rejected_rather_than_fitted():
    """With no open rows there is nothing to identify the demand equation from."""
    panel = generate_panel(n_skus=2, n_days=120, n_customers=10, seed=2)
    daily = panel.daily.copy()
    daily["stockout"] = True
    with pytest.raises(ValueError, match="no uncensored variation"):
        TobitUncensoring().fit(daily)


def test_prediction_before_fitting_raises():
    with pytest.raises(RuntimeError, match="must be fitted"):
        TobitUncensoring().expected_latent_log_demand(pd.DataFrame())


def test_unseen_sku_at_transform_time_does_not_shift_the_design(censored_panel, fitted):
    """A new SKU contributes no fixed effect instead of misaligning every column."""
    extra = censored_panel.daily.head(20).copy()
    extra["sku"] = "SKU-BRAND-NEW"
    result = fitted.transform(extra)
    assert np.isfinite(result["units_uncensored_est"]).all()
    assert (result["units_uncensored_est"] > 0).all()


def test_scoring_an_uncensored_frame_returns_a_neutral_score():
    frame = pd.DataFrame(
        {
            "stockout": [False, False],
            "units": [3.0, 4.0],
            "units_uncensored": [3.0, 4.0],
            "units_uncensored_est": [3.0, 4.0],
        }
    )
    score = score_uncensoring(frame)
    assert score.censored_rows == 0
    assert score.improvement == 0.0


def test_score_reports_a_negative_improvement_when_the_estimate_is_worse():
    """The metric must be able to say 'this made things worse', or it is decoration."""
    frame = pd.DataFrame(
        {
            "stockout": [True],
            "units": [8.0],
            "units_uncensored": [10.0],
            "units_uncensored_est": [50.0],
        }
    )
    assert score_uncensoring(frame).improvement < 0


def test_fitting_is_deterministic(censored_panel):
    a = TobitUncensoring().fit(censored_panel.daily)
    b = TobitUncensoring().fit(censored_panel.daily)
    np.testing.assert_allclose(a.coefficients, b.coefficients)
    assert a.sigma == pytest.approx(b.sigma)
