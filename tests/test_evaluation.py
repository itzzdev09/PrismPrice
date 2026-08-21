"""
Model evaluation tests.

The theme is that a metric can be arithmetically correct and still answer the
wrong question. Each test below pins a case where the *obvious* metric would
give a flattering number: accuracy on a rare alarm, MAPE on a long tail, RMSE on
a quantile model, squared error on censored durations. Those are the failures
worth a test, because the arithmetic never complains.
"""

from __future__ import annotations

import numpy as np
import pytest

from prismprice.observability.evaluation import (
    binary_classification_report,
    concordance_index,
    elasticity_health,
    pinball_loss,
    quantile_coverage,
    score_demand,
    wape,
)

# ---------------------------------------------------------------------------
# Regression and quantile metrics
# ---------------------------------------------------------------------------


def test_wape_is_zero_for_a_perfect_forecast():
    assert wape([10.0, 20.0, 30.0], [10.0, 20.0, 30.0]) == 0.0


def test_wape_weights_by_volume_where_mape_would_not():
    """MAPE divides each error by its own actual, so a one-unit miss on a
    one-unit day counts as much as a hundred-unit miss on a thousand-unit day.
    On a long-tail catalogue that is most of the file."""
    actual = [1.0, 1000.0]
    predicted = [2.0, 1000.0]  # 100% wrong on the tiny SKU, exact on the big one

    assert wape(actual, predicted) == pytest.approx(1.0 / 1001.0)
    mape = float(np.mean(np.abs(np.array(actual) - np.array(predicted)) / np.array(actual)))
    assert mape == pytest.approx(0.5), "MAPE would call this forecast 50% wrong"


def test_wape_rejects_misaligned_series():
    with pytest.raises(ValueError, match="must align"):
        wape([1.0, 2.0], [1.0])


def test_pinball_loss_penalises_the_two_tails_asymmetrically():
    """The whole point of a quantile loss, and what RMSE cannot express."""
    over = pinball_loss([10.0], [12.0], quantile=0.10)
    under = pinball_loss([10.0], [8.0], quantile=0.10)
    assert over > under, "a p10 forecast should be punished harder for overshooting"


def test_pinball_loss_at_the_median_is_half_the_absolute_error():
    assert pinball_loss([10.0], [12.0], quantile=0.50) == pytest.approx(1.0)


def test_pinball_rejects_an_impossible_quantile():
    with pytest.raises(ValueError, match="quantile must be in"):
        pinball_loss([1.0], [1.0], quantile=1.5)


def test_coverage_separates_width_from_placement():
    """An interval can be the right size and in the wrong place, and the CVaR
    term built on it would then measure the wrong tail."""
    actual = [1.0] * 10 + [5.0] * 80 + [9.0] * 10
    lower = [2.0] * 100
    upper = [8.0] * 100

    result = quantile_coverage(actual, lower, upper)
    assert result["coverage"] == pytest.approx(0.80)
    assert result["placement_skew"] == pytest.approx(0.0)


def test_a_shifted_interval_is_caught_by_placement_even_at_the_right_width():
    actual = [1.0] * 20 + [5.0] * 80
    lower = [2.0] * 100
    upper = [99.0] * 100

    result = quantile_coverage(actual, lower, upper)
    assert result["coverage"] == pytest.approx(0.80), "width looks correct"
    assert result["placement_skew"] == pytest.approx(0.20), "but every miss is on one side"


def test_demand_scorecard_flags_an_uncalibrated_interval():
    rng = np.random.default_rng(0)
    actual = rng.normal(100.0, 20.0, 2000)
    # A *fixed* narrow band, not one centred on the actual — an interval built
    # around the outcome it is meant to predict covers 100% and measures nothing.
    lower = np.full_like(actual, 98.0)
    upper = np.full_like(actual, 102.0)
    score = score_demand(actual, lower, np.full_like(actual, 100.0), upper)
    assert not score.calibrated
    assert score.coverage < 0.5


def test_a_well_calibrated_interval_passes():
    rng = np.random.default_rng(0)
    actual = rng.normal(100.0, 20.0, 20000)
    lower = np.full_like(actual, 100.0 - 1.2816 * 20.0)
    upper = np.full_like(actual, 100.0 + 1.2816 * 20.0)
    score = score_demand(actual, lower, np.full_like(actual, 100.0), upper)
    assert score.calibrated


def test_scoring_requires_observations():
    with pytest.raises(ValueError, match="no observations"):
        score_demand([], [], [], [])


# ---------------------------------------------------------------------------
# Survival: why not squared error
# ---------------------------------------------------------------------------


def test_concordance_is_one_for_perfect_risk_ordering():
    """Higher risk must mean a shorter duration."""
    assert concordance_index([1, 2, 3, 4, 5], [True] * 5, [5, 4, 3, 2, 1]) == 1.0


def test_concordance_is_zero_for_exactly_wrong_ordering():
    assert concordance_index([1, 2, 3, 4, 5], [True] * 5, [1, 2, 3, 4, 5]) == 0.0


def test_concordance_is_half_for_a_constant_risk_score():
    """No information means coin-flipping, not zero."""
    assert concordance_index([1, 2, 3, 4], [True] * 4, [7.0] * 4) == pytest.approx(0.5)


def test_censored_observations_are_not_treated_as_failures():
    """A censored row's true duration is unknown and only bounded, so it cannot
    be known to have failed first. Squared error against its observed duration
    would be measuring a number that does not exist.
    """
    durations = [1.0, 2.0, 3.0]
    risks = [3.0, 2.0, 1.0]
    all_events = concordance_index(durations, [True, True, True], risks)
    first_censored = concordance_index(durations, [False, True, True], risks)

    assert all_events == 1.0
    assert first_censored == 1.0, "dropping the censored row's comparisons, not its information"


def test_concordance_rejects_misaligned_input():
    with pytest.raises(ValueError, match="must align"):
        concordance_index([1.0, 2.0], [True], [1.0, 2.0])


# ---------------------------------------------------------------------------
# The confusion matrix, where it belongs
# ---------------------------------------------------------------------------


def test_a_breaker_that_never_fires_scores_high_accuracy_and_no_recall():
    """The reason accuracy is not the headline for a rare alarm."""
    truth = [True] * 10 + [False] * 90
    never = [False] * 100

    report = binary_classification_report(truth, never)
    assert report.accuracy == pytest.approx(0.90)
    assert report.recall == 0.0
    assert report.accuracy_is_misleading, "90% accuracy on a 10% event must be flagged"


def test_a_breaker_that_always_fires_has_perfect_recall_and_no_precision():
    truth = [True] * 10 + [False] * 90
    report = binary_classification_report(truth, [True] * 100)
    assert report.recall == 1.0
    assert report.precision == pytest.approx(0.10)


def test_balanced_accuracy_sees_through_the_imbalance():
    """Both degenerate classifiers score 0.5 balanced, where plain accuracy
    rates one of them at 90%."""
    truth = [True] * 10 + [False] * 90
    never = binary_classification_report(truth, [False] * 100)
    always = binary_classification_report(truth, [True] * 100)
    assert never.balanced_accuracy == pytest.approx(0.5)
    assert always.balanced_accuracy == pytest.approx(0.5)


def test_a_perfect_alarm_scores_perfectly():
    truth = [True, True, False, False]
    report = binary_classification_report(truth, truth)
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.specificity == 1.0
    assert report.f1 == 1.0


def test_confusion_grid_is_laid_out_for_a_heatmap():
    truth = [True, True, False, False, False]
    predicted = [True, False, True, False, False]
    grid = binary_classification_report(truth, predicted).matrix.as_grid()
    assert grid == [[2, 1], [1, 1]], "[[TN, FP], [FN, TP]]"


def test_precision_and_recall_are_reported_separately():
    """A false negative publishes a bad batch; a false positive delays a good
    one. Any single score has silently picked an exchange rate between them."""
    report = binary_classification_report([True] * 5 + [False] * 5, [True] * 3 + [False] * 7)
    assert report.recall == pytest.approx(0.6)
    assert report.specificity == pytest.approx(1.0)
    assert report.precision == pytest.approx(1.0)


def test_a_balanced_problem_is_not_flagged_as_misleading():
    truth = [True] * 50 + [False] * 50
    assert not binary_classification_report(truth, truth).accuracy_is_misleading


def test_classification_requires_observations():
    with pytest.raises(ValueError, match="no observations"):
        binary_classification_report([], [])


# ---------------------------------------------------------------------------
# Elasticity health
# ---------------------------------------------------------------------------


def test_positive_elasticities_are_a_diagnostic_not_a_finding():
    """Almost always residual confounding rather than a Giffen good, so the rate
    is a statement about the estimator."""
    health = elasticity_health([-1.8, -1.5, -2.0, 0.4], [0.3] * 4)
    assert health["sign_violation_rate"] == pytest.approx(0.25)
    assert health["sign_violations_breached"]
    assert not health["healthy"]


def test_a_healthy_elasticity_set_passes():
    health = elasticity_health([-1.8] * 100, [0.3] * 100)
    assert not health["sign_violations_breached"]
    assert health["healthy"]


def test_wide_intervals_fail_the_health_check():
    health = elasticity_health([-1.8] * 100, [5.0] * 100, max_ci_width=1.0)
    assert health["ci_width_breached"]
    assert not health["healthy"]


def test_declined_estimates_are_reported():
    """A model that refuses to answer for half the catalogue is not healthy just
    because the half it answered for looks tidy."""
    health = elasticity_health([-1.8] * 10, [0.3] * 10, n_declined=90)
    assert health["n_declined"] == 90
    assert health["n"] == 10


def test_no_finite_estimates_is_not_healthy():
    assert not elasticity_health([float("nan")], [float("nan")])["healthy"]
