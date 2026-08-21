"""
Drift detection tests.

The property that matters most is the one a broken drift monitor would fail
silently: :func:`test_binning_on_the_current_sample_would_hide_a_shift`
constructs a wholesale distribution shift and shows that re-binning on the
current data reports it as stable. A monitor that returns a green number for a
real shift is worse than no monitor, and it cannot be caught by testing the
arithmetic — only by testing the choice of bin edges.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prismprice.observability.drift import (
    categorical_drift,
    feature_drift,
    population_stability_index,
)


def normal(loc: float, scale: float, n: int = 5000, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(loc, scale, n)


# ---------------------------------------------------------------------------
# The statistic
# ---------------------------------------------------------------------------


def test_identical_distributions_have_no_drift():
    sample = normal(0.0, 1.0, seed=1)
    assert population_stability_index(sample, sample).psi == pytest.approx(0.0, abs=1e-9)


def test_same_distribution_different_draws_is_stable():
    """Sampling noise must not trip a significance threshold."""
    report = population_stability_index(normal(0.0, 1.0, seed=1), normal(0.0, 1.0, seed=2))
    assert report.verdict == "STABLE"
    assert not report.breached


def test_a_shifted_mean_registers_as_drift():
    report = population_stability_index(normal(0.0, 1.0, seed=1), normal(1.5, 1.0, seed=2))
    assert report.breached
    assert report.verdict == "SIGNIFICANT"


def test_psi_grows_monotonically_with_the_size_of_the_shift():
    reference = normal(0.0, 1.0, seed=1)
    psis = [
        population_stability_index(reference, normal(shift, 1.0, seed=2)).psi
        for shift in (0.0, 0.25, 0.5, 1.0, 2.0)
    ]
    assert psis == sorted(psis), f"PSI not monotone in the shift: {psis}"


def test_psi_is_symmetric_in_its_arguments():
    """'Training had customers live does not' and the reverse are both problems,
    which is why PSI is used here rather than a directional divergence."""
    a, b = normal(0.0, 1.0, seed=1), normal(0.8, 1.2, seed=2)
    forward = population_stability_index(a, b).psi
    backward = population_stability_index(b, a).psi
    assert forward == pytest.approx(backward, rel=0.15)


def test_a_variance_change_is_detected_even_with_the_same_mean():
    """A narrowing distribution has the same centre and different behaviour."""
    report = population_stability_index(normal(0.0, 1.0, seed=1), normal(0.0, 0.3, seed=2))
    assert report.breached


# ---------------------------------------------------------------------------
# The choice of bin edges
# ---------------------------------------------------------------------------


def test_binning_on_the_current_sample_would_hide_a_shift():
    """The defect this module exists to avoid, demonstrated.

    Quantile bins computed on whatever arrived today contain roughly equal
    shares of today's data by construction, so a wholesale shift reports as
    stable. Binning on the reference catches it. This cannot be caught by
    checking the arithmetic — only by checking which sample the edges came from.
    """
    reference = normal(0.0, 1.0, seed=1)
    current = normal(4.0, 1.0, seed=2)

    correct = population_stability_index(reference, current).psi

    # What a naive implementation does: edges from the current sample.
    edges = np.quantile(current, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(reference, bins=edges)[0] / reference.size
    actual = np.histogram(current, bins=edges)[0] / current.size
    naive = float(
        np.sum(
            (np.clip(actual, 1e-6, None) - np.clip(expected, 1e-6, None))
            * np.log(np.clip(actual, 1e-6, None) / np.clip(expected, 1e-6, None))
        )
    )

    assert correct > 1.0, "a four-sigma shift must register loudly"
    assert correct > naive, (
        f"binning on the current sample understated the shift: {naive:.2f} vs {correct:.2f}"
    )


def test_values_beyond_the_reference_range_are_counted_not_dropped():
    """A value the model has never seen is the most important thing a drift
    monitor can notice, so the outer bins are open."""
    reference = np.linspace(0.0, 1.0, 2000)
    current = np.concatenate([np.linspace(0.0, 1.0, 1000), np.full(1000, 99.0)])
    report = population_stability_index(reference, current)
    assert report.breached
    assert report.n_current == 2000, "out-of-range values were dropped"


def test_bin_count_travels_with_the_result():
    """PSI is not comparable across bin counts, so a report without one is not
    interpretable."""
    reference, current = normal(0.0, 1.0, seed=1), normal(0.5, 1.0, seed=2)
    coarse = population_stability_index(reference, current, n_bins=5)
    fine = population_stability_index(reference, current, n_bins=50)
    assert coarse.n_bins == 5
    assert fine.n_bins == 50
    assert fine.psi > coarse.psi, "more bins should register more of the same shift"


def test_a_constant_reference_feature_cannot_drift():
    """No distribution to move away from. Reporting drift here would be noise."""
    report = population_stability_index(np.full(500, 3.0), np.full(500, 7.0))
    assert report.psi == 0.0
    assert report.n_bins == 1


def test_repeated_quantiles_reduce_the_realised_bin_count():
    """A feature that is mostly one value yields duplicate edges; the report
    states the count actually used rather than the one requested."""
    reference = np.concatenate([np.zeros(900), np.linspace(1.0, 2.0, 100)])
    report = population_stability_index(reference, reference, n_bins=20)
    assert report.n_bins < 20


def test_largest_contributor_points_at_the_moved_region():
    """After 'something drifted', the next question is always 'which part'."""
    reference = normal(0.0, 1.0, seed=1)
    current = np.concatenate([normal(0.0, 1.0, 4000, seed=2), np.full(1000, 5.0)])
    report = population_stability_index(reference, current)
    assert report.largest_contributor == report.n_bins - 1, "the top bin took the mass"


def test_floored_bins_are_reported():
    """Flooring caps how loud a vacated bin can be, so the count has to travel
    with the number rather than being swallowed."""
    reference = normal(0.0, 1.0, seed=1)
    # Only the upper half of the distribution now arrives, so every bin below
    # the reference median is empty and has to be floored.
    sample = normal(0.0, 1.0, seed=2)
    current = sample[sample > 0.0]
    report = population_stability_index(reference, current)
    assert report.floored_bins > 0
    assert report.breached


# ---------------------------------------------------------------------------
# Categorical and frame-level
# ---------------------------------------------------------------------------


def test_categorical_shares_drift():
    reference = pd.Series(["a"] * 700 + ["b"] * 300)
    current = pd.Series(["a"] * 300 + ["b"] * 700)
    assert categorical_drift(reference, current).breached


def test_a_vanished_category_is_drift_not_an_absence():
    """A discontinued brand or a closed store is the most obvious kind of
    change; dropping unmatched categories would make it invisible."""
    reference = pd.Series(["a"] * 500 + ["b"] * 500)
    current = pd.Series(["a"] * 1000)
    report = categorical_drift(reference, current)
    assert report.breached
    assert report.n_bins == 2, "the vanished category must still be a bin"


def test_a_new_category_is_drift():
    reference = pd.Series(["a"] * 1000)
    current = pd.Series(["a"] * 500 + ["z"] * 500)
    assert categorical_drift(reference, current).breached


def test_frame_drift_reports_every_shared_column():
    reference = pd.DataFrame({"x": normal(0.0, 1.0, seed=1), "y": normal(5.0, 1.0, seed=3)})
    current = pd.DataFrame({"x": normal(0.0, 1.0, seed=2), "y": normal(9.0, 1.0, seed=4)})
    reports = feature_drift(reference, current)
    assert set(reports) == {"x", "y"}
    assert reports["y"].breached
    assert not reports["x"].breached


def test_frame_drift_is_ordered_worst_first():
    """A caller reading the top of the dict should be reading what to look at."""
    reference = pd.DataFrame({"calm": normal(0.0, 1.0, seed=1), "wild": normal(0.0, 1.0, seed=3)})
    current = pd.DataFrame({"calm": normal(0.0, 1.0, seed=2), "wild": normal(3.0, 1.0, seed=4)})
    assert next(iter(feature_drift(reference, current))) == "wild"


def test_frames_with_no_shared_columns_are_a_wiring_fault():
    """Reporting 'no drift' for a frame that was never compared is the worst
    possible answer, because it is reassuring."""
    with pytest.raises(ValueError, match="wiring fault"):
        feature_drift(pd.DataFrame({"a": [1.0, 2.0]}), pd.DataFrame({"b": [1.0, 2.0]}))


def test_mixed_dtype_columns_use_the_categorical_path():
    reference = pd.DataFrame({"brand": ["x"] * 800 + ["y"] * 200})
    current = pd.DataFrame({"brand": ["x"] * 200 + ["y"] * 800})
    assert feature_drift(reference, current)["brand"].breached


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_empty_samples_are_refused():
    with pytest.raises(ValueError, match="non-empty"):
        population_stability_index(np.array([]), normal(0.0, 1.0))


def test_too_few_bins_are_refused():
    with pytest.raises(ValueError, match="n_bins must be >= 2"):
        population_stability_index(normal(0.0, 1.0), normal(0.0, 1.0), n_bins=1)


def test_non_finite_values_are_excluded():
    reference = np.concatenate([normal(0.0, 1.0, 1000, seed=1), [np.nan, np.inf]])
    report = population_stability_index(reference, normal(0.0, 1.0, 1000, seed=2))
    assert report.n_reference == 1000


def test_report_serialises_for_a_dashboard():
    record = population_stability_index(
        normal(0.0, 1.0, seed=1), normal(1.0, 1.0, seed=2)
    ).as_dict()
    assert set(record) >= {"psi", "verdict", "breached", "n_bins", "floored_bins"}
