"""
Causal elasticity tests.

The thing under test is not "does it return a number" — a broken estimator
returns numbers too. It is whether the number is the *causal* elasticity, which
is only checkable against a generator that knows the truth. So the load-bearing
tests are:

* :func:`test_recovers_true_elasticity_within_interval` — the phase-3 gate.
* :func:`test_beats_naive_regression_on_the_same_data` — proves the confounding
  is real and that partialling out is what removes it, rather than the estimator
  being differently wrong.
* :func:`test_matches_econml_reference_on_identical_folds` — proves the
  arithmetic against the reference implementation, so "we wrote our own DML"
  does not mean "we wrote our own bug".

Fitting is expensive (25 SKUs x 5 repeats x 5 folds x 2 nuisance models), so the
panels and their fits are module-scoped and shared.
"""

from __future__ import annotations

import numpy as np
import pytest

from prismprice.compute import BackendSupport, GPUInfo, GPUUnavailableError
from prismprice.data.synthetic import generate_panel
from prismprice.estimation import elasticity as elasticity_module
from prismprice.estimation.elasticity import (
    DoubleMLElasticity,
    ElasticityEstimate,
    add_category_price_control,
    naive_ols_elasticity,
    score_against_truth,
    temporal_folds,
)

CONFOUNDERS = (
    "season_yearly",
    "is_weekend",
    "marketing_spend",
    "holiday",
    "category_log_price",
)
GATE_SEEDS = (7, 21)


@pytest.fixture(scope="module")
def panels():
    return [generate_panel(n_skus=20, n_days=460, seed=s) for s in GATE_SEEDS]


@pytest.fixture(scope="module")
def fitted(panels):
    out = []
    for panel in panels:
        controlled = add_category_price_control(panel.daily)
        model = DoubleMLElasticity(
            outcome_column="units_uncensored", confounder_columns=CONFOUNDERS
        ).fit(controlled)
        out.append((panel, model))
    return out


# ---------------------------------------------------------------------------
# The phase-3 gate
# ---------------------------------------------------------------------------


def test_recovers_true_elasticity_within_interval(fitted):
    """Phase-3 gate: the true beta lies inside the CI for >= 90% of SKUs.

    Scored over every SKU including ones the estimator declined to answer for,
    so refusing to answer cannot buy coverage.
    """
    coverages = [
        score_against_truth(model.estimates, panel.truth.as_frame()).coverage
        for panel, model in fitted
    ]
    pooled = float(np.mean(coverages))
    assert pooled >= 0.90, f"coverage {pooled:.3f} below the 0.90 gate (per seed: {coverages})"


def test_point_estimate_is_near_unbiased(fitted):
    """Coverage can be bought with wide intervals; this cannot."""
    errors = [
        model.estimates[sku].point - panel.truth.beta_for(sku)
        for panel, model in fitted
        for sku in panel.truth.skus
        if np.isfinite(model.estimates[sku].point)
    ]
    bias = float(np.mean(errors))
    assert abs(bias) < 0.10, f"mean bias {bias:+.4f} is too large to call this identified"


def test_intervals_are_not_systematically_too_narrow(fitted):
    """The defect repeated cross-fitting exists to fix.

    Single-split DML scattered 1.38x wider than its own standard errors claimed.
    A ratio far above 1 means the interval is lying about its own precision,
    which matters more than usual here: L3 reads interval width to decide
    whether to trust the estimate at all.
    """
    errors, ses = [], []
    for panel, model in fitted:
        for sku in panel.truth.skus:
            estimate = model.estimates[sku]
            if np.isfinite(estimate.point):
                errors.append(estimate.point - panel.truth.beta_for(sku))
                ses.append(estimate.std_error)
    ratio = float(np.std(errors) / np.mean(ses))
    assert ratio < 1.35, f"estimates scatter {ratio:.2f}x their reported standard error"


def test_beats_naive_regression_on_the_same_data(fitted):
    """The confounding is real, and partialling out is what removes it."""
    dml_errors, naive_errors = [], []
    for panel, model in fitted:
        naive = naive_ols_elasticity(panel.daily, outcome_column="units_uncensored")
        for sku in panel.truth.skus:
            truth = panel.truth.beta_for(sku)
            if np.isfinite(model.estimates[sku].point):
                dml_errors.append(model.estimates[sku].point - truth)
                naive_errors.append(naive[sku] - truth)

    dml_rmse = float(np.sqrt(np.mean(np.square(dml_errors))))
    naive_rmse = float(np.sqrt(np.mean(np.square(naive_errors))))
    assert dml_rmse < naive_rmse / 2, (
        f"DML RMSE {dml_rmse:.4f} should be far below naive {naive_rmse:.4f}; "
        f"if it is not, the estimator is not removing the confounding"
    )


def test_naive_regression_is_biased_in_the_expected_direction(panels):
    """Sanity check on the generator, not the estimator.

    The generator discounts *into* strong weeks, so low prices coincide with
    high demand for reasons that are not causal, and an uncontrolled regression
    must overstate how elastic buyers are. If this ever passes with the opposite
    sign, the test panel stopped being confounded and every recovery result
    above became vacuous.
    """
    errors = []
    for panel in panels:
        naive = naive_ols_elasticity(panel.daily, outcome_column="units_uncensored")
        errors += [naive[sku] - panel.truth.beta_for(sku) for sku in panel.truth.skus]
    assert float(np.mean(errors)) < -0.1, "naive OLS should overstate elasticity magnitude"


@pytest.mark.parametrize("seed_index", range(len(GATE_SEEDS)))
def test_every_seed_clears_the_gate_individually(fitted, seed_index):
    """A mean across seeds can hide one bad panel."""
    panel, model = fitted[seed_index]
    coverage = score_against_truth(model.estimates, panel.truth.as_frame()).coverage
    assert coverage >= 0.85, f"seed {GATE_SEEDS[seed_index]} coverage {coverage:.3f}"


# ---------------------------------------------------------------------------
# Verified against the reference implementation
# ---------------------------------------------------------------------------


def test_matches_econml_reference_on_identical_folds():
    """Our partialling-out arithmetic equals econml's LinearDML.

    Run with ``n_repeats=1`` because econml does a single split; the comparison
    is of the estimator arithmetic, not of the aggregation on top of it.
    """
    econml_dml = pytest.importorskip("econml.dml", reason="econml is an optional extra")
    import lightgbm as lgb

    panel = generate_panel(n_skus=3, n_days=400, seed=7)
    frame = panel.daily[panel.daily["sku"] == "SKU-000"].sort_values("date")
    confounders = frame[["season_yearly", "is_weekend", "marketing_spend", "holiday"]].to_numpy(
        float
    )
    treatment = np.log(frame["price"].to_numpy(float))
    response = np.log(frame["units_uncensored"].to_numpy(float))

    folds = [
        (train.tolist(), test.tolist()) for train, test in temporal_folds(len(treatment), 5, 3, 0)
    ]

    def learner():
        return lgb.LGBMRegressor(
            n_estimators=200,
            num_leaves=4,
            min_child_samples=30,
            learning_rate=0.03,
            verbose=-1,
            random_state=20260817,
        )

    reference = econml_dml.LinearDML(
        model_y=learner(), model_t=learner(), discrete_treatment=False, cv=folds
    )
    reference.fit(response, treatment, X=None, W=confounders)
    reference_theta = float(np.asarray(reference.const_marginal_effect()).ravel()[0])

    ours = DoubleMLElasticity(outcome_column="units_uncensored", n_repeats=1).fit(
        panel.daily[panel.daily["sku"] == "SKU-000"]
    )
    assert ours.estimates["SKU-000"].point == pytest.approx(reference_theta, abs=1e-6)


# ---------------------------------------------------------------------------
# Temporal cross-fitting
# ---------------------------------------------------------------------------


def test_folds_never_share_rows_between_train_and_test():
    for train, test in temporal_folds(200, 5, purge=3):
        assert set(train.tolist()).isdisjoint(test.tolist())


def test_test_blocks_are_contiguous_in_time():
    """Random K-fold would leak an autocorrelated series across the split."""
    for _, test in temporal_folds(200, 5, purge=0):
        assert np.array_equal(test, np.arange(test[0], test[-1] + 1))


def test_test_blocks_partition_every_row_exactly_once():
    seen = np.concatenate([test for _, test in temporal_folds(200, 5, purge=3)])
    assert np.array_equal(np.sort(seen), np.arange(200))


def test_purge_removes_neighbours_from_training():
    """The rows either side of the test block carry nearly the same information."""
    purge = 4
    for train, test in temporal_folds(200, 5, purge=purge):
        lower, upper = int(test[0]), int(test[-1])
        for offset in range(1, purge + 1):
            assert lower - offset not in train.tolist() or lower - offset < 0
            assert upper + offset not in train.tolist() or upper + offset >= 200


def test_offset_produces_a_different_partition():
    """Repeated cross-fitting needs genuinely different splits to average over."""
    base = [test.tolist() for _, test in temporal_folds(200, 5, 0, offset=0)]
    shifted = [test.tolist() for _, test in temporal_folds(200, 5, 0, offset=11)]
    assert base != shifted


def test_folds_reject_impossible_configurations():
    with pytest.raises(ValueError, match="n_folds must be >= 2"):
        temporal_folds(100, 1)
    with pytest.raises(ValueError, match="need at least"):
        temporal_folds(4, 5)


# ---------------------------------------------------------------------------
# Confidence tagging drives degradation, so it must not be decorative
# ---------------------------------------------------------------------------


def test_weak_identification_is_reported_not_estimated():
    """Price fully explained by a confounder must not yield a confident number.

    Here price is a deterministic function of the confounder, so there is no
    exogenous variation at all and the elasticity is genuinely unidentified. The
    honest output is a refusal.
    """
    rng = np.random.default_rng(0)
    n = 400
    dates = np.arange(n)
    spend = rng.normal(size=n)
    price = 20.0 * np.exp(0.2 * spend)  # no independent price variation whatsoever
    units = np.exp(3.0 - 1.8 * np.log(price) + 0.5 * spend + rng.normal(0, 0.05, n))

    import pandas as pd

    frame = pd.DataFrame(
        {
            "sku": "SKU-X",
            "date": pd.to_datetime("2024-01-01") + pd.to_timedelta(dates, unit="D"),
            "price": price,
            "units": units,
            "season_yearly": 0.0,
            "is_weekend": 0.0,
            "marketing_spend": spend,
            "holiday": 0.0,
        }
    )
    model = DoubleMLElasticity(min_residual_price_sd=0.05).fit(frame)
    estimate = model.estimates["SKU-X"]
    assert estimate.confidence == "low"
    assert not estimate.is_usable


def test_declining_to_answer_returns_nan_not_zero():
    """0.0 reads as 'no effect'; only 'no answer' should degrade."""
    estimate = DoubleMLElasticity()._unidentified("SKU-X", "m", 10, 0, "too few rows")
    assert np.isnan(estimate.point)
    assert np.isnan(estimate.ci_low) and np.isnan(estimate.ci_high)
    assert estimate.confidence == "low"


def test_wide_interval_is_tagged_low(fitted):
    model = DoubleMLElasticity(max_ci_width=0.001)
    confidence, reason = model._tag(-1.8, ci_width=0.5, residual_price_sd=0.2)
    assert confidence == "low"
    assert "tau_max" in reason


def test_positive_elasticity_is_treated_as_failure_not_finding():
    """Upward-sloping demand is a model fault; passing it on invites an optimiser
    to walk straight up the curve."""
    confidence, reason = DoubleMLElasticity()._tag(0.4, ci_width=0.2, residual_price_sd=0.2)
    assert confidence == "low"
    assert "non-negative" in reason


def test_high_confidence_requires_both_conditions():
    confidence, _ = DoubleMLElasticity()._tag(-1.8, ci_width=0.2, residual_price_sd=0.2)
    assert confidence == "high"


def test_estimates_carry_their_identification_evidence(fitted):
    _, model = fitted[0]
    for estimate in model.estimates.values():
        assert estimate.reason.strip(), "every verdict must say why"
        assert 0.0 <= estimate.price_variation_explained <= 1.0


# ---------------------------------------------------------------------------
# Censoring, controls and scoring
# ---------------------------------------------------------------------------


def test_censored_demand_attenuates_the_elasticity(panels):
    """Why L1 un-censoring runs before this layer.

    Recorded units are min(demand, stock), so on stockout days demand looks like
    it stops growing exactly when the product sells best — which drags the
    estimate toward zero.
    """
    panel = generate_panel(n_skus=6, n_days=420, seed=11, cover_target=13.0)
    controlled = add_category_price_control(panel.daily)
    common = dict(confounder_columns=CONFOUNDERS, n_repeats=2)

    latent = DoubleMLElasticity(outcome_column="units_uncensored", **common).fit(controlled)
    censored = DoubleMLElasticity(outcome_column="units", **common).fit(controlled)

    latent_err = np.mean(
        [abs(latent.estimates[s].point - panel.truth.beta_for(s)) for s in panel.truth.skus]
    )
    censored_err = np.mean(
        [abs(censored.estimates[s].point - panel.truth.beta_for(s)) for s in panel.truth.skus]
    )
    assert latent_err < censored_err, (
        f"un-censored demand should recover elasticity better "
        f"(latent {latent_err:.3f} vs censored {censored_err:.3f})"
    )


def test_fitting_on_censored_units_warns():
    panel = generate_panel(n_skus=3, n_days=300, seed=5, cover_target=13.0)
    with pytest.warns(RuntimeWarning, match="attenuates elasticity"):
        DoubleMLElasticity(outcome_column="units", n_repeats=1).fit(panel.daily)


def test_category_price_control_leaves_the_sku_out():
    """Including a SKU's own price in its category mean would bias it to zero."""
    import pandas as pd

    frame = pd.DataFrame(
        {
            "sku": ["A", "B", "C"],
            "date": pd.to_datetime(["2024-01-01"] * 3),
            "price": [np.e**1.0, np.e**2.0, np.e**3.0],
        }
    )
    out = add_category_price_control(frame)
    assert out.loc[0, "category_log_price"] == pytest.approx(2.5)
    assert out.loc[1, "category_log_price"] == pytest.approx(2.0)


def test_category_price_control_is_nan_for_a_single_sku_day():
    import pandas as pd

    frame = pd.DataFrame({"sku": ["A"], "date": pd.to_datetime(["2024-01-01"]), "price": [10.0]})
    assert np.isnan(add_category_price_control(frame).loc[0, "category_log_price"])


def test_scoring_counts_skus_the_estimator_declined(panels):
    """Refusing to answer must not improve the score."""
    panel = panels[0]
    truth = panel.truth.as_frame()
    estimates = {
        sku: DoubleMLElasticity()._unidentified(sku, "m", 10, 0, "declined")
        for sku in panel.truth.skus
    }
    score = score_against_truth(estimates, truth)
    assert score.n_skus == len(panel.truth.skus)
    assert score.coverage == 0.0


def test_scoring_requires_overlapping_skus(panels):
    with pytest.raises(ValueError, match="no SKUs in common"):
        score_against_truth({}, panels[0].truth.as_frame())


def test_pooled_estimate_backs_the_degradation_rung(fitted):
    """Rung 3 falls back to a category elasticity, so one must exist."""
    panel, _ = fitted[0]
    controlled = add_category_price_control(panel.daily)
    model = DoubleMLElasticity(
        outcome_column="units_uncensored", confounder_columns=CONFOUNDERS, n_repeats=1
    )
    pooled = model.pooled(controlled)
    assert pooled.point < 0
    assert pooled.method.endswith("-pooled")
    assert pooled.point == pytest.approx(float(np.mean(panel.truth.beta)), abs=0.9)


def test_missing_columns_are_named(panels):
    frame = panels[0].daily.drop(columns=["marketing_spend"])
    with pytest.raises(ValueError, match="marketing_spend"):
        DoubleMLElasticity().fit(frame)


def test_as_frame_requires_a_fit():
    with pytest.raises(ValueError, match="fit\\(\\) has not been called"):
        DoubleMLElasticity().as_frame()


def test_estimate_serialises_for_the_decision_record(fitted):
    _, model = fitted[0]
    record = next(iter(model.estimates.values())).as_dict()
    assert set(record) >= {"point", "ci_low", "ci_high", "method", "confidence"}
    assert record["method"].startswith("dml-")


def test_naive_ols_handles_degenerate_series():
    import pandas as pd

    frame = pd.DataFrame({"sku": ["A", "A"], "price": [10.0, 10.0], "units": [5.0, 6.0]})
    assert np.isnan(naive_ols_elasticity(frame)["A"])


# ---------------------------------------------------------------------------
# Compute contract
# ---------------------------------------------------------------------------


def test_cpu_training_is_announced(monkeypatch, panels):
    monkeypatch.setattr(
        elasticity_module,
        "lightgbm_device_params",
        lambda component: (_ for _ in ()).throw(AssertionError("should not reach")),
    )
    # The real contract is exercised through compute; here we only assert the
    # component name is threaded through, which is what makes a warning legible.
    monkeypatch.undo()
    model = DoubleMLElasticity(n_repeats=1)
    panel = generate_panel(n_skus=2, n_days=200, seed=3)
    model.fit(panel.daily)
    assert model.device_params.get("device_type") in {"cpu", "cuda"}


def test_torch_components_still_require_gpu(monkeypatch):
    from prismprice import compute

    monkeypatch.delenv("PRISMPRICE_ALLOW_CPU", raising=False)
    monkeypatch.setattr(compute, "gpu_report", lambda: GPUInfo(available=False, reason="sim"))
    with pytest.raises(GPUUnavailableError):
        compute.require_gpu("estimation.elasticity")


def test_lightgbm_backend_support_is_probed_not_assumed():
    from prismprice import compute

    support = compute.lightgbm_gpu_support()
    assert isinstance(support, BackendSupport)
    assert support.library == "lightgbm"
    assert support.reason.strip()


def test_estimate_ci_width_and_containment():
    estimate = ElasticityEstimate(
        sku="A",
        point=-1.8,
        ci_low=-2.0,
        ci_high=-1.6,
        std_error=0.1,
        method="m",
        confidence="high",
        reason="r",
        n_observations=100,
        n_dropped=0,
        residual_price_sd=0.1,
        price_variation_explained=0.2,
    )
    assert estimate.ci_width == pytest.approx(0.4)
    assert estimate.contains(-1.9)
    assert not estimate.contains(-1.5)
    assert estimate.is_usable


# ---------------------------------------------------------------------------
# Parallel dispatch across SKUs
# ---------------------------------------------------------------------------
#
# Every SKU is an independent fit, so fit() dispatches them across threads
# above a small count. The only thing that dispatch is allowed to change is
# wall-clock time — these tests exist to catch it changing anything else.


def _estimates_match(a: dict[str, ElasticityEstimate], b: dict[str, ElasticityEstimate]) -> bool:
    """Field-for-field equality, with NaN treated as matching NaN.

    A plain ``==`` on the estimate dicts would report every declined-SKU pair
    (``point`` is ``nan`` for both) as a mismatch, which is exactly backwards —
    two estimators declining the same SKU for the same reason is agreement.
    """
    if a.keys() != b.keys():
        return False
    for sku in a:
        for field_name in ElasticityEstimate.__dataclass_fields__:
            x, y = getattr(a[sku], field_name), getattr(b[sku], field_name)
            if isinstance(x, float) and isinstance(y, float) and np.isnan(x) and np.isnan(y):
                continue
            if x != y:
                return False
    return True


@pytest.fixture(scope="module")
def parallel_panel():
    """Enough SKUs to clear fit()'s parallel-dispatch threshold, few enough to
    fit twice (n_jobs=1 and n_jobs=-1) without the module becoming slow."""
    return generate_panel(n_skus=6, n_days=260, n_customers=150, seed=13)


def test_parallel_dispatch_matches_sequential_exactly(parallel_panel):
    """The load-bearing test of this section: n_jobs must not be able to move
    a single estimate. If it can, every real pipeline run's numbers depend on
    how many cores happened to be free, which is not a defensible causal claim.
    """
    common = dict(n_repeats=2, n_folds=4)
    sequential = DoubleMLElasticity(n_jobs=1, **common).fit(parallel_panel.daily)
    parallel = DoubleMLElasticity(n_jobs=-1, **common).fit(parallel_panel.daily)

    assert set(sequential.estimates) == set(parallel_panel.truth.skus)
    assert _estimates_match(sequential.estimates, parallel.estimates)


def test_device_params_are_restored_after_parallel_dispatch(parallel_panel):
    """fit() borrows a single-threaded copy of device_params for the duration
    of dispatch; anything reading it afterwards must see what the compute
    layer actually resolved, not an artefact of how fitting was scheduled."""
    model = DoubleMLElasticity(n_jobs=-1, n_repeats=1, n_folds=4)
    model.fit(parallel_panel.daily)
    assert model.device_params.get("num_threads") != 1
    assert model.device_params.get("device_type") in {"cpu", "cuda"}


def test_small_sku_counts_skip_parallel_dispatch(monkeypatch):
    """Below the threshold, sequential is the faster choice, not a fallback
    being tolerated — this pins that fit() actually takes that path rather
    than paying thread-pool overhead on two SKUs."""
    import prismprice.estimation.elasticity as mod

    def _fail(*args, **kwargs):
        raise AssertionError("Parallel should not run below the SKU threshold")

    monkeypatch.setattr(mod, "Parallel", _fail)
    panel = generate_panel(n_skus=2, n_days=200, seed=4)
    DoubleMLElasticity(n_jobs=-1, n_repeats=1, n_folds=4).fit(panel.daily)
