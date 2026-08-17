"""
Synthetic generator tests (L0).

The generator is the measuring instrument for every later phase, so it gets
calibrated first. If the confounding is not really there, Phase 3's DML has
nothing to prove; if the censoring is not really there, Phase 1's un-censoring
scores itself against a problem that does not exist.
"""

import numpy as np
import pandas as pd
import pytest

from prismprice.data.contracts import TRANSACTIONS_CONTRACT, validate
from prismprice.data.synthetic import generate_panel, transactions_from_panel


@pytest.fixture(scope="module")
def panel():
    return generate_panel(n_skus=6, n_days=420, n_customers=250, seed=7)


def ols_elasticity(frame: pd.DataFrame, demand_column: str) -> float:
    """Naive log-log regression of demand on own price, ignoring confounders."""
    design = np.column_stack([np.ones(len(frame)), np.log(frame["price"].to_numpy())])
    target = np.log(np.maximum(frame[demand_column].to_numpy(), 1e-6))
    return float(np.linalg.lstsq(design, target, rcond=None)[0][1])


# ---------------------------------------------------------------------------
# Structure and determinism
# ---------------------------------------------------------------------------


def test_panel_shape_and_keys(panel):
    assert len(panel.daily) == 6 * 420
    assert not panel.daily.duplicated(["sku", "date"]).any()
    assert set(panel.skus) == set(panel.daily["sku"].unique())


def test_generation_is_deterministic_given_a_seed():
    a = generate_panel(n_skus=3, n_days=120, n_customers=40, seed=99)
    b = generate_panel(n_skus=3, n_days=120, n_customers=40, seed=99)
    pd.testing.assert_frame_equal(a.daily, b.daily)
    pd.testing.assert_frame_equal(a.purchases, b.purchases)
    np.testing.assert_array_equal(a.truth.beta, b.truth.beta)


def test_different_seeds_give_different_panels():
    a = generate_panel(n_skus=3, n_days=120, n_customers=40, seed=1)
    b = generate_panel(n_skus=3, n_days=120, n_customers=40, seed=2)
    assert not np.allclose(a.truth.beta, b.truth.beta)


def test_rejects_a_panel_too_short_to_replenish():
    with pytest.raises(ValueError, match="replenishment"):
        generate_panel(n_skus=2, n_days=10, n_customers=10)


# ---------------------------------------------------------------------------
# Economic sanity
# ---------------------------------------------------------------------------


def test_true_elasticities_are_negative(panel):
    assert (panel.truth.beta < 0).all()


def test_cross_price_elasticities_are_substitutes_with_zero_diagonal(panel):
    assert np.allclose(np.diag(panel.truth.eta), 0.0)
    off_diagonal = panel.truth.eta[~np.eye(len(panel.skus), dtype=bool)]
    assert (off_diagonal >= 0).all()


def test_cost_is_below_price_everywhere(panel):
    assert (panel.daily["unit_cost"] < panel.daily["price"]).mean() > 0.95


def test_prices_and_demand_are_positive_and_finite(panel):
    for column in ("price", "units", "units_uncensored", "unit_cost"):
        values = panel.daily[column]
        assert np.isfinite(values).all()
        assert (values >= 0).all()


# ---------------------------------------------------------------------------
# The two properties later phases depend on
# ---------------------------------------------------------------------------


def test_confounding_actually_biases_a_naive_estimator(panel):
    """If naive OLS recovered the truth, Phase 3's DML would prove nothing."""
    truth = panel.truth
    biases = [
        ols_elasticity(panel.daily_for(sku), "units_uncensored") - truth.beta_for(sku)
        for sku in panel.skus
    ]
    assert np.abs(np.mean(biases)) > 0.3, (
        f"naive estimator is nearly unbiased (mean bias {np.mean(biases):.3f}); "
        "the confounding is too weak to be a useful test bed"
    )


def test_exogenous_prices_let_a_naive_estimator_succeed():
    """The control condition: with confounding switched off, OLS should be close.

    This is what makes the previous test meaningful — it shows the bias comes
    from the confounding rather than from a broken generator.
    """
    panel = generate_panel(
        n_skus=4,
        n_days=700,
        n_customers=40,
        seed=11,
        confounding_strength=0.0,
        cross_price_strength=0.0,
        noise_sd=0.08,
    )
    biases = [
        ols_elasticity(panel.daily_for(sku), "units_uncensored") - panel.truth.beta_for(sku)
        for sku in panel.skus
    ]
    assert np.abs(np.mean(biases)) < 0.25, f"mean bias {np.mean(biases):.3f} with exogenous prices"


def test_observed_demand_is_censored_below_latent_demand(panel):
    daily = panel.daily
    assert (daily["units"] <= daily["units_uncensored"] + 1e-9).all()
    assert daily["units"].sum() < daily["units_uncensored"].sum()


def test_censoring_is_confined_to_stockout_days(panel):
    daily = panel.daily
    censored = daily["units"] < daily["units_uncensored"] - 1e-9
    assert (censored == daily["stockout"]).all()


def test_low_cover_target_produces_enough_stockouts_to_score_uncensoring():
    stressed = generate_panel(n_skus=4, n_days=300, n_customers=40, seed=3, cover_target=14.0)
    assert stressed.daily["stockout"].mean() > 0.05


def test_default_cover_target_is_realistic():
    """A panel where a fifth of days are stockouts is not a retail panel."""
    assert (
        0.0 < generate_panel(n_skus=4, n_days=300, n_customers=40).daily["stockout"].mean() < 0.10
    )


# ---------------------------------------------------------------------------
# Inventory, competitor and retention layers
# ---------------------------------------------------------------------------


def test_inventory_never_goes_negative_and_cover_is_finite(panel):
    assert (panel.daily["inventory_on_hand"] >= 0).all()
    assert np.isfinite(panel.daily["inventory_cover_days"]).all()


def test_shadow_price_binds_only_below_the_cover_threshold(panel):
    from prismprice import config

    daily = panel.daily
    above = daily[daily["inventory_cover_days"] >= config.DEFAULT_MIN_INVENTORY_COVER_DAYS]
    assert (above["inventory_shadow_price"] == 0).all()


def test_competitor_feed_is_stale_some_of_the_time(panel):
    """Rung 2 must be reachable in a backtest, not just in theory."""
    age = panel.daily["competitor_age_days"]
    assert (age > 0).mean() > 0.2
    assert panel.daily["competitor_price"].notna().all()


def test_purchase_panel_has_time_varying_price_covariate(panel):
    purchases = panel.purchases
    assert {"customer_id", "date", "price_ratio", "gap_days", "repurchased"} <= set(
        purchases.columns
    )
    assert purchases["price_ratio"].std() > 0
    assert purchases["gap_days"].min() >= 0


def test_higher_prices_lengthen_the_gap_to_the_next_purchase(panel):
    """theta < 0 in the generator; the data must actually show that direction."""
    purchases = panel.purchases[panel.purchases["repurchased"]]
    expensive = purchases[purchases["price_ratio"] > purchases["price_ratio"].median()]
    cheap = purchases[purchases["price_ratio"] <= purchases["price_ratio"].median()]
    assert expensive["gap_days"].mean() > cheap["gap_days"].mean()


def test_customers_are_never_a_pricing_input(panel):
    """Customer identity exists for retention modelling and never reaches a price."""
    assert "customer_id" not in panel.daily.columns


# ---------------------------------------------------------------------------
# Ground truth surface
# ---------------------------------------------------------------------------


def test_truth_frame_joins_to_estimator_output(panel):
    frame = panel.truth.as_frame()
    assert set(frame["sku"]) == set(panel.skus)
    assert (frame["true_beta"] < 0).all()


def test_beta_lookup_matches_the_array(panel):
    for index, sku in enumerate(panel.skus):
        assert panel.truth.beta_for(sku) == pytest.approx(panel.truth.beta[index])


# ---------------------------------------------------------------------------
# Round trip through the L0 contract
# ---------------------------------------------------------------------------


def test_generated_transactions_satisfy_the_transaction_contract(panel):
    transactions = transactions_from_panel(panel)
    as_of = transactions["invoice_ts"].max().to_pydatetime()
    report = validate(transactions, TRANSACTIONS_CONTRACT, as_of=as_of)
    assert report.passed, report.summary()
