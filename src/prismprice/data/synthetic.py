"""
Synthetic panel generator with known ground truth (L0).

Implements the structural model in ``docs/data-and-modelling.md`` §1. Every
estimator in PrismPrice is proven here — against a parameter whose true value is
known — before it is allowed near real data. On real data a wrong elasticity and
a right one look identical; here they do not.

Two properties matter more than realism:

**Prices are confounded on purpose.** Promotions are scheduled when demand is
already high, so a naive regression of log-demand on log-price recovers a
biased elasticity. If it did not, Phase 3's DML would have nothing to prove.

**Demand is censored on purpose.** Observed units are capped by inventory, so
the naive series understates true demand during stockouts. That is the signal
Phase 1's un-censoring has to recover, and the truth is retained in
``units_uncensored`` so the recovery can be scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from prismprice import config

__all__ = [
    "GroundTruth",
    "SyntheticPanel",
    "generate_panel",
]


@dataclass(frozen=True)
class GroundTruth:
    """The parameters the generator used. Estimators are scored against these."""

    skus: tuple[str, ...]
    alpha: NDArray[np.float64]
    """Baseline log-demand per SKU, ``alpha_i ~ N(3.5, 0.5)``."""
    beta: NDArray[np.float64]
    """True causal own-price elasticity per SKU, ``beta_i ~ N(-1.8, 0.4)``, < 0."""
    eta: NDArray[np.float64]
    """``K x K`` cross-price elasticities; positive off-diagonal for substitutes."""
    gamma: NDArray[np.float64]
    """``K x n_confounders`` loadings on the observed confounders."""
    confounder_names: tuple[str, ...]
    theta: float
    """Repurchase-hazard sensitivity to the price shock ``(p / p_ref - 1)``.
    Negative: paying above reference lengthens the gap to the next purchase."""
    weibull_shape: float
    weibull_scale_days: float
    unit_cost: NDArray[np.float64]
    reference_price: NDArray[np.float64]
    seed: int

    def beta_for(self, sku: str) -> float:
        return float(self.beta[self.skus.index(sku)])

    def as_frame(self) -> pd.DataFrame:
        """Per-SKU truth, joinable to estimator output for scoring."""
        return pd.DataFrame(
            {
                "sku": list(self.skus),
                "true_alpha": self.alpha,
                "true_beta": self.beta,
                "unit_cost": self.unit_cost,
                "reference_price": self.reference_price,
            }
        )


@dataclass(frozen=True)
class SyntheticPanel:
    """Generated panel plus the truth that produced it."""

    daily: pd.DataFrame
    """One row per (sku, date). Carries both observed (censored) units and the
    latent uncensored demand, so un-censoring can be scored."""
    customers: pd.DataFrame
    purchases: pd.DataFrame
    """One row per repurchase event, with the price ratio paid and the gap to
    the next purchase — the survival panel for Phase 4."""
    truth: GroundTruth = field(repr=False)

    @property
    def skus(self) -> tuple[str, ...]:
        return self.truth.skus

    def daily_for(self, sku: str) -> pd.DataFrame:
        return self.daily[self.daily["sku"] == sku].reset_index(drop=True)


def _build_confounders(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Seasonality, marketing spend and holiday proximity.

    Marketing spend is autocorrelated rather than iid — campaigns run in bursts,
    and an iid confounder is trivially easy to control for, which would make the
    DML validation in Phase 3 meaningless.
    """
    day_of_year = dates.dayofyear.to_numpy()
    day_of_week = dates.dayofweek.to_numpy()

    yearly = np.sin(2 * np.pi * day_of_year / 365.25)
    weekly = (day_of_week >= 5).astype(float)

    spend = np.zeros(len(dates))
    shock = rng.normal(0.0, 1.0, len(dates))
    for t in range(1, len(dates)):
        spend[t] = 0.85 * spend[t - 1] + shock[t]
    spend = (spend - spend.mean()) / (spend.std() or 1.0)

    holiday = ((day_of_year > 330) | (day_of_year < 12)).astype(float)

    return pd.DataFrame(
        {
            "season_yearly": yearly,
            "is_weekend": weekly,
            "marketing_spend": spend,
            "holiday": holiday,
        },
        index=dates,
    )


def _build_prices(
    confounders: pd.DataFrame,
    base_price: NDArray[np.float64],
    rng: np.random.Generator,
    confounding_strength: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate the price path and promo flags.

    Promotion depth is driven by the same confounders that drive demand, which
    is what makes observational elasticity estimates wrong. ``confounding_strength``
    scales that dependence; at 0.0 the price is exogenous and even naive OLS
    recovers the truth, which is the control condition for the Phase 3 tests.
    """
    n_days = len(confounders)
    n_skus = len(base_price)

    demand_pressure = (
        0.6 * confounders["marketing_spend"].to_numpy()
        + 0.8 * confounders["holiday"].to_numpy()
        + 0.4 * confounders["season_yearly"].to_numpy()
    )

    prices = np.zeros((n_days, n_skus))
    promo = np.zeros((n_days, n_skus), dtype=bool)

    for j in range(n_skus):
        # Retailers discount into strength, not away from it.
        propensity = confounding_strength * demand_pressure + rng.normal(0.0, 0.7, n_days)
        on_promo = propensity > 1.0
        depth = np.where(on_promo, rng.uniform(0.10, 0.35, n_days), 0.0)
        # Independent price variation keeps the elasticity identified even at
        # high confounding strength — without it no estimator could succeed.
        jitter = rng.normal(0.0, 0.05, n_days)
        prices[:, j] = base_price[j] * (1.0 - depth) * np.exp(jitter)
        promo[:, j] = on_promo

    return prices, promo


def _simulate_inventory(
    latent_demand: NDArray[np.float64],
    rng: np.random.Generator,
    review_days: int,
    cover_target: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Periodic-review (R, S) replenishment, returning on-hand stock and served units.

    Order-*up-to*, not order-a-fixed-quantity: every ``review_days`` the order
    tops stock back up to ``S = trailing mean demand x cover_target``. A fixed
    quantity has no feedback term, so stock either runs away or collapses
    depending on the constant — it cannot hold a level.

    The random service multiplier makes some cycles under-order, so stockouts
    occur at a realistic rate. Served units are ``min(demand, on_hand)`` — the
    censoring that Phase 1's un-censoring has to undo.
    """
    n_days, n_skus = latent_demand.shape
    on_hand = np.zeros((n_days, n_skus))
    served = np.zeros((n_days, n_skus))

    stock = latent_demand[:review_days].mean(axis=0) * cover_target
    for t in range(n_days):
        if t > 0 and t % review_days == 0:
            recent = latent_demand[max(0, t - review_days) : t].mean(axis=0)
            target_level = recent * cover_target * rng.uniform(0.75, 1.05, n_skus)
            stock = np.maximum(stock, target_level)  # top up; never order negative
        on_hand[t] = stock
        served[t] = np.minimum(latent_demand[t], stock)
        stock = np.maximum(stock - served[t], 0.0)

    return on_hand, served


def _simulate_customers(
    dates: pd.DatetimeIndex,
    prices: NDArray[np.float64],
    reference_price: NDArray[np.float64],
    skus: tuple[str, ...],
    rng: np.random.Generator,
    n_customers: int,
    theta: float,
    weibull_shape: float,
    weibull_scale_days: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate repurchase events under a Weibull hazard with a price covariate.

    The gap to the next purchase depends on the price ratio *paid at the current
    purchase*, so the covariate is time-varying and the effect is causal by
    construction. ``theta < 0`` means paying above reference lengthens the gap.
    """
    n_days = len(dates)
    customer_ids = [f"CUST-{i:05d}" for i in range(n_customers)]

    frailty = rng.normal(0.0, 0.3, n_customers)
    first_day = rng.integers(0, max(n_days // 3, 1), n_customers)
    favourite = rng.integers(0, len(skus), n_customers)

    rows: list[dict[str, object]] = []
    for c in range(n_customers):
        day = int(first_day[c])
        order_index = 0
        while day < n_days:
            sku_idx = int(favourite[c])
            price = float(prices[day, sku_idx])
            ratio = price / float(reference_price[sku_idx])

            # Weibull inverse-CDF sampling with a proportional-hazards multiplier.
            multiplier = np.exp(theta * (ratio - 1.0) + frailty[c])
            u = rng.uniform(1e-9, 1.0)
            scale = weibull_scale_days / max(multiplier, 1e-6) ** (1.0 / weibull_shape)
            gap = float(scale * (-np.log(u)) ** (1.0 / weibull_shape))
            next_day = day + max(round(gap), 1)

            censored = next_day >= n_days
            rows.append(
                {
                    "customer_id": customer_ids[c],
                    "order_index": order_index,
                    "date": dates[day],
                    "sku": skus[sku_idx],
                    "price": price,
                    "price_ratio": ratio,
                    "gap_days": float(min(next_day, n_days - 1) - day),
                    "repurchased": not censored,
                }
            )
            order_index += 1
            day = next_day

    purchases = pd.DataFrame(rows)
    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "frailty": frailty,
            "first_purchase_date": [dates[int(d)] for d in first_day],
            "favourite_sku": [skus[int(i)] for i in favourite],
        }
    )
    return customers, purchases


def generate_panel(
    n_skus: int = 8,
    n_days: int = 540,
    n_customers: int = 600,
    start: datetime | None = None,
    seed: int = config.DEFAULT_SEED,
    confounding_strength: float = 1.0,
    cross_price_strength: float = 0.25,
    noise_sd: float = 0.18,
    review_days: int = 14,
    cover_target: float = 24.0,
) -> SyntheticPanel:
    """Generate a synthetic category panel with known structural parameters.

    Args:
        n_skus: SKUs in the category. Cross-price effects are within-category.
        n_days: Length of the daily panel.
        n_customers: Customers simulated for the retention layer.
        seed: Everything derives from this; the same seed gives the same panel.
        confounding_strength: How strongly promotions track demand drivers.
            ``0.0`` makes price exogenous — the control condition under which a
            naive estimator is expected to succeed.
        cross_price_strength: Scale of substitute cross-elasticities.
        noise_sd: SD of the unobserved log-demand shock.
        review_days: Replenishment review cycle.
        cover_target: Order-up-to level in days of cover. The default produces a
            realistic ~2% of censored days. Un-censoring tests pass a lower value
            (12-16) to force enough stockouts to score a recovery against.

    Returns:
        A :class:`SyntheticPanel` carrying observed data and the ground truth.
    """
    if n_skus < 1:
        raise ValueError("n_skus must be at least 1")
    if n_days < review_days * 2:
        raise ValueError(f"n_days must be at least {review_days * 2} for replenishment cycles")

    rng = np.random.default_rng(seed)
    start = start or datetime(2024, 1, 1, tzinfo=timezone.utc)
    dates = pd.date_range(start, periods=n_days, freq="D", tz="UTC")
    skus = tuple(f"SKU-{i:03d}" for i in range(n_skus))

    alpha = rng.normal(3.5, 0.5, n_skus)
    beta = np.clip(rng.normal(-1.8, 0.4, n_skus), None, -0.4)

    eta = rng.uniform(0.0, cross_price_strength, (n_skus, n_skus))
    np.fill_diagonal(eta, 0.0)

    confounders = _build_confounders(dates, rng)
    confounder_names = tuple(confounders.columns)
    gamma = rng.normal(0.25, 0.15, (n_skus, len(confounder_names)))

    base_price = rng.uniform(12.0, 60.0, n_skus)
    margin = rng.uniform(0.40, 0.55, n_skus)
    unit_cost = base_price * (1.0 - margin)

    prices, promo = _build_prices(confounders, base_price, rng, confounding_strength)
    log_prices = np.log(prices)

    epsilon = rng.normal(0.0, noise_sd, (n_days, n_skus))

    # log q_it = alpha_i + beta_i log p_it + sum_j eta_ij log p_jt + gamma_i' X_t + eps
    own = log_prices * beta
    cross = log_prices @ eta.T
    structural = confounders.to_numpy() @ gamma.T
    latent = np.exp(alpha + own + cross + structural + epsilon)

    on_hand, served = _simulate_inventory(latent, rng, review_days, cover_target)
    stockout = served < latent - 1e-9

    reference_price = np.median(prices, axis=0)

    competitor_noise = rng.normal(1.02, 0.04, (n_days, n_skus))
    competitor_price = prices * np.clip(competitor_noise, 0.7, 1.4)
    # Scraped feeds are stale: carry the last observation forward at random.
    observed_mask = rng.random((n_days, n_skus)) < 0.6
    # Seed day 0 so the forward-fill has something to carry; otherwise a SKU
    # whose first draw missed starts the panel with a NaN competitor price.
    observed_mask[0, :] = True
    competitor_observed = pd.DataFrame(np.where(observed_mask, competitor_price, np.nan)).ffill()
    competitor_age_days = np.zeros((n_days, n_skus))
    for j in range(n_skus):
        age = 0.0
        for t in range(n_days):
            age = 0.0 if observed_mask[t, j] else age + 1.0
            competitor_age_days[t, j] = age

    # Cover is stock divided by *trailing average* demand, not same-day demand.
    # Dividing by a single noisy day produces a cover series that swings wildly
    # and would make the inventory guard fire at random.
    trailing_demand = (
        pd.DataFrame(latent).rolling(28, min_periods=1).mean().to_numpy().clip(min=1e-6)
    )
    cover_days = on_hand / trailing_demand
    shadow = np.where(
        cover_days < config.DEFAULT_MIN_INVENTORY_COVER_DAYS,
        unit_cost
        * (
            np.exp(
                1.5
                * np.clip(
                    (config.DEFAULT_MIN_INVENTORY_COVER_DAYS - cover_days)
                    / config.DEFAULT_MIN_INVENTORY_COVER_DAYS,
                    0.0,
                    1.0,
                )
            )
            - 1.0
        ),
        0.0,
    )

    frames: list[pd.DataFrame] = []
    for j, sku in enumerate(skus):
        frame = pd.DataFrame(
            {
                "sku": sku,
                "date": dates,
                "price": prices[:, j],
                "units": served[:, j],
                "units_uncensored": latent[:, j],
                "inventory_on_hand": on_hand[:, j],
                "stockout": stockout[:, j],
                "inventory_cover_days": cover_days[:, j],
                "inventory_shadow_price": shadow[:, j],
                "unit_cost": unit_cost[j],
                "is_promo": promo[:, j],
                "competitor_price": competitor_observed.iloc[:, j].to_numpy(),
                "competitor_age_days": competitor_age_days[:, j],
            }
        )
        for name in confounder_names:
            frame[name] = confounders[name].to_numpy()
        frames.append(frame)

    daily = pd.concat(frames, ignore_index=True).sort_values(["sku", "date"]).reset_index(drop=True)
    daily["stockout"] = daily["stockout"].astype(bool)
    daily["is_promo"] = daily["is_promo"].astype(bool)

    customers, purchases = _simulate_customers(
        dates,
        prices,
        reference_price,
        skus,
        rng,
        n_customers,
        theta=-1.2,
        weibull_shape=1.4,
        weibull_scale_days=45.0,
    )

    truth = GroundTruth(
        skus=skus,
        alpha=alpha,
        beta=beta,
        eta=eta,
        gamma=gamma,
        confounder_names=confounder_names,
        theta=-1.2,
        weibull_shape=1.4,
        weibull_scale_days=45.0,
        unit_cost=unit_cost,
        reference_price=reference_price,
        seed=seed,
    )

    return SyntheticPanel(daily=daily, customers=customers, purchases=purchases, truth=truth)


def transactions_from_panel(panel: SyntheticPanel) -> pd.DataFrame:
    """Explode the daily panel into transaction lines matching the L0 contract.

    Useful for exercising :data:`prismprice.data.contracts.TRANSACTIONS_CONTRACT`
    end to end without a real extract.
    """
    rows = panel.daily[panel.daily["units"] > 0].reset_index(drop=True)
    return pd.DataFrame(
        {
            "invoice_id": [f"INV-{i:07d}" for i in range(len(rows))],
            "line_number": 0,
            "sku": rows["sku"].astype("string"),
            "quantity": rows["units"].round().astype("int64"),
            "unit_price": rows["price"].astype(float),
            "invoice_ts": rows["date"] + timedelta(hours=12),
            "customer_id": pd.Series([None] * len(rows), dtype="string"),
        }
    )
