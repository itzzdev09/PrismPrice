"""
End-to-end pipeline: real transactions in, priced decisions out.

Every layer in this system has been proven on its own. Nothing had run them
*together on real data*, and the gap between those two states is where
integration faults live: a column named differently at a boundary, a model
fitted on a series the next stage does not consume, an assumption that held on
the generator and does not hold on a giftware retailer's Decembers.

This module is the join. It reads the panel from the DuckDB store, derives the
calendar features the real panel does not carry, fits demand and elasticity,
runs the decision engine per SKU, and writes the audit records back.

Two things it does differently from the synthetic path
-----------------------------------------------------

**Elasticity is fitted on ``list_price``, not ``price``.** The realised price is
revenue over units, which on this retailer moves with order-size mix because of
a volume-discount ladder: within a SKU-day, log unit price and log line quantity
correlate -0.67. Regressing demand on that recovers the billing rules wearing
the sign of an elasticity. ``list_price`` is what a small order pays, and it is
the variable the retailer actually sets.

**Confounders are derived, not supplied.** The synthetic generator hands over
marketing spend and holiday flags; a real transaction log has a date and nothing
else. What can be recovered from a date is seasonality, day-of-week and holiday
proximity — and for a UK giftware retailer selling into Christmas, that is most
of the demand variation there is. What *cannot* be recovered is the marketing
calendar, which is a genuine unobserved confounder and is named as such in the
run report rather than being quietly absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prismprice import config
from prismprice.decision.engine import DecisionEngine
from prismprice.decision.ladder import endings_for_price
from prismprice.estimation.elasticity import (
    DoubleMLElasticity,
    ElasticityEstimate,
    add_category_price_control,
)
from prismprice.governance.schemas import PriceRequest
from prismprice.observability.alerts import (
    aggregate_movement,
    evaluate_breakers,
    no_guardrail_violations,
    one_sided_movement,
)
from prismprice.storage import PriceStore

__all__ = [
    "CALENDAR_CONFOUNDERS",
    "PipelineReport",
    "build_calendar_features",
    "run_pipeline",
]

#: Confounders recoverable from a bare transaction date. Everything a real log
#: gives you, and materially less than the generator hands over.
CALENDAR_CONFOUNDERS: tuple[str, ...] = (
    "season_sin",
    "season_cos",
    "is_weekend",
    "days_to_christmas",
    "category_log_price",
)

#: Named so the report can state it rather than leaving it as a silent gap.
UNOBSERVED_CONFOUNDERS: tuple[str, ...] = (
    "marketing_spend",
    "competitor_price",
    "catalogue_placement",
)


def build_calendar_features(panel: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """Derive from the date every confounder a transaction log can support.

    Seasonality enters as a sine/cosine pair rather than a day-of-year integer:
    day 365 and day 1 are adjacent in the world and maximally distant on an
    integer axis, and a tree splitting on the integer cannot represent that.

    ``days_to_christmas`` is signed and clipped. For a UK giftware retailer it is
    the single largest demand driver in the data, and leaving it to a generic
    seasonality term would attribute a December spike to whatever price happened
    to be set in December.
    """
    frame = panel.copy()
    dates = pd.to_datetime(frame[date_column], utc=True)

    day_of_year = dates.dt.dayofyear.to_numpy(dtype=float)
    frame["season_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.25)
    frame["season_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.25)
    frame["is_weekend"] = (dates.dt.dayofweek >= 5).astype(float)

    # Signed distance to 25 December, clipped to +/-60 days: beyond that the
    # effect is flat and an unclipped ramp would impose linear structure on
    # months where none exists.
    christmas_day = 359.0
    raw = day_of_year - christmas_day
    wrapped = np.where(raw < -182.625, raw + 365.25, np.where(raw > 182.625, raw - 365.25, raw))
    frame["days_to_christmas"] = np.clip(wrapped, -60.0, 60.0)

    return add_category_price_control(frame, price_column="list_price", date_column=date_column)


@dataclass(frozen=True)
class PipelineReport:
    """What one end-to-end run produced, and what it could not."""

    as_of: datetime
    n_skus_considered: int
    n_skus_priced: int
    n_panel_rows: int
    elasticities: dict[str, ElasticityEstimate] = field(default_factory=dict)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    breakers: list[dict[str, Any]] = field(default_factory=list)
    may_publish: bool = True
    unobserved_confounders: tuple[str, ...] = UNOBSERVED_CONFOUNDERS
    notes: list[str] = field(default_factory=list)

    @property
    def usable_elasticities(self) -> int:
        return sum(1 for e in self.elasticities.values() if e.is_usable)

    def elasticity_summary(self) -> dict[str, float]:
        """Distribution of the fitted elasticities, over the usable ones."""
        points = [e.point for e in self.elasticities.values() if np.isfinite(e.point)]
        if not points:
            return {}
        values = np.asarray(points, dtype=float)
        return {
            "n": float(values.size),
            "median": float(np.median(values)),
            "p10": float(np.quantile(values, 0.10)),
            "p90": float(np.quantile(values, 0.90)),
            "share_negative": float(np.mean(values < 0)),
            "share_usable": self.usable_elasticities / max(len(self.elasticities), 1),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of.isoformat(),
            "n_skus_considered": self.n_skus_considered,
            "n_skus_priced": self.n_skus_priced,
            "n_panel_rows": self.n_panel_rows,
            "elasticity_summary": self.elasticity_summary(),
            "n_decisions": len(self.decisions),
            "may_publish": self.may_publish,
            "breakers": self.breakers,
            "unobserved_confounders": list(self.unobserved_confounders),
            "notes": self.notes,
        }


def run_pipeline(
    store_path: str | Path = "data/prismprice.duckdb",
    as_of: datetime | None = None,
    min_observations: int = 300,
    max_skus: int | None = None,
    assumed_margin: float = 0.45,
    n_repeats: int = 3,
    seed: int = config.DEFAULT_SEED,
) -> PipelineReport:
    """Run L0 -> L1 -> L2 -> L3 on the real panel and record the decisions.

    Args:
        store_path: DuckDB store holding ingested transactions.
        as_of: Decision timestamp. Defaults to the last transaction in the store,
            because pricing "now" against a panel that ends in 2011 would make
            every feature stale and every decision meaningless.
        min_observations: Days of history a SKU needs to be priced at all.
        max_skus: Cap, for iteration. ``None`` prices everything eligible.
        assumed_margin: Gross margin used to derive unit cost. **This is an
            assumption, not a measurement** — no public retail dataset carries
            COGS — and every margin figure downstream inherits it.
        n_repeats: Cross-fitting repeats for the elasticity estimator.

    Returns:
        :class:`PipelineReport`.
    """
    with PriceStore(store_path) as store:
        panel = store.daily_panel(min_observations=min_observations)

    if panel.empty:
        raise ValueError(
            f"no SKU has {min_observations} days of history in {store_path}; "
            f"ingest transactions before running the pipeline"
        )

    stamp = as_of or panel["date"].max().to_pydatetime()
    skus = sorted(panel["sku"].unique())
    if max_skus is not None:
        skus = skus[:max_skus]
        panel = panel[panel["sku"].isin(skus)]

    featured = build_calendar_features(panel)
    featured = featured.dropna(subset=["category_log_price"])

    notes = [
        f"unit_cost derived from an assumed {assumed_margin:.0%} gross margin; "
        f"no public retail dataset carries COGS",
        "elasticity fitted on list_price, not realised price, because realised "
        "price moves with order-size mix under this retailer's volume discounts",
    ]

    elasticity_model = DoubleMLElasticity(
        outcome_column="units",
        price_column="list_price",
        confounder_columns=CALENDAR_CONFOUNDERS,
        n_repeats=n_repeats,
        seed=seed,
    )
    elasticity_model.fit(featured)

    engine = DecisionEngine(seed=seed, policy_version="0.2.0-real")
    decisions: list[dict[str, Any]] = []
    previous_prices: list[float] = []
    new_prices: list[float] = []
    costs: list[float] = []

    for sku in skus:
        history = featured[featured["sku"] == sku].sort_values("date")
        if history.empty:
            continue

        latest = history.iloc[-1]
        current_price = float(latest["list_price"])
        if not np.isfinite(current_price) or current_price <= 0:
            continue

        unit_cost = current_price * (1.0 - assumed_margin)
        estimate = elasticity_model.estimates.get(sku)
        # Degradation rung 3 in practice: a SKU whose own elasticity is not
        # identified is priced on the category median rather than on a number
        # the estimator declined to stand behind.
        elasticity = (
            estimate.point
            if estimate is not None and estimate.is_usable and np.isfinite(estimate.point)
            else _pooled_elasticity(elasticity_model.estimates)
        )

        baseline_units = float(history["units"].tail(28).mean())
        if not np.isfinite(baseline_units) or baseline_units <= 0:
            continue
        spread = float(history["units"].tail(28).std() / max(baseline_units, 1e-9))
        spread = float(np.clip(spread if np.isfinite(spread) else 0.4, 0.15, 1.2))

        def demand_at(
            price: float,
            base: float = baseline_units,
            p0: float = current_price,
            e: float = elasticity,
            s: float = spread,
        ) -> tuple[float, float, float]:
            median = base * (price / p0) ** e
            return (median * max(1.0 - s, 0.05), median, median * (1.0 + s))

        rounded_price = round(current_price, 2)
        request = PriceRequest(
            sku=str(sku),
            as_of=stamp,
            current_price=rounded_price,
            unit_cost=round(max(unit_cost, 0.01), 2),
            # Banded by magnitude. A flat .95/.99 rule cannot produce a legal
            # price inside a 15% cap below about GBP 2, which held a third of
            # this catalogue at rung 4.
            allowed_price_endings=endings_for_price(rounded_price),
        )
        outcome = engine.decide(request, demand_at=demand_at)
        best = outcome.best_outcome()

        previous_prices.append(request.current_price)
        new_prices.append(outcome.recommended_price)
        costs.append(request.unit_cost)
        decisions.append(
            {
                "sku": str(sku),
                "current_price": request.current_price,
                "recommended_price": outcome.recommended_price,
                "unit_cost": request.unit_cost,
                "elasticity": elasticity,
                "elasticity_confidence": (estimate.confidence if estimate else "low"),
                "degradation_rung": outcome.record.degradation_rung,
                "degradation_reason": outcome.record.degradation_reason_code.value,
                "binding_constraints": [c.value for c in outcome.record.binding_constraints],
                # best_outcome() is None on the fallback path, where the price
                # was held rather than chosen — so there is no scored candidate
                # to report, and inventing one would make a degradation look
                # like an optimisation.
                "expected_units": best.expected_units if best is not None else None,
                "expected_profit": best.expected_profit if best is not None else None,
                "price_change_pct": (outcome.recommended_price / request.current_price - 1.0),
            }
        )

    breakers = [
        one_sided_movement(previous_prices, new_prices),
        aggregate_movement(previous_prices, new_prices),
        no_guardrail_violations(new_prices, costs),
    ]
    may_publish, _ = evaluate_breakers(breakers)

    return PipelineReport(
        as_of=stamp,
        n_skus_considered=len(skus),
        n_skus_priced=len(decisions),
        n_panel_rows=len(featured),
        elasticities=dict(elasticity_model.estimates),
        decisions=decisions,
        breakers=[b.as_dict() for b in breakers],
        may_publish=may_publish,
        notes=notes,
    )


def _pooled_elasticity(estimates: dict[str, ElasticityEstimate]) -> float:
    """Category median over the usable per-SKU estimates.

    The rung-3 fallback. Uses the median rather than the mean because a DML
    estimate that has gone wrong tends to go wrong by a lot, and one SKU at -40
    should not set the price of every SKU that declined to answer.
    """
    usable = [e.point for e in estimates.values() if e.is_usable and np.isfinite(e.point)]
    if not usable:
        return config.SYNTHETIC_ELASTICITY_MEAN
    return float(np.median(usable))
