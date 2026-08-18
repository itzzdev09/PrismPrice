"""
UCI Online Retail augmentation (L0).

The public UCI Online Retail dataset has real transaction logs and no cost,
inventory or competitor columns. Those are *assumed* here, not measured, and the
assumptions travel with the data as an :class:`AssumptionSet` rather than being
buried in a notebook. Any margin figure computed downstream inherits them.

Augmentation is deterministic given a seed, so a decision made on augmented data
is still reconstructible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from prismprice import config

__all__ = [
    "AssumptionSet",
    "augment_competitor",
    "augment_costs",
    "augment_inventory",
    "augment_omnibus_anchor",
    "build_augmented_panel",
    "normalise_uci",
    "to_daily_demand",
]


@dataclass(frozen=True)
class AssumptionSet:
    """Every number the augmentation invented, and how.

    Attached to the augmented panel so a reviewer can see which columns are
    observation and which are assumption without reading the code.
    """

    seed: int
    margin_range: tuple[float, float] = config.SYNTHETIC_MARGIN_RANGE
    competitor_mean_ratio: float = 1.02
    competitor_sd: float = 0.04
    competitor_observation_rate: float = 0.6
    replenishment_days: int = 14
    cover_target_days: float = 12.0
    shadow_price_kappa: float = 1.5
    min_cover_days: float = config.DEFAULT_MIN_INVENTORY_COVER_DAYS

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "margin_range": list(self.margin_range),
            "competitor_mean_ratio": self.competitor_mean_ratio,
            "competitor_sd": self.competitor_sd,
            "competitor_observation_rate": self.competitor_observation_rate,
            "replenishment_days": self.replenishment_days,
            "cover_target_days": self.cover_target_days,
            "shadow_price_kappa": self.shadow_price_kappa,
            "min_cover_days": self.min_cover_days,
        }

    def describe(self) -> str:
        return (
            "Augmented columns are ASSUMED, not observed:\n"
            f"  unit_cost               = mean historical price x (1 - m), "
            f"m ~ U{self.margin_range} fixed per SKU\n"
            f"  competitor_price        = price x N({self.competitor_mean_ratio}, "
            f"{self.competitor_sd}), observed {self.competitor_observation_rate:.0%} of days\n"
            f"  inventory_on_hand       = periodic review every {self.replenishment_days}d, "
            f"target {self.cover_target_days}d cover\n"
            f"  inventory_shadow_price  = exponential in cover shortfall below "
            f"{self.min_cover_days}d (kappa={self.shadow_price_kappa})\n"
            f"  seed                    = {self.seed}"
        )


#: Column aliases across the two UCI releases. The 2011 "Online Retail" set and
#: the 2009-2011 "Online Retail II" set carry the same fields under different
#: names (``InvoiceNo``/``Invoice``, ``UnitPrice``/``Price``,
#: ``CustomerID``/``Customer ID``). Code that handles only one fails silently on
#: the other — the rename is a no-op and the missing-column error names a field
#: the user can see in their file.
_UCI_ALIASES: dict[str, tuple[str, ...]] = {
    "invoice_id": ("Invoice", "InvoiceNo"),
    "sku": ("StockCode",),
    "description": ("Description",),
    "quantity": ("Quantity",),
    "invoice_ts": ("InvoiceDate",),
    "unit_price": ("Price", "UnitPrice"),
    "customer_id": ("Customer ID", "CustomerID"),
    "country": ("Country",),
}

#: Fields without which a transaction cannot be validated at all.
_UCI_REQUIRED = ("invoice_id", "sku", "quantity", "invoice_ts", "unit_price")


def _as_identifier(values: pd.Series) -> pd.Series:
    """Render an id column as a string without inventing a decimal point.

    Identifiers routinely arrive as floats because one null in the column forces
    pandas to float64 on read. ``astype(str)`` then turns 13085 into "13085.0",
    which joins to nothing. Whole-valued numbers are rendered as integers and
    nulls stay null; anything genuinely non-numeric is left as-is.
    """
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        return numeric.astype("Int64").astype("string")
    return values.astype("string")


def normalise_uci(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename and type a raw UCI extract to match the transaction contract.

    Accepts either UCI release; see :data:`_UCI_ALIASES`.

    Guest checkouts keep a null ``customer_id``. Collapsing them to a sentinel
    would create one synthetic mega-customer and corrupt every retention curve
    downstream, so they stay null and the retention layer excludes them.
    """
    resolved: dict[str, str] = {}
    for canonical, aliases in _UCI_ALIASES.items():
        for alias in aliases:
            if alias in raw.columns:
                resolved[alias] = canonical
                break

    missing = [c for c in _UCI_REQUIRED if c not in resolved.values()]
    if missing:
        raise ValueError(
            f"Raw UCI extract is missing required columns {missing}. "
            f"Saw columns: {sorted(map(str, raw.columns))}"
        )

    df = raw.rename(columns=resolved).copy()
    # Invoice ids arrive mixed: ordinary invoices parse as integers and
    # cancellations do not ("C489449"), so the column is object dtype and a
    # plain astype leaves the numbers rendered as ints and the cancellations as
    # strings in the same column.
    df["invoice_id"] = _as_identifier(df["invoice_id"])
    df["sku"] = _as_identifier(df["sku"])
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").astype("Int64")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").astype(float)
    df["invoice_ts"] = pd.to_datetime(df["invoice_ts"], errors="coerce", utc=True)

    # Customer ids are stored as floats in the UCI workbooks, so a naive cast
    # produces "13085.0". That is not merely untidy: it will not join to any
    # other rendering of the same id, and a retention layer keyed on it would
    # silently match nothing.
    df["customer_id"] = _as_identifier(df["customer_id"]) if "customer_id" in df.columns else pd.NA
    for optional in ("description", "country"):
        df[optional] = df[optional].astype("string") if optional in df.columns else pd.NA

    df["line_number"] = df.groupby("invoice_id").cumcount().astype("int64")
    df["is_return"] = df["invoice_id"].str.startswith("C", na=False)
    return df


def to_daily_demand(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction lines to a (sku, date) panel of units and mean price.

    Returns are netted off rather than dropped: a return is real information
    about demand, and discarding it inflates the series.
    """
    tx = transactions.dropna(subset=["invoice_ts", "sku", "quantity", "unit_price"])
    tx = tx[tx["unit_price"] > 0]

    tx = tx.assign(date=tx["invoice_ts"].dt.normalize())
    tx = tx.assign(revenue=tx["quantity"].astype(float) * tx["unit_price"])

    grouped = tx.groupby(["sku", "date"], as_index=False).agg(
        units=("quantity", lambda s: float(s.astype(float).sum())),
        revenue=("revenue", "sum"),
        transactions=("invoice_id", "nunique"),
    )
    grouped["units"] = grouped["units"].clip(lower=0.0)
    grouped["price"] = np.where(
        grouped["units"] > 0, grouped["revenue"] / grouped["units"].replace(0, np.nan), np.nan
    )
    grouped["price"] = grouped.groupby("sku")["price"].transform(lambda s: s.ffill().bfill())
    return grouped.dropna(subset=["price"]).reset_index(drop=True)


def augment_costs(daily: pd.DataFrame, assumptions: AssumptionSet) -> pd.DataFrame:
    """Add ``unit_cost`` from a per-SKU category margin target.

    ``c_i = mean(p_i) x (1 - m)`` with ``m`` fixed per SKU. Cost is therefore
    constant over time, which is wrong in reality — landed costs move — but a
    stable wrong assumption is auditable and a drifting one is not.
    """
    rng = np.random.default_rng(assumptions.seed)
    skus = daily["sku"].drop_duplicates().sort_values().to_numpy()
    low, high = assumptions.margin_range
    margins = pd.Series(rng.uniform(low, high, len(skus)), index=skus, name="margin")

    mean_price = daily.groupby("sku")["price"].mean()
    unit_cost = (mean_price * (1.0 - margins)).rename("unit_cost")

    out = daily.merge(unit_cost, left_on="sku", right_index=True, how="left")
    out["assumed_margin"] = out["sku"].map(margins)
    return out


def augment_inventory(daily: pd.DataFrame, assumptions: AssumptionSet) -> pd.DataFrame:
    """Add inventory on hand, days of cover, stockout flag and shadow price."""
    rng = np.random.default_rng(assumptions.seed + 1)
    frames: list[pd.DataFrame] = []

    for _sku, group in daily.groupby("sku", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        demand = group["units"].to_numpy(dtype=float)
        n = len(demand)

        on_hand = np.zeros(n)
        window = min(assumptions.replenishment_days, n)
        stock = float(demand[:window].mean() * assumptions.cover_target_days)

        for t in range(n):
            if t > 0 and t % assumptions.replenishment_days == 0:
                # Order up to a target level, not by a fixed quantity: a fixed
                # quantity has no feedback term and drifts without bound.
                recent = demand[max(0, t - assumptions.replenishment_days) : t]
                target_level = float(recent.mean()) * assumptions.cover_target_days
                stock = max(stock, target_level * float(rng.uniform(0.75, 1.05)))
            on_hand[t] = stock
            stock = max(stock - demand[t], 0.0)

        average_daily = np.maximum(
            pd.Series(demand).rolling(28, min_periods=1).mean().to_numpy(), 1e-6
        )
        cover = on_hand / average_daily
        shortfall = np.clip(
            (assumptions.min_cover_days - cover) / assumptions.min_cover_days, 0.0, 1.0
        )
        shadow = np.where(
            cover < assumptions.min_cover_days,
            group["unit_cost"].to_numpy()
            * (np.exp(assumptions.shadow_price_kappa * shortfall) - 1.0),
            0.0,
        )

        group["inventory_on_hand"] = on_hand
        group["inventory_cover_days"] = cover
        group["inventory_shadow_price"] = shadow
        group["stockout"] = on_hand <= 0.0
        frames.append(group)

    return pd.concat(frames, ignore_index=True)


def augment_competitor(daily: pd.DataFrame, assumptions: AssumptionSet) -> pd.DataFrame:
    """Add a competitor price series with realistic scraping staleness.

    Observations arrive on only a fraction of days and are carried forward, so
    ``competitor_age_days`` is non-zero most of the time — which is what makes
    degradation rung 2 reachable in a backtest rather than theoretical.
    """
    rng = np.random.default_rng(assumptions.seed + 2)
    frames: list[pd.DataFrame] = []

    for _sku, group in daily.groupby("sku", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        n = len(group)
        ratio = np.clip(
            rng.normal(assumptions.competitor_mean_ratio, assumptions.competitor_sd, n), 0.7, 1.5
        )
        true_price = group["price"].to_numpy() * ratio
        seen = rng.random(n) < assumptions.competitor_observation_rate
        seen[0] = True

        observed = pd.Series(np.where(seen, true_price, np.nan)).ffill().to_numpy()
        age = np.zeros(n)
        counter = 0.0
        for t in range(n):
            counter = 0.0 if seen[t] else counter + 1.0
            age[t] = counter

        group["competitor_price"] = observed
        group["competitor_age_days"] = age
        frames.append(group)

    return pd.concat(frames, ignore_index=True)


def augment_omnibus_anchor(daily: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    """Add the trailing 30-day minimum price required by ``PP-G003``.

    Strictly trailing and strictly exclusive of the current day: including today
    would let a price validate itself against itself, which is exactly the
    circularity the Omnibus Directive exists to prevent.
    """
    frames: list[pd.DataFrame] = []
    for _sku, group in daily.groupby("sku", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        group["min_price_last_30d"] = (
            group["price"].shift(1).rolling(window_days, min_periods=1).min()
        )
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def build_augmented_panel(
    raw_uci: pd.DataFrame, seed: int = config.DEFAULT_SEED
) -> tuple[pd.DataFrame, AssumptionSet]:
    """Full pipeline: raw UCI extract to a decision-ready daily panel.

    Returns the panel and the assumption set that produced its invented columns.
    """
    assumptions = AssumptionSet(seed=seed)
    transactions = normalise_uci(raw_uci)
    daily = to_daily_demand(transactions[~transactions["is_return"]])
    daily = augment_costs(daily, assumptions)
    daily = augment_inventory(daily, assumptions)
    daily = augment_competitor(daily, assumptions)
    daily = augment_omnibus_anchor(daily)
    return daily.sort_values(["sku", "date"]).reset_index(drop=True), assumptions
