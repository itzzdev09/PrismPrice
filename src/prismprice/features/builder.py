"""
Point-in-time feature assembly (L1).

**The contract:** given ``(sku, as_of)``, return features computed using only
rows with ``date < as_of``. Strictly less-than, never less-than-or-equal — a
decision made on the morning of day T cannot see day T's own sales.

This is the layer where leakage happens, and leakage does not announce itself:
it shows up as a backtest that looks excellent and a live model that does not.
The guarantee is therefore tested directly — build features at T, append the
next 30 days of data, rebuild, and assert the values are byte-identical.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

__all__ = [
    "FeatureBuilder",
    "FeatureVector",
    "InsufficientHistoryError",
]


class InsufficientHistoryError(LookupError):
    """Raised when a SKU has no history before ``as_of``.

    A distinct exception because the caller's response is a different code path
    — the cold-start embedding prior (see :mod:`prismprice.features.embeddings`)
    — not a retry or a default value.
    """

    def __init__(self, sku: str, as_of: datetime) -> None:
        super().__init__(f"No observations for {sku!r} before {as_of.isoformat()}")
        self.sku = sku
        self.as_of = as_of


class FeatureVector(BaseModel):
    """Features for one ``(sku, as_of)``, with the provenance to trust them."""

    model_config = ConfigDict(frozen=True)

    sku: str
    as_of: datetime

    # -- demand history ---------------------------------------------------
    units_7d: float
    units_28d: float
    units_91d: float
    demand_trend: float
    """Ratio of 7-day to 28-day mean demand. > 1 is accelerating."""
    demand_volatility: float
    """Coefficient of variation of daily units over the trailing 28 days."""
    zero_sales_streak: int

    # -- price history ----------------------------------------------------
    current_price: float
    reference_price_60d: float
    """Trailing 60-day median. The anchor customers judge a price against."""
    discount_depth: float
    min_price_last_30d: float
    """EU Omnibus anchor for PP-G003. Strictly trailing."""
    price_changes_28d: int

    # -- seasonality ------------------------------------------------------
    week_of_year: int
    day_of_week: int
    is_weekend: float
    holiday_proximity: float

    # -- inventory --------------------------------------------------------
    inventory_on_hand: float
    inventory_cover_days: float
    inventory_shadow_price: float

    # -- competition ------------------------------------------------------
    competitor_price: float | None
    competitor_gap: float | None
    """``price / competitor_price - 1``. Positive means we are more expensive."""
    competitor_age_days: float | None

    # -- provenance -------------------------------------------------------
    observations_used: int
    days_since_last_observation: float
    is_stale: bool
    """True when the newest observation is older than the staleness SLA. The
    decision path degrades on this rather than pretending the data is current."""

    def to_series(self) -> pd.Series:
        """Numeric view for model input; identity and provenance stripped."""
        data = self.model_dump()
        for key in ("sku", "as_of", "is_stale"):
            data.pop(key)
        return pd.Series(data, dtype="float64")


@dataclass
class FeatureBuilder:
    """Assembles point-in-time features from a daily panel.

    Args:
        daily: Panel with at least ``sku``, ``date``, ``price``, ``units``.
        reference_window_days: Window for the reference (anchor) price.
        omnibus_window_days: Window for the PP-G003 trailing minimum.
        staleness_sla_days: Age beyond which the vector is flagged stale.
        demand_column: Which demand series to build from. Point to the
            un-censored estimate once :mod:`~prismprice.features.uncensoring`
            has run, so trailing demand reflects demand rather than supply.
    """

    daily: pd.DataFrame
    reference_window_days: int = 60
    omnibus_window_days: int = 30
    staleness_sla_days: float = 2.0
    demand_column: str = "units"

    def __post_init__(self) -> None:
        required = {"sku", "date", "price", self.demand_column}
        missing = required - set(self.daily.columns)
        if missing:
            raise ValueError(f"Daily panel is missing required columns: {sorted(missing)}")

        frame = self.daily.copy()
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        frame = frame.sort_values(["sku", "date"]).reset_index(drop=True)
        self._by_sku: dict[str, pd.DataFrame] = {
            str(sku): group.reset_index(drop=True) for sku, group in frame.groupby("sku")
        }

    # -- helpers ----------------------------------------------------------

    def _history(self, sku: str, as_of: datetime) -> pd.DataFrame:
        group = self._by_sku.get(sku)
        if group is None:
            raise InsufficientHistoryError(sku, as_of)
        cutoff = pd.Timestamp(as_of)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        # Strictly before as_of. This single comparison is the leakage boundary.
        history = group[group["date"] < cutoff]
        if history.empty:
            raise InsufficientHistoryError(sku, as_of)
        return history

    @staticmethod
    def _tail_mean(series: pd.Series, days: int) -> float:
        tail = series.tail(days)
        return float(tail.mean()) if len(tail) else 0.0

    @staticmethod
    def _last(history: pd.DataFrame, column: str, default: float | None = None) -> float | None:
        if column not in history.columns:
            return default
        series = history[column].dropna()
        return float(series.iloc[-1]) if len(series) else default

    @staticmethod
    def _zero_streak(units: Sequence[float]) -> int:
        streak = 0
        for value in reversed(list(units)):
            if value > 0:
                break
            streak += 1
        return streak

    # -- build ------------------------------------------------------------

    def build(self, sku: str, as_of: datetime) -> FeatureVector:
        """Features for one SKU as of one moment, using only prior data."""
        history = self._history(sku, as_of)
        units = history[self.demand_column].astype(float)
        prices = history["price"].astype(float)

        cutoff = pd.Timestamp(as_of)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")

        mean_7 = self._tail_mean(units, 7)
        mean_28 = self._tail_mean(units, 28)
        volatility_window = units.tail(28)
        volatility = (
            float(volatility_window.std() / mean_28)
            if mean_28 > 0 and len(volatility_window) > 1
            else 0.0
        )

        current_price = float(prices.iloc[-1])
        reference = float(prices.tail(self.reference_window_days).median())
        omnibus_min = float(prices.tail(self.omnibus_window_days).min())

        recent_prices = prices.tail(28)
        price_changes = int((recent_prices.diff().abs() > 1e-9).sum())

        competitor_price = self._last(history, "competitor_price")
        competitor_gap = (
            current_price / competitor_price - 1.0
            if competitor_price not in (None, 0.0) and competitor_price is not None
            else None
        )

        last_date = history["date"].iloc[-1]
        age_days = float((cutoff - last_date).total_seconds() / 86_400.0)
        day_of_year = int(cutoff.dayofyear)

        return FeatureVector(
            sku=sku,
            as_of=cutoff.to_pydatetime(),
            units_7d=mean_7,
            units_28d=mean_28,
            units_91d=self._tail_mean(units, 91),
            demand_trend=float(mean_7 / mean_28) if mean_28 > 0 else 0.0,
            demand_volatility=volatility,
            zero_sales_streak=self._zero_streak(units.tail(28).tolist()),
            current_price=current_price,
            reference_price_60d=reference,
            discount_depth=float(1.0 - current_price / reference) if reference > 0 else 0.0,
            min_price_last_30d=omnibus_min,
            price_changes_28d=price_changes,
            week_of_year=int(cutoff.isocalendar().week),
            day_of_week=int(cutoff.dayofweek),
            is_weekend=float(cutoff.dayofweek >= 5),
            holiday_proximity=float(
                np.exp(-min(abs(day_of_year - 359), 365 - abs(day_of_year - 359)) / 14.0)
            ),
            inventory_on_hand=self._last(history, "inventory_on_hand", 0.0) or 0.0,
            inventory_cover_days=self._last(history, "inventory_cover_days", 0.0) or 0.0,
            inventory_shadow_price=self._last(history, "inventory_shadow_price", 0.0) or 0.0,
            competitor_price=competitor_price,
            competitor_gap=competitor_gap,
            competitor_age_days=self._last(history, "competitor_age_days"),
            observations_used=len(history),
            days_since_last_observation=age_days,
            is_stale=age_days > self.staleness_sla_days,
        )

    def build_many(self, pairs: Iterable[tuple[str, datetime]]) -> list[FeatureVector]:
        """Build several vectors, skipping SKUs with no usable history."""
        vectors: list[FeatureVector] = []
        for sku, as_of in pairs:
            try:
                vectors.append(self.build(sku, as_of))
            except InsufficientHistoryError:
                continue
        return vectors

    def build_matrix(self, pairs: Iterable[tuple[str, datetime]]) -> pd.DataFrame:
        """Numeric design matrix indexed by ``(sku, as_of)``, ready for a model."""
        vectors = self.build_many(pairs)
        if not vectors:
            return pd.DataFrame()
        matrix = pd.DataFrame([v.to_series() for v in vectors])
        matrix.index = pd.MultiIndex.from_tuples(
            [(v.sku, v.as_of) for v in vectors], names=["sku", "as_of"]
        )
        return matrix

    def training_frame(self, sku: str, min_history_days: int = 91, stride: int = 1) -> pd.DataFrame:
        """Rolling-origin frame for one SKU: features at T, outcome at T.

        The outcome column is the realised demand on the decision date itself,
        which the feature vector by construction could not see. This is the only
        place the two are joined, and it is deliberately one function so there is
        one place to audit for leakage.
        """
        group = self._by_sku.get(sku)
        if group is None:
            raise InsufficientHistoryError(sku, datetime.now())

        rows: list[pd.Series] = []
        for position in range(min_history_days, len(group), stride):
            as_of = group["date"].iloc[position].to_pydatetime()
            try:
                vector = self.build(sku, as_of)
            except InsufficientHistoryError:
                continue
            series = vector.to_series()
            series["target_units"] = float(group[self.demand_column].iloc[position])
            series["target_price"] = float(group["price"].iloc[position])
            series.name = (sku, as_of)
            rows.append(series)

        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows)
        frame.index = pd.MultiIndex.from_tuples(list(frame.index), names=["sku", "as_of"])
        return frame
