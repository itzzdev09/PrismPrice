"""
KPI computations (L7).

Implements the metrics defined in ``docs/metrics.md``. Every one has a
threshold and a stated action there; a metric with no action attached is
decoration and does not belong on a dashboard.

Two of these deserve their reasoning restated in code, because both are easy to
replace with a more familiar number that is worse.

**CPPC — contribution profit per customer, 90-day rolling.** The headline, and
the only single number that cannot be gamed by winning one half of the
trade-off. Discount-led volume lifts the denominator and depresses the
numerator; margin extraction lifts per-transaction profit and shrinks the
customer count as defectors leave. Both failures show up here and neither shows
up in gross margin percentage — which is why gross margin is on the dashboard as
a *diagnostic* and this is the metric that matters.

**Price adherence — the share of recommendations published unmodified.** The
honest health metric for a decision-support system. A system producing perfect
recommendations that nobody publishes has failed, and that failure is invisible
in every profit metric: the prices were never taken, so the profit never moved,
so nothing looks wrong. It is a worse outcome than the service being down, and
only this number shows it.

The guardrail bind rate is a policy diagnostic rather than an error rate. A code
binding on most candidates means the *constraint* is setting the price, not the
model. That can be entirely correct — a margin floor should bind in a low-margin
category — but it has to be a decision somebody made rather than a fact nobody
noticed. ``PP-G008`` is excluded from the alert by default because ladder
compliance binding on nearly every candidate is the expected state, and an alert
that always fires is an alert nobody reads.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "KPIResult",
    "contribution_profit_per_customer",
    "degradation_distribution",
    "gross_margin_pct",
    "guardrail_bind_rate",
    "price_adherence",
    "recommendation_churn",
]


@dataclass(frozen=True)
class KPIResult:
    """One metric, its threshold, and whether it has been breached.

    ``breached`` is computed here rather than by a dashboard, so the definition
    of "bad" lives next to the definition of the number and the two cannot
    drift apart.
    """

    name: str
    value: float
    threshold: float | None = None
    breached: bool = False
    action: str = ""
    detail: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "threshold": self.threshold,
            "breached": self.breached,
            "action": self.action,
            "detail": self.detail or {},
        }


def contribution_profit_per_customer(
    transactions: pd.DataFrame,
    as_of: datetime,
    window_days: int = 90,
    quantity_column: str = "units",
    price_column: str = "price",
    cost_column: str = "unit_cost",
    customer_column: str = "customer_id",
    date_column: str = "date",
) -> KPIResult:
    """CPPC over a rolling window. The headline metric (metrics.md §1).

    ``sum(q * (p - c)) / distinct customers transacting in the window``.

    Ninety days because the repurchase cycle for the target categories sits
    inside it: shorter and price-shock churn has not yet expressed itself,
    longer and the signal lags the decision that caused it.

    Guest checkouts (null customer id) contribute to the numerator and not to
    the denominator. That is deliberate and it is a real limitation, not an
    oversight: the profit was earned and must be counted, but the customer
    cannot be identified and counting them as one shared customer would make the
    ratio meaningless. The guest share is reported so the distortion is visible.

    Raises:
        ValueError: on missing columns or a non-positive window.
    """
    required = {quantity_column, price_column, cost_column, customer_column, date_column}
    missing = sorted(required - set(transactions.columns))
    if missing:
        raise ValueError(f"transactions frame is missing columns: {missing}")
    if window_days < 1:
        raise ValueError(f"window_days must be >= 1, got {window_days}")

    dates = pd.to_datetime(transactions[date_column], utc=True)
    # `pd.Timestamp(dt, tz=...)` raises when dt is already tz-aware, which every
    # datetime in this codebase is. Localise or convert depending on what
    # arrived, rather than assuming naive input the callers never send.
    stamp = pd.Timestamp(as_of)
    stamp = stamp.tz_localize("UTC") if stamp.tz is None else stamp.tz_convert("UTC")
    cutoff = stamp - timedelta(days=window_days)
    window = transactions[(dates > cutoff) & (dates <= stamp)]

    if window.empty:
        return KPIResult(
            name="cppc_90d",
            value=float("nan"),
            action="No transactions in the window; check ingestion before reading this as zero.",
            detail={"window_days": window_days, "n_transactions": 0},
        )

    contribution = float(
        (
            window[quantity_column].astype(float)
            * (window[price_column].astype(float) - window[cost_column].astype(float))
        ).sum()
    )
    identified = window[customer_column].dropna()
    n_customers = int(identified.nunique())
    guest_share = float(window[customer_column].isna().mean())

    value = contribution / n_customers if n_customers else float("nan")

    return KPIResult(
        name="cppc_90d",
        value=value,
        action="Declining three periods running: escalate. This is the metric that matters.",
        detail={
            "window_days": window_days,
            "contribution": contribution,
            "n_customers": n_customers,
            "guest_share": guest_share,
            "n_transactions": len(window),
        },
    )


def gross_margin_pct(
    transactions: pd.DataFrame,
    margin_floor_pct: float,
    quantity_column: str = "units",
    price_column: str = "price",
    cost_column: str = "unit_cost",
) -> KPIResult:
    """``sum(q(p-c)) / sum(qp)``, against the floor minus one point.

    On breach the stated action is to check the guardrail bind rate *before*
    blaming the model: a margin floor that is binding on most candidates is
    setting the price itself, and retuning the objective would not move it.
    """
    quantity = transactions[quantity_column].astype(float)
    price = transactions[price_column].astype(float)
    cost = transactions[cost_column].astype(float)

    revenue = float((quantity * price).sum())
    if revenue <= 0:
        return KPIResult(name="gross_margin_pct", value=float("nan"), threshold=margin_floor_pct)

    margin = float((quantity * (price - cost)).sum()) / revenue
    threshold = margin_floor_pct - 0.01

    return KPIResult(
        name="gross_margin_pct",
        value=margin,
        threshold=threshold,
        breached=margin < threshold,
        action="Check the guardrail bind rate before blaming the model.",
        detail={"revenue": revenue},
    )


def price_adherence(
    recommended: Sequence[float],
    published: Sequence[float],
    tolerance: float = 0.005,
    threshold: float = 0.70,
) -> KPIResult:
    """Share of recommendations published unmodified (metrics.md §2).

    The honest health metric for decision support. Low adherence means the
    system is producing numbers nobody uses — a worse outcome than being down,
    and invisible in every profit metric, because prices that were never taken
    cannot move profit.

    Args:
        recommended: What the system proposed.
        published: What actually went live, index-aligned.
        tolerance: Relative difference still counted as "unmodified", so a
            rounding difference is not scored as a rejection.

    Raises:
        ValueError: on mismatched lengths — a misaligned comparison would
            produce a plausible number from nonsense.
    """
    if len(recommended) != len(published):
        raise ValueError(
            f"recommended and published must align, got {len(recommended)} and {len(published)}"
        )
    if not recommended:
        raise ValueError("no recommendations to score")

    proposed = np.asarray(recommended, dtype=float)
    live = np.asarray(published, dtype=float)
    relative = np.abs(live - proposed) / np.maximum(np.abs(proposed), 1e-12)
    adherence = float(np.mean(relative <= tolerance))

    return KPIResult(
        name="price_adherence",
        value=adherence,
        threshold=threshold,
        breached=adherence < threshold,
        action="The system is not trusted — find out why before tuning it.",
        detail={"n": len(proposed), "n_modified": int(np.sum(relative > tolerance))},
    )


def guardrail_bind_rate(
    binding_codes: Sequence[Sequence[str]],
    threshold: float = 0.60,
    excluded: frozenset[str] = frozenset({"PP-G008"}),
) -> dict[str, KPIResult]:
    """Share of candidates each guardrail rejected, by reason code.

    A policy diagnostic, not an error rate. A code binding on most candidates
    means the constraint is setting the price rather than the model — which may
    be correct, and must be a decision somebody made.

    ``PP-G008`` (ladder compliance) is excluded from the alert by default:
    binding on nearly every candidate is its expected state, and an alert that
    always fires is an alert nobody reads. It is still reported.
    """
    if not binding_codes:
        raise ValueError("no candidate evaluations to score")

    total = len(binding_codes)
    counts: dict[str, int] = {}
    for codes in binding_codes:
        for code in set(codes):
            counts[code] = counts.get(code, 0) + 1

    results: dict[str, KPIResult] = {}
    for code, count in sorted(counts.items(), key=lambda item: -item[1]):
        rate = count / total
        results[code] = KPIResult(
            name=f"bind_rate_{code}",
            value=rate,
            threshold=threshold,
            breached=rate > threshold and code not in excluded,
            action=(
                "Expected for ladder compliance; reported, not alerted."
                if code in excluded
                else "This constraint is setting the price, not the model. Confirm that is intended."
            ),
            detail={"code": code, "n_binding": count, "n_candidates": total},
        )
    return results


def recommendation_churn(
    previous: dict[str, float],
    current: dict[str, float],
    prior: dict[str, float],
    threshold: float = 0.10,
) -> KPIResult:
    """Share of SKUs whose recommendation reversed direction (metrics.md §3).

    Catches oscillation. A system that moves a price up on Monday and down on
    Wednesday destroys trust even when each decision scored well in isolation —
    and each *did* score well, which is why this cannot be found by looking at
    decisions one at a time.

    Args:
        prior: Prices two periods ago.
        previous: Prices one period ago.
        current: Prices now.
    """
    shared = set(prior) & set(previous) & set(current)
    if not shared:
        raise ValueError("no SKUs present in all three periods")

    reversals = 0
    for sku in shared:
        first = previous[sku] - prior[sku]
        second = current[sku] - previous[sku]
        if first * second < 0:
            reversals += 1

    rate = reversals / len(shared)
    return KPIResult(
        name="recommendation_churn",
        value=rate,
        threshold=threshold,
        breached=rate > threshold,
        action="Oscillation destroys trust even when each decision scored well.",
        detail={"n_skus": len(shared), "n_reversals": reversals},
    )


def degradation_distribution(rungs: Sequence[int], threshold: float = 0.05) -> dict[str, KPIResult]:
    """Share of decisions at each degradation rung (metrics.md §5).

    The single most informative operational chart. A system that always answers
    hides its own degradation inside a healthy-looking success rate; the rung
    histogram is what makes "we returned a price" and "we returned a *good*
    price" different numbers.

    Alerts when rung 3 or worse exceeds ``threshold`` of decisions.
    """
    if not rungs:
        raise ValueError("no decisions to score")

    total = len(rungs)
    counts = {rung: sum(1 for r in rungs if r == rung) for rung in range(1, 6)}

    results = {
        f"rung_{rung}": KPIResult(
            name=f"rung_{rung}",
            value=count / total,
            detail={"n": count, "total": total},
        )
        for rung, count in counts.items()
    }

    degraded = sum(count for rung, count in counts.items() if rung >= 3) / total
    results["degraded_share"] = KPIResult(
        name="degraded_share",
        value=degraded,
        threshold=threshold,
        breached=degraded > threshold,
        action="Rung 3+ above threshold: the system is answering, but not well.",
        detail={"total": total},
    )
    return results
