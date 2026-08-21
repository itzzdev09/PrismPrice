"""Observability layer (L7): drift detection, KPIs and circuit breakers."""

from prismprice.observability.drift import (
    DriftReport,
    categorical_drift,
    feature_drift,
    population_stability_index,
)
from prismprice.observability.kpis import (
    KPIResult,
    contribution_profit_per_customer,
    degradation_distribution,
    gross_margin_pct,
    guardrail_bind_rate,
    price_adherence,
    recommendation_churn,
)

__all__ = [
    "DriftReport",
    "KPIResult",
    "categorical_drift",
    "contribution_profit_per_customer",
    "degradation_distribution",
    "feature_drift",
    "gross_margin_pct",
    "guardrail_bind_rate",
    "population_stability_index",
    "price_adherence",
    "recommendation_churn",
]
