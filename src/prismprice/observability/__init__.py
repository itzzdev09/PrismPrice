"""Observability layer (L7): drift detection, KPIs and circuit breakers."""

from prismprice.observability.alerts import (
    BreakerResult,
    BreakerStatus,
    aggregate_movement,
    evaluate_breakers,
    no_guardrail_violations,
    one_sided_movement,
)
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
    "BreakerResult",
    "BreakerStatus",
    "DriftReport",
    "KPIResult",
    "aggregate_movement",
    "categorical_drift",
    "contribution_profit_per_customer",
    "degradation_distribution",
    "evaluate_breakers",
    "feature_drift",
    "gross_margin_pct",
    "guardrail_bind_rate",
    "no_guardrail_violations",
    "one_sided_movement",
    "population_stability_index",
    "price_adherence",
    "recommendation_churn",
]
