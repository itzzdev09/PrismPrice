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
from prismprice.observability.evaluation import (
    ClassificationReport,
    ConfusionMatrix,
    DemandScore,
    binary_classification_report,
    concordance_index,
    elasticity_health,
    pinball_loss,
    quantile_coverage,
    score_demand,
    wape,
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
    "ClassificationReport",
    "ConfusionMatrix",
    "DemandScore",
    "DriftReport",
    "KPIResult",
    "aggregate_movement",
    "binary_classification_report",
    "categorical_drift",
    "concordance_index",
    "contribution_profit_per_customer",
    "degradation_distribution",
    "elasticity_health",
    "evaluate_breakers",
    "feature_drift",
    "gross_margin_pct",
    "guardrail_bind_rate",
    "no_guardrail_violations",
    "one_sided_movement",
    "pinball_loss",
    "population_stability_index",
    "price_adherence",
    "quantile_coverage",
    "recommendation_churn",
    "score_demand",
    "wape",
]
