"""
Model evaluation (L7) — how good is each model, measured the right way.

One report answering "how good is it", with a metric chosen to fit what each
model actually produces.

Why there is no confusion matrix for the pricing models
------------------------------------------------------

A confusion matrix counts true and false positives against a class label.
**Nothing in the pricing path emits a class label.** Demand is a conditional
quantile, elasticity is a continuous causal parameter, retention is a
time-to-event hazard, and the decision layer is a constrained optimiser. To
produce a confusion matrix from any of them you would have to invent a
threshold, binarise a continuous output against it, and then report accuracy on
a distinction the model was never asked to make. The number would be real and
the claim would be false — worse than reporting nothing, because it looks
rigorous.

So each model is scored on its own terms:

===================  ==========================================================
Demand               WAPE, pinball loss, and **interval coverage** — the one to
                     watch, because the objective samples the demand
                     distribution and dishonest p10/p90 make the CVaR term
                     decorative.
Elasticity           Sign-violation rate, CI width, and the share the estimator
                     declined to answer for. Against synthetic truth, coverage
                     and bias as well.
Retention            Concordance (Harrell's C) — the fraction of comparable
                     pairs the model ranks correctly. The right metric for
                     time-to-event, where a squared error on a censored
                     observation is meaningless.
Decision             Realised uplift against named baselines.
===================  ==========================================================

Where a confusion matrix *is* the right tool
--------------------------------------------

The circuit breakers and drift alarms **are** binary classifiers: each looks at
a batch and emits halt / do-not-halt. That is exactly the shape a confusion
matrix describes, and the asymmetry between the two error types is the whole
design question — a false negative publishes a bad batch and a false positive
delays a good one, and those costs are nothing like each other.
:func:`binary_classification_report` therefore reports precision, recall and
specificity separately rather than collapsing to accuracy, because on a rare
event accuracy is dominated by the majority class: a breaker that never fires
scores 99% on a catalogue where 1% of runs are bad.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "ClassificationReport",
    "ConfusionMatrix",
    "DemandScore",
    "binary_classification_report",
    "concordance_index",
    "pinball_loss",
    "quantile_coverage",
    "score_demand",
    "wape",
]


# ---------------------------------------------------------------------------
# Regression / quantile metrics
# ---------------------------------------------------------------------------


def wape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Weighted absolute percentage error: ``sum|q - qhat| / sum q``.

    Preferred to MAPE, which divides each error by its own actual and therefore
    explodes on the zero- and low-volume days that make up most of a long-tail
    catalogue. WAPE weights by volume, so it answers "how wrong were we about
    the units that mattered" rather than "how wrong were we about the quietest
    SKU in the file".
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if a.shape != p.shape:
        raise ValueError(f"actual and predicted must align, got {a.shape} and {p.shape}")
    denominator = float(np.sum(np.abs(a)))
    if denominator <= 0:
        return float("nan")
    return float(np.sum(np.abs(a - p)) / denominator)


def pinball_loss(actual: Sequence[float], predicted: Sequence[float], quantile: float) -> float:
    """Pinball (quantile) loss — the loss the demand model is actually fitted on.

    Scoring a quantile model with RMSE rewards it for predicting the mean, which
    is precisely the behaviour the quantile objective exists to avoid.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    delta = a - p
    return float(np.mean(np.maximum(quantile * delta, (quantile - 1.0) * delta)))


def quantile_coverage(
    actual: Sequence[float], lower: Sequence[float], upper: Sequence[float]
) -> dict[str, float]:
    """Empirical coverage of a prediction interval, and where the misses fall.

    Width and placement are reported separately because an interval can be the
    right size and in the wrong place. A nominal 80% interval covering 80% of
    outcomes with *all* the misses below p10 is not calibrated — it is shifted,
    and the CVaR term built on it is measuring the wrong tail.
    """
    a = np.asarray(actual, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if not (a.shape == lo.shape == hi.shape):
        raise ValueError("actual, lower and upper must align")

    below = float(np.mean(a < lo))
    above = float(np.mean(a > hi))
    return {
        "coverage": float(np.mean((a >= lo) & (a <= hi))),
        "below_lower": below,
        "above_upper": above,
        "placement_skew": below - above,
    }


@dataclass(frozen=True)
class DemandScore:
    """Demand model quality, on the terms the objective consumes it."""

    n: int
    wape: float
    pinball_p10: float
    pinball_p50: float
    pinball_p90: float
    coverage: float
    below_lower: float
    above_upper: float
    nominal_coverage: float = 0.80
    tolerance: float = 0.03

    @property
    def calibrated(self) -> bool:
        """Width within tolerance *and* misses roughly balanced.

        Both, because either alone can hold while the interval is wrong.
        """
        width_ok = abs(self.coverage - self.nominal_coverage) <= self.tolerance
        placement_ok = abs(self.below_lower - self.above_upper) <= self.tolerance * 2
        return width_ok and placement_ok

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "wape": self.wape,
            "pinball_p10": self.pinball_p10,
            "pinball_p50": self.pinball_p50,
            "pinball_p90": self.pinball_p90,
            "coverage": self.coverage,
            "nominal_coverage": self.nominal_coverage,
            "below_lower": self.below_lower,
            "above_upper": self.above_upper,
            "calibrated": self.calibrated,
        }


def score_demand(
    actual: Sequence[float],
    p10: Sequence[float],
    p50: Sequence[float],
    p90: Sequence[float],
) -> DemandScore:
    """Full demand-model scorecard from held-out predictions."""
    a = np.asarray(actual, dtype=float)
    if a.size == 0:
        raise ValueError("no observations to score")

    coverage = quantile_coverage(actual, p10, p90)
    return DemandScore(
        n=int(a.size),
        wape=wape(actual, p50),
        pinball_p10=pinball_loss(actual, p10, 0.10),
        pinball_p50=pinball_loss(actual, p50, 0.50),
        pinball_p90=pinball_loss(actual, p90, 0.90),
        coverage=coverage["coverage"],
        below_lower=coverage["below_lower"],
        above_upper=coverage["above_upper"],
    )


def concordance_index(
    durations: Sequence[float], events: Sequence[bool], risk_scores: Sequence[float]
) -> float:
    """Harrell's C: share of comparable pairs the risk score orders correctly.

    The right metric for a survival model. A squared error against an observed
    duration is meaningless when the observation is censored — the true duration
    is unknown and only bounded — so pairs are compared instead, and a pair is
    only comparable when the ordering of their outcomes is actually known.

    Higher risk should mean a *shorter* duration, so the model is right when the
    earlier event carries the higher score. 0.5 is coin-flipping.
    """
    t = np.asarray(durations, dtype=float)
    e = np.asarray(events, dtype=bool)
    r = np.asarray(risk_scores, dtype=float)
    if not (t.shape == e.shape == r.shape):
        raise ValueError("durations, events and risk scores must align")

    concordant = 0.0
    comparable = 0.0
    for i in range(t.size):
        if not e[i]:
            # A censored observation cannot be known to have failed first.
            continue
        # Comparable with anything that survived strictly longer. Equal
        # durations are not comparable: neither is known to have failed first.
        others = t > t[i]
        comparable += float(np.sum(others))
        # A tie in the *risk score* is half credit — the model expressed no
        # preference, which is neither right nor wrong. Counting only strict
        # wins scores a constant risk score at 0.0 rather than 0.5, i.e. as
        # perfectly wrong rather than as uninformative.
        concordant += float(np.sum(r[others] < r[i]))
        concordant += 0.5 * float(np.sum(r[others] == r[i]))

    if comparable <= 0:
        return float("nan")
    return concordant / comparable


# ---------------------------------------------------------------------------
# The place a confusion matrix genuinely belongs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfusionMatrix:
    """Counts for a binary decision. ``positive`` means "the alarm fired"."""

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def total(self) -> int:
        return self.true_positive + self.false_positive + self.true_negative + self.false_negative

    def as_grid(self) -> list[list[int]]:
        """``[[TN, FP], [FN, TP]]`` — the layout a heatmap expects."""
        return [
            [self.true_negative, self.false_positive],
            [self.false_negative, self.true_positive],
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "true_negative": self.true_negative,
            "false_negative": self.false_negative,
            "grid": self.as_grid(),
        }


@dataclass(frozen=True)
class ClassificationReport:
    """Precision, recall and specificity for an alarm-shaped decision."""

    matrix: ConfusionMatrix
    precision: float
    recall: float
    specificity: float
    f1: float
    accuracy: float
    balanced_accuracy: float

    @property
    def accuracy_is_misleading(self) -> bool:
        """True when the classes are imbalanced enough that accuracy flatters.

        A breaker that never fires scores 99% accuracy on a catalogue where 1%
        of runs are bad. The flag exists so a dashboard can refuse to lead with
        the number.
        """
        positives = self.matrix.true_positive + self.matrix.false_negative
        if self.matrix.total == 0:
            return False
        rate = positives / self.matrix.total
        return rate < 0.20 or rate > 0.80

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.matrix.as_dict(),
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "accuracy_is_misleading": self.accuracy_is_misleading,
        }


def binary_classification_report(
    truth: Sequence[bool], predicted: Sequence[bool]
) -> ClassificationReport:
    """Score an alarm against what actually happened.

    Built for the circuit breakers and drift alarms, which are genuinely binary
    classifiers: each looks at a batch and emits halt / do-not-halt.

    Precision, recall and specificity are reported separately rather than
    collapsed, because the two error types cost nothing like each other here. A
    false negative publishes a bad batch of prices; a false positive delays a
    good one. Any single score that trades them off has silently picked an
    exchange rate nobody agreed to.

    Args:
        truth: Whether the batch really was bad.
        predicted: Whether the alarm fired.
    """
    t = np.asarray(truth, dtype=bool)
    p = np.asarray(predicted, dtype=bool)
    if t.shape != p.shape:
        raise ValueError(f"truth and predicted must align, got {t.shape} and {p.shape}")
    if t.size == 0:
        raise ValueError("no observations to score")

    matrix = ConfusionMatrix(
        true_positive=int(np.sum(t & p)),
        false_positive=int(np.sum(~t & p)),
        true_negative=int(np.sum(~t & ~p)),
        false_negative=int(np.sum(t & ~p)),
    )

    predicted_positive = matrix.true_positive + matrix.false_positive
    actual_positive = matrix.true_positive + matrix.false_negative
    actual_negative = matrix.true_negative + matrix.false_positive

    precision = matrix.true_positive / predicted_positive if predicted_positive else float("nan")
    recall = matrix.true_positive / actual_positive if actual_positive else float("nan")
    specificity = matrix.true_negative / actual_negative if actual_negative else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
        else float("nan")
    )
    balanced = (
        (recall + specificity) / 2.0
        if np.isfinite(recall) and np.isfinite(specificity)
        else float("nan")
    )

    return ClassificationReport(
        matrix=matrix,
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=f1,
        accuracy=(matrix.true_positive + matrix.true_negative) / matrix.total,
        balanced_accuracy=balanced,
    )


def elasticity_health(
    points: Sequence[float],
    ci_widths: Sequence[float],
    n_declined: int = 0,
    max_ci_width: float = 1.0,
    sign_violation_limit: float = 0.02,
) -> dict[str, Any]:
    """DML health check (metrics.md §4), not a finding about buyers.

    A positive estimated elasticity is almost always residual confounding rather
    than a Giffen good. The share of them is therefore a diagnostic on the
    estimator, and a rate above ``sign_violation_limit`` means the identification
    is not working — not that customers buy more when prices rise.
    """
    values = np.asarray([p for p in points if np.isfinite(p)], dtype=float)
    widths = np.asarray([w for w in ci_widths if np.isfinite(w)], dtype=float)

    if values.size == 0:
        return {"n": 0, "healthy": False, "note": "no finite estimates"}

    sign_violations = float(np.mean(values >= 0))
    median_width = float(np.median(widths)) if widths.size else float("nan")

    return {
        "n": int(values.size),
        "n_declined": n_declined,
        "median": float(np.median(values)),
        "sign_violation_rate": sign_violations,
        "sign_violation_limit": sign_violation_limit,
        "sign_violations_breached": sign_violations > sign_violation_limit,
        "median_ci_width": median_width,
        "ci_width_breached": bool(np.isfinite(median_width) and median_width > max_ci_width),
        "healthy": bool(
            sign_violations <= sign_violation_limit
            and (not np.isfinite(median_width) or median_width <= max_ci_width)
        ),
    }
