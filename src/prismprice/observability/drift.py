"""
Feature and prediction drift (L7).

A model is fitted on one distribution and asked to price against another. Drift
is the gap between them, and the reason it needs monitoring rather than
inspection is that nothing fails when it opens: the demand model keeps returning
quantiles, the objective keeps sampling them, the guardrails keep passing, and
the prices are wrong.

Population Stability Index
--------------------------

For a feature binned into ``k`` buckets, with reference share ``e_j`` and
current share ``a_j``::

    PSI = sum_j (a_j - e_j) * ln(a_j / e_j)

Zero when the distributions match, and growing as they separate. It is symmetric
in the two distributions, which is why it is preferred here to KL divergence:
"the training data had customers the live data does not" and "the live data has
customers the training data did not" are both problems, and a directional
measure only shouts about one of them.

Three implementation details that change the number
---------------------------------------------------

**Bin edges come from the reference distribution, and only from it.** Re-binning
on the current data makes PSI structurally blind: quantile bins computed on
whatever arrived today will always contain roughly equal shares of today's data,
so a distribution that has shifted wholesale reports as stable. This is the
mistake that makes a drift monitor worse than no monitor, because it produces a
green number.

**Empty bins are floored, and the floor is reported.** A bin with zero current
observations makes the logarithm infinite. Flooring the share at a small epsilon
keeps the statistic finite, but it also caps how loud an *entirely* vacated bin
can be — so the count of floored bins travels with the result rather than being
swallowed.

**PSI is meaningless without its bin count.** More bins means more PSI for the
same underlying shift; a threshold of 0.25 calibrated on 10 bins is a different
test at 50. :class:`DriftReport` carries ``n_bins`` for exactly this reason, and
two reports with different bin counts should not be compared.

What PSI does not tell you
--------------------------

Whether the drift matters. A feature the model barely uses can move enormously
and harmlessly; a feature it leans on can move slightly and break it. PSI ranks
*change*, not *impact*, and the honest use is as a trigger for investigation
rather than as a verdict. :attr:`DriftReport.largest_contributor` exists because
the first question after "something drifted" is always "which part of it".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = [
    "DriftReport",
    "categorical_drift",
    "feature_drift",
    "population_stability_index",
]

#: Conventional PSI bands. `0.25` is the threshold docs/metrics.md §4 alerts on.
_MODERATE = 0.10
_SIGNIFICANT = 0.25

#: Share floor for empty bins, so the logarithm stays finite.
_EPSILON = 1e-6


@dataclass(frozen=True)
class DriftReport:
    """PSI for one feature, with enough detail to act on it."""

    name: str
    psi: float
    n_bins: int
    """Carried because PSI is not comparable across bin counts."""
    n_reference: int
    n_current: int
    bin_contributions: tuple[float, ...]
    floored_bins: int
    """Bins whose share had to be floored to keep the statistic finite. A high
    count means whole regions of the reference distribution are now unvisited,
    and the PSI understates that rather than overstating it."""

    @property
    def verdict(self) -> str:
        if self.psi >= _SIGNIFICANT:
            return "SIGNIFICANT"
        if self.psi >= _MODERATE:
            return "MODERATE"
        return "STABLE"

    @property
    def breached(self) -> bool:
        """Whether this trips the §4 alert threshold."""
        return self.psi >= _SIGNIFICANT

    @property
    def largest_contributor(self) -> int:
        """Index of the bin contributing most of the PSI.

        The first question after "something drifted" is which part of it, and a
        single scalar cannot answer that.
        """
        if not self.bin_contributions:
            return -1
        return int(np.argmax(np.abs(np.asarray(self.bin_contributions))))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "psi": self.psi,
            "verdict": self.verdict,
            "breached": self.breached,
            "n_bins": self.n_bins,
            "n_reference": self.n_reference,
            "n_current": self.n_current,
            "floored_bins": self.floored_bins,
            "largest_contributor": self.largest_contributor,
        }


def _bin_edges(reference: NDArray[np.float64], n_bins: int) -> NDArray[np.float64]:
    """Quantile edges from the reference distribution alone.

    Deduplicated: a feature that is constant over much of its range produces
    repeated quantiles, and ``np.histogram`` rejects non-monotonic edges. The
    effective bin count therefore falls, which is why the report carries the
    count it actually used rather than the one that was requested.
    """
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges: NDArray[np.float64] = np.unique(np.quantile(reference, quantiles))
    # Open the outer edges so current values beyond the reference range land in
    # the end bins instead of being dropped — a value the model has never seen
    # is the single most important thing a drift monitor can notice.
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def population_stability_index(
    reference: NDArray[np.float64] | pd.Series,
    current: NDArray[np.float64] | pd.Series,
    name: str = "feature",
    n_bins: int = 10,
) -> DriftReport:
    """PSI between a reference and a current sample.

    Args:
        reference: The distribution the model was fitted on.
        current: What is arriving now.
        name: Feature name, for reporting.
        n_bins: Requested quantile bins. The realised count can be lower when
            the reference has repeated quantiles, and the report says which.

    Returns:
        :class:`DriftReport`.

    Raises:
        ValueError: on an empty sample or fewer than two bins.
    """
    reference_values = np.asarray(reference, dtype=float)
    current_values = np.asarray(current, dtype=float)

    reference_values = reference_values[np.isfinite(reference_values)]
    current_values = current_values[np.isfinite(current_values)]

    if reference_values.size == 0 or current_values.size == 0:
        raise ValueError(
            f"{name}: need non-empty reference and current samples, got "
            f"{reference_values.size} and {current_values.size}"
        )
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")

    edges = _bin_edges(reference_values, n_bins)
    if edges.size < 3:
        # A constant reference feature has no distribution to drift from.
        return DriftReport(
            name=name,
            psi=0.0,
            n_bins=1,
            n_reference=int(reference_values.size),
            n_current=int(current_values.size),
            bin_contributions=(),
            floored_bins=0,
        )

    reference_counts, _ = np.histogram(reference_values, bins=edges)
    current_counts, _ = np.histogram(current_values, bins=edges)

    expected = reference_counts / reference_counts.sum()
    actual = current_counts / current_counts.sum()

    floored = int(np.sum((expected < _EPSILON) | (actual < _EPSILON)))
    expected = np.clip(expected, _EPSILON, None)
    actual = np.clip(actual, _EPSILON, None)

    contributions = (actual - expected) * np.log(actual / expected)

    return DriftReport(
        name=name,
        psi=float(np.sum(contributions)),
        n_bins=int(edges.size - 1),
        n_reference=int(reference_values.size),
        n_current=int(current_values.size),
        bin_contributions=tuple(float(c) for c in contributions),
        floored_bins=floored,
    )


def categorical_drift(
    reference: pd.Series,
    current: pd.Series,
    name: str = "feature",
) -> DriftReport:
    """PSI over category shares rather than quantile bins.

    Categories present in one sample and absent from the other are kept as bins
    with a floored share, not dropped. A category that has vanished — a
    discontinued brand, a closed store — is drift, and dropping it would make
    the most obvious kind of change invisible.
    """
    reference_counts = reference.value_counts()
    current_counts = current.value_counts()
    categories = sorted(set(reference_counts.index) | set(current_counts.index), key=str)

    if not categories:
        raise ValueError(f"{name}: no categories in either sample")

    expected = np.array([reference_counts.get(c, 0) for c in categories], dtype=float)
    actual = np.array([current_counts.get(c, 0) for c in categories], dtype=float)

    expected = expected / max(expected.sum(), 1.0)
    actual = actual / max(actual.sum(), 1.0)

    floored = int(np.sum((expected < _EPSILON) | (actual < _EPSILON)))
    expected = np.clip(expected, _EPSILON, None)
    actual = np.clip(actual, _EPSILON, None)

    contributions = (actual - expected) * np.log(actual / expected)

    return DriftReport(
        name=name,
        psi=float(np.sum(contributions)),
        n_bins=len(categories),
        n_reference=int(reference.size),
        n_current=int(current.size),
        bin_contributions=tuple(float(c) for c in contributions),
        floored_bins=floored,
    )


def feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str] | None = None,
    n_bins: int = 10,
) -> dict[str, DriftReport]:
    """PSI for every shared column, numeric or categorical.

    Args:
        reference: Training-window features.
        current: Live features.
        columns: Restrict to these. ``None`` uses every column present in both.
        n_bins: Bins for numeric features.

    Returns:
        Reports keyed by column name, ordered worst-first so a caller reading
        the top of the dict is reading the thing to investigate.

    Raises:
        ValueError: when the frames share no columns, which is a wiring fault
            rather than a drift finding and should not report as "no drift".
    """
    shared = [c for c in (columns or reference.columns) if c in current.columns]
    if not shared:
        raise ValueError(
            "reference and current frames share no columns; this is a wiring fault, "
            "not an absence of drift"
        )

    reports: dict[str, DriftReport] = {}
    for column in shared:
        if pd.api.types.is_numeric_dtype(reference[column]) and pd.api.types.is_numeric_dtype(
            current[column]
        ):
            reports[column] = population_stability_index(
                reference[column].to_numpy(), current[column].to_numpy(), name=column, n_bins=n_bins
            )
        else:
            reports[column] = categorical_drift(
                reference[column].astype(str), current[column].astype(str), name=column
            )

    return dict(sorted(reports.items(), key=lambda item: -item[1].psi))
