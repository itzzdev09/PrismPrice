"""
Off-policy evaluation (L5).

The question this layer exists to answer: *would the new pricing policy have
done better than the one we were running?* — asked before the new policy touches
a real price.

Every alternative is worse. A backtest that replays logged demand assumes demand
would not have responded to the different price, which is exactly the assumption
a pricing system is not allowed to make. A live A/B test answers honestly and
charges real money for the answer. OPE reweights what was already logged so it
reads as evidence about a policy that was never run.

The reweighting only works because the logging policy was **stochastic and
recorded its own propensities**. That is a constraint on the bandit, not a
convenience here: an action taken with unknown probability contributes nothing,
and a deterministic logging policy makes the whole exercise impossible. This is
why :mod:`prismprice.learning.bandit` records a propensity on every decision
even when it is exploiting.

Three estimators, and the differences matter
--------------------------------------------

``IPS`` is unbiased and can have enormous variance. One logged action taken with
probability 0.01 that the target policy would take almost always carries a
weight near 100, and a single such row can dominate the estimate.

``SNIPS`` divides by the sum of weights instead of ``n``. That introduces a small
bias and usually cuts variance a lot. It is the sane default.

``Doubly robust`` combines a reward model with an IPS correction on its
residuals. It is consistent if *either* the reward model or the propensities are
right — hence the name — and is what to reach for when a reasonable reward model
exists.

Why the diagnostics are not optional
------------------------------------

An IPS estimate always returns a number, and the number is meaningless when the
logging policy rarely took the actions the target policy prefers. Nothing in the
arithmetic says so. Every estimate here therefore carries **effective sample
size** and **maximum importance weight**, and :meth:`OPEEstimate.is_trustworthy`
reads them: an ESS of 12 out of 5,000 logged decisions means the answer rests on
twelve rows regardless of how tight the interval looks.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "LoggedDecision",
    "OPEEstimate",
    "PolicyDistribution",
    "doubly_robust",
    "importance_weights",
    "inverse_propensity",
    "self_normalised_ips",
]

#: A policy, as the distribution it would put over actions in a given context.
#: Returning the whole distribution rather than just the probability of the
#: logged action is what makes doubly-robust estimation possible.
PolicyDistribution = Callable[["LoggedDecision"], Mapping[float, float]]

#: Reward model ``q(x, a)`` for the doubly-robust estimator.
RewardModel = Callable[["LoggedDecision", float], float]


@dataclass(frozen=True)
class LoggedDecision:
    """One decision the logging policy actually took, and what it earned.

    Args:
        action: The price published.
        propensity: ``P(action | context)`` under the **logging** policy, as
            recorded at decision time. Reconstructing it afterwards from a
            fitted model is a different quantity and silently biases every
            estimate built on it.
        reward: Realised outcome — contribution, or whatever the objective
            optimises.
        context_id: Identifier, for clustering and reporting.
        context: Features, for the reward model.
    """

    action: float
    propensity: float
    reward: float
    context_id: str = ""
    context: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < self.propensity <= 1.0:
            raise ValueError(
                f"propensity must be in (0, 1], got {self.propensity}. A zero propensity "
                f"means the logging policy could not have taken this action, so the row "
                f"cannot be reweighted into evidence about anything."
            )


@dataclass(frozen=True)
class OPEEstimate:
    """A policy value estimate, with the diagnostics that say whether to believe it."""

    method: str
    value: float
    std_error: float
    ci_low: float
    ci_high: float
    n: int
    effective_sample_size: float
    """``(sum w)^2 / sum w^2``. How many logged rows the estimate effectively
    rests on. Far below ``n`` means a handful of rows are carrying it."""
    max_weight: float
    clipped_fraction: float

    @property
    def ess_ratio(self) -> float:
        return self.effective_sample_size / self.n if self.n else 0.0

    def is_trustworthy(self, min_ess_ratio: float = 0.10) -> bool:
        """Whether the overlap is good enough for the number to mean anything.

        A separate question from whether the interval is narrow. Poor overlap
        produces confident nonsense, so this is checked *instead of* reading the
        interval, not after it.
        """
        return self.ess_ratio >= min_ess_ratio and np.isfinite(self.value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "value": self.value,
            "std_error": self.std_error,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n": self.n,
            "effective_sample_size": self.effective_sample_size,
            "ess_ratio": self.ess_ratio,
            "max_weight": self.max_weight,
            "clipped_fraction": self.clipped_fraction,
        }


def importance_weights(
    logs: Sequence[LoggedDecision],
    target: PolicyDistribution,
    clip: float | None = None,
) -> tuple[NDArray[np.float64], float]:
    """``pi_target(a|x) / pi_logging(a|x)`` for each logged decision.

    Args:
        logs: Logged decisions.
        target: The policy under evaluation.
        clip: Upper bound on any weight. Clipping trades bias for variance and
            is often worth it, but the fraction clipped is returned so the trade
            is visible rather than silent.

    Returns:
        ``(weights, clipped_fraction)``.
    """
    if not logs:
        raise ValueError("no logged decisions to evaluate")

    raw = np.empty(len(logs), dtype=float)
    for index, record in enumerate(logs):
        distribution = target(record)
        raw[index] = distribution.get(record.action, 0.0) / record.propensity

    if clip is None:
        return raw, 0.0

    clipped = np.minimum(raw, clip)
    fraction = float(np.mean(raw > clip))
    return clipped, fraction


def _diagnostics(weights: NDArray[np.float64]) -> tuple[float, float]:
    total = float(np.sum(weights))
    sum_squares = float(np.sum(weights**2))
    ess = (total**2) / sum_squares if sum_squares > 0 else 0.0
    return ess, float(np.max(weights)) if weights.size else 0.0


def inverse_propensity(
    logs: Sequence[LoggedDecision],
    target: PolicyDistribution,
    clip: float | None = None,
    z: float = 1.959963984540054,
) -> OPEEstimate:
    """Unbiased IPS estimate of the target policy's value.

    Unbiased, and that is the whole of its virtue. A single logged action taken
    with probability 0.01 which the target policy would almost always take
    carries a weight near 100 and can dominate the average, so read the ESS
    before the interval.
    """
    weights, clipped = importance_weights(logs, target, clip)
    rewards = np.array([record.reward for record in logs], dtype=float)

    contributions = weights * rewards
    value = float(np.mean(contributions))
    error = (
        float(np.std(contributions, ddof=1) / np.sqrt(len(logs))) if len(logs) > 1 else float("nan")
    )
    ess, max_weight = _diagnostics(weights)

    return OPEEstimate(
        method="ips",
        value=value,
        std_error=error,
        ci_low=value - z * error,
        ci_high=value + z * error,
        n=len(logs),
        effective_sample_size=ess,
        max_weight=max_weight,
        clipped_fraction=clipped,
    )


def self_normalised_ips(
    logs: Sequence[LoggedDecision],
    target: PolicyDistribution,
    clip: float | None = None,
    z: float = 1.959963984540054,
) -> OPEEstimate:
    """SNIPS: divide by the sum of weights rather than by ``n``.

    Slightly biased, usually far lower variance, and — unlike IPS — it cannot
    return a value outside the range of observed rewards, which is the failure
    mode that most often makes an IPS number obviously untrustworthy at a glance.
    """
    weights, clipped = importance_weights(logs, target, clip)
    rewards = np.array([record.reward for record in logs], dtype=float)

    total = float(np.sum(weights))
    if total <= 0:
        return OPEEstimate(
            method="snips",
            value=float("nan"),
            std_error=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n=len(logs),
            effective_sample_size=0.0,
            max_weight=0.0,
            clipped_fraction=clipped,
        )

    value = float(np.sum(weights * rewards) / total)
    # Delta-method error on the ratio: the residual (r - V) is what the weights
    # actually average over once the normalisation is accounted for.
    residual = weights * (rewards - value)
    error = float(np.sqrt(np.sum(residual**2)) / total)
    ess, max_weight = _diagnostics(weights)

    return OPEEstimate(
        method="snips",
        value=value,
        std_error=error,
        ci_low=value - z * error,
        ci_high=value + z * error,
        n=len(logs),
        effective_sample_size=ess,
        max_weight=max_weight,
        clipped_fraction=clipped,
    )


def doubly_robust(
    logs: Sequence[LoggedDecision],
    target: PolicyDistribution,
    reward_model: RewardModel,
    clip: float | None = None,
    z: float = 1.959963984540054,
) -> OPEEstimate:
    """Doubly-robust estimate: reward model plus an IPS correction.

    Consistent if *either* the reward model or the propensities are correct::

        V = mean[ sum_a pi(a|x) q(x,a)  +  w * (r - q(x,a_logged)) ]

    The first term is the reward model's own answer, evaluated under the target
    policy; the second reweights only the model's *residuals*, so a good model
    shrinks the term that carries the variance. When the model is useless this
    degrades to IPS rather than to nonsense, which is the property worth having.
    """
    weights, clipped = importance_weights(logs, target, clip)

    direct = np.empty(len(logs), dtype=float)
    correction = np.empty(len(logs), dtype=float)

    for index, record in enumerate(logs):
        distribution = target(record)
        direct[index] = sum(
            probability * reward_model(record, action)
            for action, probability in distribution.items()
        )
        correction[index] = weights[index] * (record.reward - reward_model(record, record.action))

    contributions = direct + correction
    value = float(np.mean(contributions))
    error = (
        float(np.std(contributions, ddof=1) / np.sqrt(len(logs))) if len(logs) > 1 else float("nan")
    )
    ess, max_weight = _diagnostics(weights)

    return OPEEstimate(
        method="doubly_robust",
        value=value,
        std_error=error,
        ci_low=value - z * error,
        ci_high=value + z * error,
        n=len(logs),
        effective_sample_size=ess,
        max_weight=max_weight,
        clipped_fraction=clipped,
    )
