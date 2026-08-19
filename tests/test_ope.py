"""
Off-policy evaluation tests.

The phase-8 gate is that OPE recovers a **known** policy value from synthetic
logs. That is checkable here and never in production: the reward function is
chosen, so the target policy's true value is arithmetic, and an estimator can be
scored on the thing it exists to estimate rather than on whether its output
looks sensible.
"""

from __future__ import annotations

import numpy as np
import pytest

from prismprice.learning.ope import (
    LoggedDecision,
    doubly_robust,
    importance_weights,
    inverse_propensity,
    self_normalised_ips,
)

LADDER = [25.95, 27.95, 29.99, 31.99, 33.95]
UNIT_COST = 15.0
TARGET_PRICE = 31.99
BEHAVIOUR = [0.10, 0.20, 0.40, 0.20, 0.10]


def expected_reward(price: float) -> float:
    """Known reward curve, so the target policy's value is arithmetic."""
    return 100.0 * (price / 30.0) ** -1.8 * (price - UNIT_COST)


TRUE_TARGET_VALUE = expected_reward(TARGET_PRICE)


def make_logs(n: int = 6000, noise: float = 120.0, seed: int = 7) -> list[LoggedDecision]:
    rng = np.random.default_rng(seed)
    logs = []
    for _ in range(n):
        index = int(rng.choice(len(LADDER), p=BEHAVIOUR))
        price = LADDER[index]
        logs.append(
            LoggedDecision(
                action=price,
                propensity=BEHAVIOUR[index],
                reward=expected_reward(price) + rng.normal(0.0, noise),
            )
        )
    return logs


def deterministic_target(record: LoggedDecision) -> dict[float, float]:
    return {price: (1.0 if price == TARGET_PRICE else 0.0) for price in LADDER}


# ---------------------------------------------------------------------------
# The phase-8 gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("estimator", [inverse_propensity, self_normalised_ips])
def test_ope_recovers_the_known_policy_value(estimator):
    """Phase-8 gate: the estimate covers the true value of a policy never run."""
    estimate = estimator(make_logs(), deterministic_target)
    assert estimate.ci_low <= TRUE_TARGET_VALUE <= estimate.ci_high, (
        f"{estimate.method}: {estimate.value:.2f} "
        f"[{estimate.ci_low:.2f}, {estimate.ci_high:.2f}] misses {TRUE_TARGET_VALUE:.2f}"
    )


def test_doubly_robust_recovers_the_known_policy_value():
    estimate = doubly_robust(
        make_logs(),
        deterministic_target,
        reward_model=lambda record, action: expected_reward(action),
    )
    assert estimate.ci_low <= TRUE_TARGET_VALUE <= estimate.ci_high


def test_ips_is_unbiased_across_independent_log_sets():
    """Unbiasedness is a statement about the average, not about one sample."""
    values = [
        inverse_propensity(make_logs(n=3000, seed=s), deterministic_target).value for s in range(12)
    ]
    assert float(np.mean(values)) == pytest.approx(TRUE_TARGET_VALUE, rel=0.03)


def test_snips_has_lower_variance_than_ips():
    """The reason SNIPS is the sane default, measured rather than asserted."""
    ips, snips = [], []
    for seed in range(12):
        logs = make_logs(n=3000, seed=seed)
        ips.append(inverse_propensity(logs, deterministic_target).value)
        snips.append(self_normalised_ips(logs, deterministic_target).value)
    assert float(np.std(snips)) < float(np.std(ips))


def test_doubly_robust_survives_a_useless_reward_model():
    """Consistent if *either* nuisance is right. Here the model is wrong on
    purpose, so the propensities have to carry it."""
    estimate = doubly_robust(
        make_logs(), deterministic_target, reward_model=lambda record, action: 0.0
    )
    assert estimate.ci_low <= TRUE_TARGET_VALUE <= estimate.ci_high


def test_evaluating_the_logging_policy_returns_its_own_average():
    """Sanity anchor: evaluating the policy that generated the logs must give
    back the logs' mean reward, with weights all equal to 1."""
    logs = make_logs()

    def logging_policy(record: LoggedDecision) -> dict[float, float]:
        return dict(zip(LADDER, BEHAVIOUR, strict=True))

    estimate = self_normalised_ips(logs, logging_policy)
    observed_mean = float(np.mean([record.reward for record in logs]))
    assert estimate.value == pytest.approx(observed_mean, rel=1e-9)


# ---------------------------------------------------------------------------
# The diagnostics that decide whether to believe any of it
# ---------------------------------------------------------------------------


def test_poor_overlap_shows_up_as_low_effective_sample_size():
    """An IPS estimate always returns a number; nothing in the arithmetic says
    it rests on a handful of rows."""
    rare = [0.97, 0.01, 0.005, 0.01, 0.005]
    rng = np.random.default_rng(3)
    logs = []
    for _ in range(4000):
        index = int(rng.choice(len(LADDER), p=rare))
        logs.append(
            LoggedDecision(
                action=LADDER[index],
                propensity=rare[index],
                reward=expected_reward(LADDER[index]),
            )
        )

    estimate = inverse_propensity(logs, deterministic_target)
    assert estimate.ess_ratio < 0.10
    assert not estimate.is_trustworthy()


def test_good_overlap_is_reported_as_trustworthy():
    assert self_normalised_ips(make_logs(), deterministic_target).is_trustworthy()


def test_effective_sample_size_tracks_the_logged_share():
    """The target takes one action the logger played 20% of the time, so the
    estimate should rest on roughly 20% of the rows."""
    estimate = self_normalised_ips(make_logs(n=6000), deterministic_target)
    assert 0.15 < estimate.ess_ratio < 0.25


def test_clipping_is_reported_rather_than_silent():
    logs = make_logs()
    weights, fraction = importance_weights(logs, deterministic_target, clip=2.0)
    assert float(np.max(weights)) <= 2.0
    assert fraction > 0.0
    assert inverse_propensity(logs, deterministic_target, clip=2.0).clipped_fraction > 0.0


def test_zero_propensity_is_refused_at_construction():
    """A row the logging policy could not have produced cannot be reweighted
    into evidence about anything."""
    with pytest.raises(ValueError, match="propensity must be in"):
        LoggedDecision(action=30.0, propensity=0.0, reward=1.0)


def test_a_target_outside_the_logged_support_is_visibly_unusable():
    """No overlap at all: SNIPS returns NaN rather than a confident number."""
    logs = make_logs(n=500)
    estimate = self_normalised_ips(logs, lambda record: {99.99: 1.0})
    assert np.isnan(estimate.value)
    assert not estimate.is_trustworthy()


def test_empty_logs_are_refused():
    with pytest.raises(ValueError, match="no logged decisions"):
        inverse_propensity([], deterministic_target)
