"""
Safe contextual bandit tests.

The bandit's most important output is not the price it picks but the
**propensity it records**: off-policy evaluation is only possible because the
probability of each action was known at decision time.
:func:`test_bandit_logs_can_be_evaluated_off_policy` closes that loop by feeding
real bandit output into the estimators rather than hand-written propensities.
"""

from __future__ import annotations

import numpy as np
import pytest

from prismprice.learning.bandit import ActionValue, SafeExplorationPolicy
from prismprice.learning.ope import LoggedDecision, self_normalised_ips

LADDER = [25.95, 27.95, 29.99, 31.99, 33.95]
UNIT_COST = 15.0
TARGET_PRICE = 31.99
BEHAVIOUR = [0.10, 0.20, 0.40, 0.20, 0.10]


def expected_reward(price: float) -> float:
    """Known reward curve, so the target policy's value is arithmetic."""
    return 100.0 * (price / 30.0) ** -1.8 * (price - UNIT_COST)


TRUE_TARGET_VALUE = expected_reward(TARGET_PRICE)


def deterministic_target(record: LoggedDecision) -> dict[float, float]:
    return {price: (1.0 if price == TARGET_PRICE else 0.0) for price in LADDER}


# ---------------------------------------------------------------------------
# Safe exploration
# ---------------------------------------------------------------------------


def _actions(values: dict[float, float], std_error: float = 0.0) -> list[ActionValue]:
    return [ActionValue(price=p, expected_value=v, std_error=std_error) for p, v in values.items()]


def test_greedy_action_holds_most_of_the_mass():
    policy = SafeExplorationPolicy(exploration_rate=0.10)
    distribution = policy.action_distribution(_actions({30.0: 100.0, 31.0: 99.0, 32.0: 98.0}))
    assert distribution[30.0] == pytest.approx(0.90)
    assert sum(distribution.values()) == pytest.approx(1.0)


def test_unsafe_actions_get_no_probability():
    """The floor is on value, so a badly worse action is simply unreachable."""
    policy = SafeExplorationPolicy(exploration_rate=0.20, safety_alpha=0.05)
    distribution = policy.action_distribution(_actions({30.0: 100.0, 31.0: 99.0, 32.0: 10.0}))
    assert distribution[32.0] == 0.0
    assert distribution[31.0] > 0.0


def test_safety_is_judged_on_the_pessimistic_value():
    """Using the point estimate would explore hardest exactly where the model is
    least sure, which is the opposite of safe."""
    policy = SafeExplorationPolicy(exploration_rate=0.20, safety_alpha=0.05)
    certain = policy.action_distribution(_actions({30.0: 100.0, 31.0: 98.0}, std_error=0.0))
    uncertain = policy.action_distribution(_actions({30.0: 100.0, 31.0: 98.0}, std_error=20.0))
    assert certain[31.0] > 0.0
    assert uncertain[31.0] == 0.0, "a wide interval must not buy exploration"


def test_the_greedy_action_is_always_reachable():
    """Refusing the best known action because its own interval is wide would
    leave the policy with nothing to do."""
    policy = SafeExplorationPolicy(exploration_rate=0.20)
    distribution = policy.action_distribution(_actions({30.0: 100.0, 31.0: 99.0}, std_error=50.0))
    assert distribution[30.0] > 0.0


def test_every_choice_records_a_usable_propensity():
    """The bandit's most important output. Without it nothing downstream works."""
    policy = SafeExplorationPolicy(exploration_rate=0.30, seed=1)
    for _ in range(50):
        choice = policy.choose(_actions({30.0: 100.0, 31.0: 99.5, 32.0: 99.0}))
        assert 0.0 < choice.propensity <= 1.0
        assert choice.propensity == pytest.approx(choice.distribution[choice.price])


def test_exploration_rate_is_realised_in_the_long_run():
    policy = SafeExplorationPolicy(exploration_rate=0.30, seed=5)
    choices = [policy.choose(_actions({30.0: 100.0, 31.0: 99.5, 32.0: 99.0})) for _ in range(3000)]
    realised = float(np.mean([c.is_exploratory for c in choices]))
    assert 0.25 < realised < 0.35


def test_risk_budget_is_spent_and_then_stops_exploration():
    """A budget that cannot run out is not a budget."""
    policy = SafeExplorationPolicy(exploration_rate=0.50, risk_budget=20.0, seed=2)
    actions = _actions({30.0: 100.0, 31.0: 98.0})

    for _ in range(400):
        policy.choose(actions)

    assert policy.spent >= 20.0
    assert policy.budget_remaining == 0.0
    assert policy.action_distribution(actions)[31.0] == 0.0, "explored past an exhausted budget"


def test_exhausted_budget_still_returns_a_valid_distribution():
    policy = SafeExplorationPolicy(exploration_rate=0.50, risk_budget=0.0, seed=2)
    distribution = policy.action_distribution(_actions({30.0: 100.0, 31.0: 98.0}))
    assert distribution[30.0] == pytest.approx(1.0)
    assert sum(distribution.values()) == pytest.approx(1.0)


def test_exploration_is_reproducible_from_the_seed():
    """A decision log that cannot be replayed is not an audit trail."""
    actions = _actions({30.0: 100.0, 31.0: 99.5, 32.0: 99.0})
    first = [SafeExplorationPolicy(seed=11).choose(actions).price for _ in range(1)]
    second = [SafeExplorationPolicy(seed=11).choose(actions).price for _ in range(1)]
    assert first == second


def test_bandit_rejects_impossible_configuration():
    with pytest.raises(ValueError, match="exploration_rate must be in"):
        SafeExplorationPolicy(exploration_rate=1.5)
    with pytest.raises(ValueError, match="safety_alpha must be in"):
        SafeExplorationPolicy(safety_alpha=1.0)
    with pytest.raises(ValueError, match="no candidate actions"):
        SafeExplorationPolicy().choose([])


# ---------------------------------------------------------------------------
# The loop closes
# ---------------------------------------------------------------------------


def test_bandit_logs_can_be_evaluated_off_policy():
    """The point of recording propensities, demonstrated end to end.

    Real bandit output — not hand-written propensities — is fed to OPE, and the
    estimator recovers the known value of a policy the bandit never followed.
    """
    rng = np.random.default_rng(4)
    policy = SafeExplorationPolicy(exploration_rate=0.40, safety_alpha=0.50, seed=9)
    actions = _actions({price: expected_reward(price) for price in LADDER}, std_error=5.0)

    logs = []
    for _ in range(8000):
        choice = policy.choose(actions)
        logs.append(
            LoggedDecision(
                action=choice.price,
                propensity=choice.propensity,
                reward=expected_reward(choice.price) + rng.normal(0.0, 80.0),
            )
        )

    estimate = self_normalised_ips(logs, deterministic_target)
    assert estimate.ci_low <= TRUE_TARGET_VALUE <= estimate.ci_high

    # The target policy commits to an action the bandit played only as
    # exploration, at 0.40/4 = 0.10. So the effective sample size lands at
    # ~10% of the log by construction, and that is the arithmetic rather than a
    # weakness: evaluating a policy the logger rarely followed *should* rest on
    # the rows where it did. Asserting the relationship is worth more than
    # asserting a boolean threshold the design happens to sit exactly on.
    exploration_share = 0.40 / (len(LADDER) - 1)
    assert estimate.ess_ratio == pytest.approx(exploration_share, rel=0.15)
    assert estimate.max_weight == pytest.approx(1.0 / exploration_share, rel=0.05)
    assert estimate.is_trustworthy(min_ess_ratio=exploration_share * 0.9)
