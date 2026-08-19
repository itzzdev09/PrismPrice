"""
Safe contextual bandit (L5).

Exploration is how a pricing system stops being trapped by its own history. A
policy that only ever prices where it already priced never learns what happens
elsewhere, and the causal estimator downstream is left inferring elasticity from
whatever variation the business happened to create for unrelated reasons. This
module deliberately creates that variation instead.

It also spends real money doing it, which is why *safe* is not decoration.

Three properties, and none of them are optional
-----------------------------------------------

**Every action records its propensity.** This is the module's most important
output and it is easy to mistake for bookkeeping. Off-policy evaluation is only
possible because the probability of each action was known *at decision time*;
reconstructing it later from a fitted model estimates a different quantity and
biases every downstream comparison. So the propensity is written even when the
bandit is exploiting and the probability is 0.9-something.

**Exploration is bounded below by the baseline, not by a step size.** The
constraint is on *value*: an action may only be explored if its pessimistic
value — lower confidence bound, not point estimate — stays within
``safety_alpha`` of the incumbent policy's value. Using the point estimate would
mean exploring hardest exactly where the model is least sure, which is the
opposite of safe and is how an exploration scheme discovers that its uncertainty
was real by losing money.

**The risk budget is cumulative and it is spent, not checked.** A per-decision
bound permits a thousand small losses; the budget tracks the total expected
shortfall handed out so far and stops exploring when it is gone. Exploration
that cannot run out is not a budget, it is a hope.

The bandit proposes; L4 still disposes. Actions offered here are already the
guardrail-feasible set, so exploration cannot reach an illegal price — it can
only choose differently among legal ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from prismprice import config

__all__ = [
    "ActionValue",
    "BanditChoice",
    "SafeExplorationPolicy",
]

#: z for a one-sided 95% lower confidence bound.
_Z_95_ONE_SIDED = 1.6448536269514722


@dataclass(frozen=True)
class ActionValue:
    """One candidate action and what the objective thinks it is worth.

    Args:
        price: The candidate.
        expected_value: ``J(p)`` from the decision engine.
        std_error: Uncertainty on that value. Monte-Carlo error at minimum;
            widen it to include model uncertainty if you have it. Passing zero
            declares the value known exactly, which turns the safety bound into
            a point comparison and removes the protection.
    """

    price: float
    expected_value: float
    std_error: float = 0.0

    def lower_confidence_bound(self, z: float = _Z_95_ONE_SIDED) -> float:
        """Pessimistic value. What the safety constraint is judged on."""
        return self.expected_value - z * self.std_error


@dataclass(frozen=True)
class BanditChoice:
    """A chosen action, with everything off-policy evaluation will need."""

    price: float
    propensity: float
    is_exploratory: bool
    distribution: dict[float, float]
    safe_actions: tuple[float, ...]
    expected_shortfall: float
    """Expected value given up versus the greedy action. The amount this single
    decision spends from the risk budget."""
    budget_remaining: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "price": self.price,
            "propensity": self.propensity,
            "is_exploratory": self.is_exploratory,
            "expected_shortfall": self.expected_shortfall,
            "budget_remaining": self.budget_remaining,
            "n_safe_actions": len(self.safe_actions),
        }


@dataclass
class SafeExplorationPolicy:
    """Epsilon-greedy over guardrail-feasible actions, bounded by a value floor.

    Args:
        exploration_rate: Probability mass reserved for non-greedy actions.
        safety_alpha: Permitted shortfall against the incumbent. ``0.05`` means
            an explored action's *pessimistic* value must stay within 5% of the
            baseline's.
        risk_budget: Total expected value the policy may spend on exploration
            over its lifetime. ``None`` means unbudgeted, which should be used
            only in simulation.
        seed: RNG seed. Exploration is random, and a decision log that cannot be
            replayed is not an audit trail.
    """

    exploration_rate: float = 0.10
    safety_alpha: float = 0.05
    risk_budget: float | None = None
    seed: int = config.DEFAULT_SEED

    spent: float = field(default=0.0, init=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.exploration_rate <= 1.0:
            raise ValueError(f"exploration_rate must be in [0, 1], got {self.exploration_rate}")
        if not 0.0 <= self.safety_alpha < 1.0:
            raise ValueError(f"safety_alpha must be in [0, 1), got {self.safety_alpha}")
        if self.risk_budget is not None and self.risk_budget < 0:
            raise ValueError(f"risk_budget must be >= 0, got {self.risk_budget}")
        self._rng = np.random.default_rng(self.seed)

    @property
    def budget_remaining(self) -> float | None:
        if self.risk_budget is None:
            return None
        return max(self.risk_budget - self.spent, 0.0)

    def safe_actions(
        self, actions: list[ActionValue], baseline_value: float | None = None
    ) -> list[ActionValue]:
        """The actions exploration is permitted to reach.

        Judged on the lower confidence bound, so an action is explorable only if
        it is *probably* acceptable, not merely acceptable in expectation. The
        greedy action is always included: refusing to take the best known action
        because its own interval is wide would leave the policy with nothing to
        do.
        """
        if not actions:
            raise ValueError("no candidate actions to choose between")

        greedy = max(actions, key=lambda a: a.expected_value)
        baseline = greedy.expected_value if baseline_value is None else baseline_value
        floor = baseline - abs(baseline) * self.safety_alpha

        safe = [a for a in actions if a.lower_confidence_bound() >= floor]
        if greedy not in safe:
            safe.append(greedy)
        return safe

    def action_distribution(
        self, actions: list[ActionValue], baseline_value: float | None = None
    ) -> dict[float, float]:
        """The full probability distribution over actions.

        Returned in full, not just for the chosen action, because off-policy
        evaluation of a *future* policy needs to ask what this one would have
        done across the board.
        """
        safe = self.safe_actions(actions, baseline_value)
        greedy = max(actions, key=lambda a: a.expected_value)

        exploring = self.exploration_rate if self._can_explore() else 0.0
        others = [a for a in safe if a.price != greedy.price]

        distribution = {a.price: 0.0 for a in actions}
        if not others or exploring == 0.0:
            distribution[greedy.price] = 1.0
            return distribution

        share = exploring / len(others)
        distribution[greedy.price] = 1.0 - exploring
        for action in others:
            distribution[action.price] = share
        return distribution

    def choose(
        self, actions: list[ActionValue], baseline_value: float | None = None
    ) -> BanditChoice:
        """Pick an action and record everything OPE will need to reuse it."""
        distribution = self.action_distribution(actions, baseline_value)
        greedy = max(actions, key=lambda a: a.expected_value)

        prices = list(distribution)
        probabilities = np.array([distribution[p] for p in prices], dtype=float)
        probabilities = probabilities / probabilities.sum()

        index = int(self._rng.choice(len(prices), p=probabilities))
        chosen_price = prices[index]
        chosen = next(a for a in actions if a.price == chosen_price)

        shortfall = max(greedy.expected_value - chosen.expected_value, 0.0)
        is_exploratory = chosen_price != greedy.price
        if is_exploratory:
            self.spent += shortfall

        return BanditChoice(
            price=chosen_price,
            propensity=float(probabilities[index]),
            is_exploratory=is_exploratory,
            distribution=distribution,
            safe_actions=tuple(a.price for a in self.safe_actions(actions, baseline_value)),
            expected_shortfall=shortfall,
            budget_remaining=self.budget_remaining,
        )

    def _can_explore(self) -> bool:
        """False once the risk budget is exhausted.

        Checked before choosing rather than after spending, so the budget is a
        limit rather than a report on how far it was exceeded.
        """
        if self.risk_budget is None:
            return True
        return self.spent < self.risk_budget
