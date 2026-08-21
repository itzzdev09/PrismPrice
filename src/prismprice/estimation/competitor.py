"""
Competitor reaction and price-war risk (L2).

Every other estimator in this system treats the competitor's price as an
exogenous input: something observed, fed to a guardrail as a ceiling, and
otherwise left alone. That is safe exactly as long as the competitor is not
also running a pricing system.

Once they are, the competitor price stops being data and becomes a *response*.
Cutting to undercut them changes the number you were undercutting, and a policy
that optimises against today's observed competitor price is optimising against a
quantity its own action destroys. Two such policies pointed at each other walk
prices to the floor without either ever choosing to.

The reaction function
---------------------

What the competitor does next, given what we did::

    p_them(t) = alpha + beta * p_us(t-1) + gamma * p_them(t-1) + delta'X(t)

``beta`` is the reaction: how much of our move they follow. ``gamma`` is their
own price stickiness. The lag matters — they observe our price and then respond,
so writing both at time ``t`` would make the regression simultaneous and the
coefficient meaningless.

**The immediate reaction understates the danger.** With stickiness, a move of
ours keeps propagating: the long-run response is ``beta / (1 - gamma)``, which
for ``beta = 0.4, gamma = 0.5`` is ``0.8`` — double the headline number. War risk
must be judged on the long-run figure, and
:attr:`ReactionFunction.long_run_response` is what the stability test reads.

The spiral condition
--------------------

With our own rule ``p_us = a + b * p_them`` and theirs ``p_them = c + d * p_us``,
substitution gives a unique interior fixed point::

    p_us* = (a + b*c) / (1 - b*d)

which exists and is stable **iff** ``|b*d| < 1``. That product is the whole
diagnostic. Below one, mutual reactions damp out and prices settle. At or above
one, each round of matching is amplified rather than absorbed and there is no
equilibrium to settle at — prices run until they hit a floor that somebody else
put there.

This is why the guardrail layer is not sufficient on its own. ``PP-G002`` will
stop any *individual* price below the margin floor, and it will stop it every
day for a year while margin bleeds away above the floor. A constraint catches
the illegal price; only a model of the other player's response catches the
losing strategy.

What this cannot do
-------------------

Estimating a reaction from observational data has the identification problem
that ``estimation/elasticity.py`` exists to solve, in a different costume. Two
firms facing the same cost shock or the same seasonal demand move together
without either reacting to the other, and a naive regression reads that
co-movement as reaction — biased *upward*, so the estimate says "price war risk"
when the truth is "shared January". ``controls`` exists for this and the
inflation is measured in the tests rather than asserted away. Genuine
identification would need price variation the competitor cannot anticipate,
which is what the experiment layer produces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "EquilibriumAnalysis",
    "ReactionFunction",
    "estimate_reaction",
    "find_equilibrium",
    "simulate_price_path",
]


@dataclass(frozen=True)
class ReactionFunction:
    """How a competitor's price responds to ours."""

    intercept: float
    reaction: float
    """``beta``: fraction of our move they follow in the next period."""
    persistence: float
    """``gamma``: how much of their own previous price carries forward."""
    reaction_std_error: float
    r_squared: float
    n_observations: int
    control_names: tuple[str, ...] = ()

    @property
    def long_run_response(self) -> float:
        """``beta / (1 - gamma)``: total eventual follow-through of one move.

        The number the stability test must use. A competitor who follows 40% of
        our move immediately but holds half their own price each period ends up
        following 80% of it, and judging war risk on the 40% would call a
        divergent pair stable.
        """
        if self.persistence >= 1.0:
            return float("inf")
        return self.reaction / (1.0 - self.persistence)

    @property
    def is_significant(self) -> bool:
        """Whether the reaction is distinguishable from no reaction at all."""
        if self.reaction_std_error <= 0:
            return abs(self.reaction) > 0
        return abs(self.reaction) > 1.96 * self.reaction_std_error

    def respond(self, our_price: float, their_current_price: float) -> float:
        """Their next price, given ours and theirs today."""
        return float(
            self.intercept + self.reaction * our_price + self.persistence * their_current_price
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "intercept": self.intercept,
            "reaction": self.reaction,
            "persistence": self.persistence,
            "long_run_response": self.long_run_response,
            "reaction_std_error": self.reaction_std_error,
            "is_significant": self.is_significant,
            "r_squared": self.r_squared,
            "n_observations": self.n_observations,
            "controls": list(self.control_names),
        }


def estimate_reaction(
    our_prices: NDArray[np.float64],
    their_prices: NDArray[np.float64],
    controls: dict[str, NDArray[np.float64]] | None = None,
) -> ReactionFunction:
    """Fit the competitor's reaction from a history of both prices.

    Regresses their price at ``t`` on our price at ``t-1``, their own price at
    ``t-1``, and any controls at ``t``. The lag on our price is what keeps the
    regression from being simultaneous: contemporaneous prices are jointly
    determined, and a coefficient fitted on them answers no question.

    Args:
        our_prices: Our price series, time-ordered.
        their_prices: Their price series, same index.
        controls: Common factors both firms respond to — cost indices,
            seasonality, category demand. **Omitting a real one biases the
            reaction upward**, because co-movement gets read as reaction, and
            the direction of that bias is toward declaring a price war that is
            not happening.

    Returns:
        The fitted :class:`ReactionFunction`.

    Raises:
        ValueError: on mismatched or too-short series.
    """
    ours = np.asarray(our_prices, dtype=float)
    theirs = np.asarray(their_prices, dtype=float)

    if ours.shape != theirs.shape:
        raise ValueError(f"series must be the same length, got {ours.shape} and {theirs.shape}")
    if ours.size < 10:
        raise ValueError(
            f"need at least 10 observations to fit a reaction, got {ours.size}; a "
            f"coefficient from fewer is noise with a standard error attached"
        )

    target = theirs[1:]
    columns = [np.ones_like(target), ours[:-1], theirs[:-1]]
    names: list[str] = []

    if controls:
        for name, values in controls.items():
            series = np.asarray(values, dtype=float)
            if series.shape != ours.shape:
                raise ValueError(f"control {name!r} must match the price series length")
            columns.append(series[1:])
            names.append(name)

    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)

    fitted = design @ coefficients
    residuals = target - fitted
    dof = max(target.size - design.shape[1], 1)
    sigma_squared = float(residuals @ residuals) / dof

    try:
        covariance = sigma_squared * np.linalg.inv(design.T @ design)
        reaction_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
    except np.linalg.LinAlgError:
        reaction_error = float("nan")

    total = float(np.sum((target - target.mean()) ** 2))
    r_squared = 1.0 - float(residuals @ residuals) / total if total > 0 else 0.0

    return ReactionFunction(
        intercept=float(coefficients[0]),
        reaction=float(coefficients[1]),
        persistence=float(coefficients[2]),
        reaction_std_error=reaction_error,
        r_squared=r_squared,
        n_observations=int(target.size),
        control_names=tuple(names),
    )


@dataclass(frozen=True)
class EquilibriumAnalysis:
    """Where two reacting policies settle, or whether they settle at all."""

    our_price: float
    their_price: float
    spiral_coefficient: float
    """``b * d``: the product of the two reaction slopes. The whole diagnostic."""
    is_stable: bool
    converged: bool
    iterations: int
    settled_at_bound: bool = False
    """True when the path stopped because it hit the floor or ceiling rather
    than because the two rules balanced. A divergent pair *does* stop moving —
    at whatever bound a guardrail put there — and reading that as an equilibrium
    is the exact mistake this module exists to prevent."""

    @property
    def is_equilibrium(self) -> bool:
        """Whether the resting point is a real fixed point.

        Requires all three: it converged, the reactions damp, and it did not
        merely run into a constraint. A price pinned to the margin floor by a
        guardrail satisfies the first and neither of the others.
        """
        return self.converged and self.is_stable and not self.settled_at_bound

    @property
    def war_risk(self) -> str:
        """Plain-language verdict, for a decision record and an alert.

        Banded rather than continuous because the action differs by band: below
        0.5 nothing changes, above 1.0 the matching rule must be abandoned, and
        the range between is where a human should be looking.
        """
        magnitude = abs(self.spiral_coefficient)
        if magnitude >= 1.0:
            return "DIVERGENT"
        if magnitude >= 0.8:
            return "FRAGILE"
        if magnitude >= 0.5:
            return "DAMPED"
        return "STABLE"

    def as_dict(self) -> dict[str, Any]:
        return {
            "our_price": self.our_price,
            "their_price": self.their_price,
            "spiral_coefficient": self.spiral_coefficient,
            "is_stable": self.is_stable,
            "converged": self.converged,
            "iterations": self.iterations,
            "settled_at_bound": self.settled_at_bound,
            "is_equilibrium": self.is_equilibrium,
            "war_risk": self.war_risk,
        }


def find_equilibrium(
    our_intercept: float,
    our_slope: float,
    their_reaction: ReactionFunction,
    max_iterations: int = 500,
    tolerance: float = 1e-9,
    price_floor: float = 0.0,
    price_ceiling: float = 1e6,
    start_price: float = 30.0,
) -> EquilibriumAnalysis:
    """Fixed point of two mutually reacting pricing rules.

    Our rule is ``p_us = our_intercept + our_slope * p_them``; theirs is the
    fitted reaction. The interior fixed point is closed-form, but the iteration
    is run anyway because it is the thing that actually happens in the world —
    and because a divergent pair has no fixed point to compute, only a path to
    the floor.

    Args:
        our_intercept: Constant in our own matching rule.
        our_slope: How much of their price we follow. ``0`` means we ignore
            them, which is always stable.
        their_reaction: Their fitted response.
        price_floor: Where a descent stops — the margin floor a guardrail
            enforces. A divergent pair runs *to this*, which is exactly the
            outcome the guardrail cannot prevent by itself.
        price_ceiling: Upper bound, standing in for the competitive ceiling.
            Divergence runs in whichever direction the feedback points, and an
            unbounded run reports a resting price of 1e11 that means nothing.
        start_price: Where both sides begin the walk.

    Returns:
        :class:`EquilibriumAnalysis`.
    """
    spiral = our_slope * their_reaction.long_run_response
    stable = abs(spiral) < 1.0

    our_price = float(start_price)
    their_price = float(start_price)
    converged = False
    iterations_run = 0

    for step in range(1, max_iterations + 1):
        iterations_run = step
        next_ours = float(
            np.clip(our_intercept + our_slope * their_price, price_floor, price_ceiling)
        )
        next_theirs = float(
            np.clip(their_reaction.respond(next_ours, their_price), price_floor, price_ceiling)
        )

        if abs(next_ours - our_price) < tolerance and abs(next_theirs - their_price) < tolerance:
            our_price, their_price = next_ours, next_theirs
            converged = True
            break
        our_price, their_price = next_ours, next_theirs

    # Bounds are what a guardrail enforces. Stopping at one is not the same
    # event as two reaction functions balancing, and the caller must be able to
    # tell them apart: the first means the strategy lost and a constraint caught
    # the price, the second means the strategy settled.
    at_bound = bool(
        np.isclose(our_price, price_floor, atol=1e-6)
        or np.isclose(our_price, price_ceiling, atol=1e-6)
        or np.isclose(their_price, price_floor, atol=1e-6)
        or np.isclose(their_price, price_ceiling, atol=1e-6)
    )

    return EquilibriumAnalysis(
        our_price=float(our_price),
        their_price=float(their_price),
        spiral_coefficient=float(spiral),
        is_stable=stable,
        converged=converged,
        iterations=iterations_run,
        settled_at_bound=at_bound,
    )


def simulate_price_path(
    our_rule: Any,
    their_reaction: ReactionFunction,
    initial_our_price: float,
    initial_their_price: float,
    periods: int = 60,
    price_floor: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Play both policies forward and return the two price paths.

    The demonstration that makes the spiral coefficient concrete: a matching
    rule that looks reasonable in isolation walks to ``price_floor`` when the
    other side is also matching.

    Args:
        our_rule: ``their_price -> our_price``.
        their_reaction: Their fitted response.
        price_floor: Lower bound both sides respect, standing in for the margin
            guardrail.

    Returns:
        ``(our_path, their_path)``, each of length ``periods + 1`` including the
        starting prices.
    """
    if periods < 1:
        raise ValueError(f"periods must be >= 1, got {periods}")

    ours = np.empty(periods + 1, dtype=float)
    theirs = np.empty(periods + 1, dtype=float)
    ours[0], theirs[0] = initial_our_price, initial_their_price

    for t in range(1, periods + 1):
        ours[t] = max(float(our_rule(theirs[t - 1])), price_floor)
        theirs[t] = max(their_reaction.respond(ours[t], theirs[t - 1]), price_floor)

    return ours, theirs
