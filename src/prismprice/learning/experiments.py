"""
Designed experiments (L5).

The honest answer to "you cannot get causal elasticity from observational data"
is not to document the limitation — it is to build the mechanism that earns the
claim. This module creates price variation on purpose, in a pattern that
supports estimation afterwards.

Switchback, not customer-level split
------------------------------------

The obvious design — show different customers different prices — is unavailable
here, and not for a technical reason. ``PP-G009`` makes the decision context
structurally identity-free: :class:`~prismprice.governance.schemas.PriceRequest`
*cannot carry a customer identifier*, construction raises. Personalised pricing
is out of scope by design, so the randomisation unit has to be something other
than the person.

That leaves time and place. **Switchback** alternates the whole SKU between
treatment arms over time blocks; **geo/store split** assigns whole locations.
Both randomise units that a single customer does not straddle within a block,
which is what keeps the arms comparable.

Two properties worth stating plainly
------------------------------------

**Assignment is a pure function of (unit, block, seed).** No stored state, no
sequence dependence, no assignment table to lose. The same inputs give the same
arm on any machine, at any later date — which is what makes an experiment
reconstructible months later from a decision log, and what lets a batch job be
re-run without reshuffling anyone.

**Blocks must be longer than the effect they measure.** A switchback that flips
faster than demand responds attributes the tail of one arm's effect to the next
arm. :func:`minimum_block_hours` computes the floor from the carryover you
expect, because the temptation is always to shorten blocks for more samples,
and the samples you gain are contaminated.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

__all__ = [
    "Assignment",
    "SwitchbackDesign",
    "geo_split",
    "minimum_block_hours",
]


def _stable_uniform(*parts: object) -> float:
    """Deterministic uniform draw in [0, 1) from arbitrary keys.

    Uses BLAKE2b rather than :func:`hash`, whose salt changes per process: a
    per-process assignment would silently rerandomise the experiment on every
    deploy, and the resulting arms would not be comparable to the ones already
    logged.
    """
    digest = hashlib.blake2b("|".join(str(p) for p in parts).encode("utf-8"), digest_size=8)
    return int.from_bytes(digest.digest(), "big") / float(1 << 64)


@dataclass(frozen=True)
class Assignment:
    """Which arm a unit is in, and everything needed to reproduce that."""

    unit: str
    arm: str
    block_index: int
    block_start: datetime
    propensity: float
    """Probability this unit was assigned this arm. Carried so switchback logs
    can feed off-policy evaluation directly, rather than being a separate kind
    of evidence that has to be analysed by hand."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "arm": self.arm,
            "block_index": self.block_index,
            "block_start": self.block_start.isoformat(),
            "propensity": self.propensity,
        }


@dataclass(frozen=True)
class SwitchbackDesign:
    """Alternates a unit between arms over fixed time blocks.

    Args:
        arms: Arm names. Order is irrelevant; assignment is by hash.
        block_hours: Length of one block. See :func:`minimum_block_hours`.
        epoch: Time origin for block indexing. Fixing it means block boundaries
            do not move when the experiment is restarted.
        weights: Assignment probabilities. Defaults to uniform. Unequal weights
            are useful for a cautious rollout, and the propensity is recorded
            either way.
        seed: Experiment seed. Two experiments with different seeds assign
            independently; the same seed reproduces exactly.
    """

    arms: tuple[str, ...] = ("control", "treatment")
    block_hours: float = 24.0
    epoch: datetime = datetime(2026, 1, 1, tzinfo=timezone.utc)
    weights: tuple[float, ...] | None = None
    seed: int = 20260817

    def __post_init__(self) -> None:
        if len(self.arms) < 2:
            raise ValueError(f"a design needs at least 2 arms, got {self.arms}")
        if self.block_hours <= 0:
            raise ValueError(f"block_hours must be > 0, got {self.block_hours}")
        if self.weights is not None:
            if len(self.weights) != len(self.arms):
                raise ValueError("weights must have one entry per arm")
            if abs(sum(self.weights) - 1.0) > 1e-9:
                raise ValueError(f"weights must sum to 1, got {sum(self.weights)}")
            if any(w <= 0 for w in self.weights):
                raise ValueError(
                    "every arm needs a positive weight; a zero-probability arm cannot be "
                    "reweighted into evidence later"
                )

    def block_index(self, at: datetime) -> int:
        """Which block *at* falls in. Negative before the epoch."""
        delta = at - self.epoch
        return int(np.floor(delta.total_seconds() / (self.block_hours * 3600.0)))

    def block_start(self, index: int) -> datetime:
        return self.epoch + timedelta(hours=self.block_hours * index)

    def assign(self, unit: str, at: datetime) -> Assignment:
        """Assign *unit* for the block containing *at*.

        Pure: the same arguments always give the same arm, on any machine.
        """
        index = self.block_index(at)
        draw = _stable_uniform(self.seed, unit, index)

        weights = self.weights or tuple(1.0 / len(self.arms) for _ in self.arms)
        cumulative = 0.0
        for arm, weight in zip(self.arms, weights, strict=True):
            cumulative += weight
            if draw < cumulative:
                return Assignment(
                    unit=unit,
                    arm=arm,
                    block_index=index,
                    block_start=self.block_start(index),
                    propensity=weight,
                )

        # Only reachable through floating-point summation just below 1.0.
        return Assignment(
            unit=unit,
            arm=self.arms[-1],
            block_index=index,
            block_start=self.block_start(index),
            propensity=weights[-1],
        )

    def assign_many(self, units: Sequence[str], at: datetime) -> list[Assignment]:
        return [self.assign(unit, at) for unit in units]

    def balance(self, units: Sequence[str], blocks: int) -> dict[str, float]:
        """Realised share of each arm over *blocks* consecutive blocks.

        Worth checking before launch rather than after. Hash-based assignment is
        unbiased in expectation but a small unit count over few blocks can land
        badly, and discovering that afterwards means discarding the experiment.
        """
        counts: dict[str, int] = {arm: 0 for arm in self.arms}
        for index in range(blocks):
            when = self.block_start(index)
            for unit in units:
                counts[self.assign(unit, when).arm] += 1
        total = sum(counts.values())
        return {arm: count / total for arm, count in counts.items()} if total else {}


def geo_split(
    unit: str, arms: tuple[str, ...] = ("control", "treatment"), seed: int = 20260817
) -> Assignment:
    """Assign a whole location to an arm, once and permanently.

    The right unit when the treatment cannot be switched quickly — a printed
    shelf label, a regional campaign. Trades statistical power for realism: far
    fewer independent units than a switchback, but no carryover between blocks
    to reason about at all.
    """
    draw = _stable_uniform(seed, "geo", unit)
    index = min(int(draw * len(arms)), len(arms) - 1)
    return Assignment(
        unit=unit,
        arm=arms[index],
        block_index=0,
        block_start=datetime(1970, 1, 1, tzinfo=timezone.utc),
        propensity=1.0 / len(arms),
    )


def minimum_block_hours(carryover_hours: float, safety_factor: float = 3.0) -> float:
    """Shortest block length that does not contaminate the next arm.

    A switchback flipping faster than demand responds attributes the tail of one
    arm's effect to the arm that follows it, biasing the contrast toward zero —
    the direction that makes a real effect look like no effect, and therefore
    the direction that gets an experiment quietly written off.

    Args:
        carryover_hours: How long an effect persists after the price reverts.
            Includes stockpiling: a shopper who bought two at the discount does
            not need one at full price tomorrow.
        safety_factor: Multiple of the carryover to allow. 3x leaves the
            contaminated fraction of a block small.

    Returns:
        Minimum block length in hours.

    Raises:
        ValueError: on non-positive carryover or a safety factor below 1.
    """
    if carryover_hours <= 0:
        raise ValueError(f"carryover_hours must be > 0, got {carryover_hours}")
    if safety_factor < 1.0:
        raise ValueError(
            f"safety_factor must be >= 1, got {safety_factor}; a block shorter than the "
            f"carryover measures the previous arm as much as this one"
        )
    return carryover_hours * safety_factor
