"""
Designed experiment tests.

Assignment must be a pure function of (unit, block, seed): no stored state, no
sequence dependence, and identical on any machine at any later date. That is
what makes an experiment reconstructible months later from a decision log.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from prismprice.learning.experiments import (
    SwitchbackDesign,
    geo_split,
    minimum_block_hours,
)
from prismprice.learning.ope import LoggedDecision

# ---------------------------------------------------------------------------
# Designed experiments
# ---------------------------------------------------------------------------


def test_assignment_is_a_pure_function():
    """No stored state, so an experiment can be reconstructed months later."""
    design = SwitchbackDesign()
    when = datetime(2026, 3, 1, 12, tzinfo=timezone.utc)
    assert design.assign("SKU-001", when).arm == design.assign("SKU-001", when).arm


def test_assignment_is_stable_across_processes():
    """`hash()` is salted per process, so it would rerandomise on every deploy
    and the new arms would not be comparable to the logged ones."""
    design = SwitchbackDesign(seed=42)
    when = datetime(2026, 3, 1, 12, tzinfo=timezone.utc)
    assert design.assign("SKU-001", when).arm == "control"


def test_units_switch_arms_across_blocks():
    design = SwitchbackDesign(block_hours=24.0)
    start = datetime(2026, 3, 1, tzinfo=timezone.utc)
    arms = {design.assign("SKU-001", start + timedelta(days=d)).arm for d in range(30)}
    assert arms == {"control", "treatment"}, "a unit that never switches is not a switchback"


def test_arm_is_constant_within_a_block():
    design = SwitchbackDesign(block_hours=24.0)
    start = datetime(2026, 3, 1, 0, tzinfo=timezone.utc)
    arms = {design.assign("SKU-001", start + timedelta(hours=h)).arm for h in range(24)}
    assert len(arms) == 1


def test_assignment_is_balanced_over_many_blocks():
    design = SwitchbackDesign(seed=3)
    balance = design.balance([f"SKU-{i:03d}" for i in range(50)], blocks=40)
    assert 0.45 < balance["control"] < 0.55


def test_unequal_weights_are_honoured_and_recorded():
    design = SwitchbackDesign(weights=(0.9, 0.1), seed=3)
    balance = design.balance([f"SKU-{i:03d}" for i in range(100)], blocks=40)
    assert 0.85 < balance["control"] < 0.95
    assignment = design.assign("SKU-001", datetime(2026, 3, 1, tzinfo=timezone.utc))
    assert assignment.propensity in (0.9, 0.1)


def test_switchback_propensities_feed_ope_directly():
    """An experiment log is the same kind of evidence as a bandit log."""
    design = SwitchbackDesign(weights=(0.7, 0.3), seed=8)
    assignment = design.assign("SKU-001", datetime(2026, 3, 1, tzinfo=timezone.utc))
    record = LoggedDecision(action=30.0, propensity=assignment.propensity, reward=100.0)
    assert 0.0 < record.propensity <= 1.0


def test_zero_weight_arm_is_refused():
    with pytest.raises(ValueError, match="positive weight"):
        SwitchbackDesign(weights=(1.0, 0.0))


def test_design_rejects_impossible_configuration():
    with pytest.raises(ValueError, match="at least 2 arms"):
        SwitchbackDesign(arms=("only",))
    with pytest.raises(ValueError, match="block_hours must be > 0"):
        SwitchbackDesign(block_hours=0.0)
    with pytest.raises(ValueError, match="weights must sum to 1"):
        SwitchbackDesign(weights=(0.5, 0.9))


def test_geo_split_is_permanent_for_a_location():
    first = geo_split("store-042")
    later = geo_split("store-042")
    assert first.arm == later.arm


def test_block_length_must_exceed_carryover():
    """A switchback flipping faster than demand responds biases the contrast
    toward zero — the direction that makes a real effect look like none."""
    assert minimum_block_hours(carryover_hours=8.0) == pytest.approx(24.0)
    assert minimum_block_hours(carryover_hours=8.0, safety_factor=5.0) == pytest.approx(40.0)


def test_block_length_rejects_a_shrinking_safety_factor():
    with pytest.raises(ValueError, match="safety_factor must be >= 1"):
        minimum_block_hours(carryover_hours=8.0, safety_factor=0.5)
    with pytest.raises(ValueError, match="carryover_hours must be > 0"):
        minimum_block_hours(carryover_hours=0.0)
