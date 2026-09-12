"""
Robust training for the multi-SKU learned policy.

The single-SKU operator in :mod:`prismprice.decision.robust_markdown` handles
elasticity ambiguity inside an exact Bellman recursion. That is unavailable here
— the joint state cannot be enumerated, which is the whole reason the policy is
learned — so ambiguity is handled by randomising the training environment across
the interval instead.

The two are not the same guarantee and these tests are written to keep that
distinction visible. Domain randomisation optimises the *average* over the
interval; the exact operator optimises a *tail* of it and can therefore issue a
certificate. What is asserted here is correspondingly weaker and honest about
it: that the default path is untouched, that the sampler is actually consumed,
and that training over an interval does not simply break the policy.

These tests set ``PRISMPRICE_ALLOW_CPU`` for the same reason
``tests/test_demand.py`` does — CI runners have no GPU, and the policy in
question calls :func:`prismprice.compute.require_gpu`, which is meant to refuse
rather than degrade.
"""

from __future__ import annotations

import numpy as np
import pytest

from prismprice.decision.joint_markdown import (
    JointMarkdownProblem,
    NeuralMarkdownPolicy,
    evaluate_joint_policy,
)

pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def allow_cpu(monkeypatch):
    monkeypatch.setenv("PRISMPRICE_ALLOW_CPU", "1")


@pytest.fixture(scope="module")
def problem() -> JointMarkdownProblem:
    """Small enough to train inside a test, coupled enough to be a real instance."""
    return JointMarkdownProblem(
        prices=(6.0, 8.0, 10.0),
        full_prices=(10.0, 10.0),
        base_demands=(4.0, 3.0),
        elasticity=-2.0,
        unit_costs=(4.0, 4.0),
        salvage_values=(1.0, 1.0),
        horizon=6,
        inventories=(20, 15),
        markdown_budget=40.0,
    )


def _quick(seed: int = 5) -> NeuralMarkdownPolicy:
    """A deliberately tiny training budget. These tests check plumbing and
    invariants, not convergence — a policy trained to optimality inside a unit
    test would make the suite unrunnable and prove nothing extra."""
    return NeuralMarkdownPolicy(hidden=16, episodes=128, batch=32, seed=seed)


def test_default_training_is_unchanged_by_the_new_parameter(problem):
    """The robust path must be strictly additive.

    ``elasticity_sampler=None`` has to reproduce the old behaviour exactly, or
    every previously-recorded multi-SKU result silently changed meaning when
    this parameter was added. Passing it explicitly and omitting it are checked
    against each other on the same seed.
    """
    omitted = _quick().fit(problem)
    explicit = _quick().fit(problem, elasticity_sampler=None)
    assert omitted.history == explicit.history


def test_a_constant_sampler_reproduces_fixed_training(problem):
    """A sampler that always returns the problem's own elasticity is a no-op.

    This is what isolates the *mechanism* from the *randomisation*: if these
    diverge, the difference measured in the benchmark would be an artefact of
    routing demand through a rebuilt problem object rather than of the interval.
    """
    fixed = _quick().fit(problem)
    constant = _quick().fit(problem, elasticity_sampler=lambda rng: problem.elasticity)
    assert fixed.history == constant.history


def test_the_sampler_is_consumed_once_per_episode(problem):
    """Not once per period, and not once per batch.

    An elasticity resampled inside a season would describe customers whose price
    sensitivity changes daily, which averages the ambiguity away rather than
    exposing the policy to it — the failure would be invisible in the loss curve
    and would quietly turn a robust run into a slightly noisier fixed one.
    """
    calls: list[float] = []

    def sampler(rng: np.random.Generator) -> float:
        drawn = float(rng.uniform(-2.6, -1.4))
        calls.append(drawn)
        return drawn

    policy = _quick()
    policy.fit(problem, elasticity_sampler=sampler)

    assert len(calls) == policy.episodes // policy.batch * policy.batch
    assert len(set(calls)) > 1, "sampler returned a constant; the draw is not being used"


def test_robust_training_changes_the_policy(problem):
    """Training over an interval must actually produce a different policy.

    A weak assertion by design. It is here to catch the sampler being accepted
    and then ignored — the most likely way this feature breaks — not to claim
    the robust policy is better, which is a benchmark question and not
    answerable at this training budget.
    """
    fixed = _quick().fit(problem)
    robust = _quick().fit(problem, elasticity_sampler=lambda rng: float(rng.uniform(-3.0, -1.2)))
    assert fixed.history != robust.history


def test_robust_training_still_produces_a_usable_policy(problem):
    """Whatever it learned, it must price every SKU on the ladder and stay solvent."""
    robust = _quick().fit(problem, elasticity_sampler=lambda rng: float(rng.uniform(-3.0, -1.2)))

    prices = robust.price_at(problem.horizon, list(problem.inventories), problem.markdown_budget)
    assert len(prices) == problem.n_skus
    assert all(p in problem.prices for p in prices)

    scored = evaluate_joint_policy(problem, robust.price_at, n_seasons=40, seed=3)
    assert np.isfinite(scored["mean_profit"])
    assert 0.0 <= scored["budget_utilisation"] <= 1.0 + 1e-9


def test_training_is_reproducible_under_a_seed(problem):
    """Reproducibility is a stated guarantee of this repo, and the sampler draws
    from the training generator rather than a fresh one so that it holds."""
    first = _quick(seed=11).fit(
        problem, elasticity_sampler=lambda rng: float(rng.uniform(-3.0, -1.2))
    )
    second = _quick(seed=11).fit(
        problem, elasticity_sampler=lambda rng: float(rng.uniform(-3.0, -1.2))
    )
    assert first.history == second.history
