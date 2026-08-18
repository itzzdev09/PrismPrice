"""
Provenance tests.

The guarantee under test is not "the constants have nice values" — no test can
assert that, because a policy dial has no correct value. It is the weaker and
checkable claim that **every constant that can move a price says where it came
from**, and that the claim cannot be satisfied by writing the word LITERATURE
above a number nobody can look up.

The load-bearing test is :func:`test_every_config_constant_is_registered`. It
walks ``config`` and fails on any public numeric constant with no record, so a
future constant cannot be added silently — which is the failure mode that made
the previous configuration unfalsifiable in the first place.
"""

from __future__ import annotations

import inspect
import math

import pytest

from prismprice import config
from prismprice.provenance import (
    Parameter,
    Provenance,
    ProvenanceError,
    audit_decision_path,
    describe,
    frontier,
    gamma_from_tail_tradeoff,
    lambda_from_tradeoff,
    registry,
    require_sourced_decision_path,
)

# ---------------------------------------------------------------------------
# The enforcement itself
# ---------------------------------------------------------------------------


def test_no_decision_path_constant_is_unsourced():
    """CI gate. An unsourced number that can change a price is a defect."""
    require_sourced_decision_path()
    assert audit_decision_path() == []


def _public_config_constants() -> dict[str, object]:
    """Public module-level values in ``config`` that look like tunable numbers."""
    return {
        name: value
        for name, value in vars(config).items()
        if not name.startswith("_")
        and name.isupper()
        and not inspect.ismodule(value)
        and isinstance(value, (int, float, tuple))
        and not isinstance(value, bool)
    }


def test_every_config_constant_is_registered():
    """A new constant cannot be added to config without a provenance record.

    This is the test that makes the whole module more than documentation: it
    fails on the *absence* of a record, so the enforcement does not depend on
    anyone remembering to write one.
    """
    registered = set(registry())
    # Private inverse-specification inputs are inlined into the Parameter they
    # produce, so they carry their evidence through that record.
    exempt = {"ALLOW_CPU_ENV_VAR"}
    missing = sorted(set(_public_config_constants()) - registered - exempt)
    assert not missing, (
        f"config constants with no provenance record: {missing}. "
        f"Declare each via provenance.register(Parameter(...))."
    )


def test_registered_values_match_the_exported_constants():
    """The record and the number cannot drift apart."""
    reg = registry()
    for name, value in _public_config_constants().items():
        if name in reg:
            assert reg[name].value == value, f"{name}: record says {reg[name].value}, config says {value}"


# ---------------------------------------------------------------------------
# The record cannot be satisfied by hand-waving
# ---------------------------------------------------------------------------


def test_literature_without_a_citation_is_rejected():
    with pytest.raises(ProvenanceError, match="no citation"):
        Parameter(
            name="X", value=1.0, provenance=Provenance.LITERATURE, rationale="because a paper"
        )


def test_policy_without_an_owner_is_rejected():
    with pytest.raises(ProvenanceError, match="no owner"):
        Parameter(
            name="X",
            value=1.0,
            provenance=Provenance.POLICY,
            rationale="a preference",
            elicitation="what would you trade?",
            sensitivity=(0.0, 2.0),
        )


def test_policy_without_an_elicitation_is_rejected():
    with pytest.raises(ProvenanceError, match="no elicitation"):
        Parameter(
            name="X",
            value=1.0,
            provenance=Provenance.POLICY,
            rationale="a preference",
            owner="someone",
            sensitivity=(0.0, 2.0),
        )


def test_policy_without_a_sensitivity_bracket_is_rejected():
    """A preference shipped as a point estimate hides whether the answer flips."""
    with pytest.raises(ProvenanceError, match="no sensitivity"):
        Parameter(
            name="X",
            value=1.0,
            provenance=Provenance.POLICY,
            rationale="a preference",
            owner="someone",
            elicitation="what would you trade?",
        )


def test_measured_values_are_rejected_in_configuration():
    """An estimate belongs in a model artefact; a hand-typed one is a placeholder."""
    with pytest.raises(ProvenanceError, match="belong in model artefacts"):
        Parameter(
            name="X", value=-1.8, provenance=Provenance.MEASURED, rationale="fitted it once"
        )


def test_every_parameter_needs_a_rationale():
    with pytest.raises(ProvenanceError, match="rationale is required"):
        Parameter(name="X", value=1.0, provenance=Provenance.TECHNICAL, rationale="   ")


def test_value_outside_its_own_sensitivity_bracket_is_rejected():
    """Catches a bracket copied from a neighbouring parameter."""
    with pytest.raises(ProvenanceError, match="outside its own sensitivity"):
        Parameter(
            name="X",
            value=5.0,
            provenance=Provenance.POLICY,
            rationale="a preference",
            owner="someone",
            elicitation="what would you trade?",
            sensitivity=(0.0, 1.0),
        )


def test_inverted_sensitivity_bracket_is_rejected():
    with pytest.raises(ProvenanceError, match="inverted"):
        Parameter(
            name="X",
            value=1.0,
            provenance=Provenance.TECHNICAL,
            rationale="technical",
            sensitivity=(2.0, 0.5),
        )


# ---------------------------------------------------------------------------
# Objective weights are solved, not chosen
# ---------------------------------------------------------------------------


def test_lambda_is_the_ratio_of_the_stated_trade():
    assert lambda_from_tradeoff(0.30, 1.00) == pytest.approx(0.30)
    assert lambda_from_tradeoff(1.50, 3.00) == pytest.approx(0.50)


def test_lambda_zero_means_pure_margin():
    """Clearance: no weight on modelled future value at all."""
    assert lambda_from_tradeoff(0.0, 1.0) == 0.0


def test_lambda_is_scale_free_in_the_stated_trade():
    """Only the ratio identifies lambda, so the currency unit cannot matter."""
    assert lambda_from_tradeoff(30.0, 100.0) == pytest.approx(lambda_from_tradeoff(0.30, 1.00))


def test_a_trade_gaining_no_future_value_does_not_identify_lambda():
    with pytest.raises(ValueError, match="does not identify lambda"):
        lambda_from_tradeoff(0.30, 0.0)


def test_negative_sacrifice_is_rejected():
    with pytest.raises(ValueError, match="must be >= 0"):
        lambda_from_tradeoff(-0.1, 1.0)


def test_gamma_is_the_ratio_of_the_stated_tail_trade():
    assert gamma_from_tail_tradeoff(0.20, 1.00) == pytest.approx(0.20)


def test_gamma_zero_is_risk_neutral():
    assert gamma_from_tail_tradeoff(0.0, 1.0) == 0.0


def test_gamma_requires_a_real_tail_reduction():
    with pytest.raises(ValueError, match="cvar_reduced must be > 0"):
        gamma_from_tail_tradeoff(0.20, 0.0)


def test_config_weights_match_their_stated_trades():
    """The exported weights are the solved value, not a number typed alongside one."""
    solved_lambda = lambda_from_tradeoff(0.30, 1.00)
    solved_gamma = gamma_from_tail_tradeoff(0.20, 1.00)
    assert math.isclose(config.DEFAULT_CLV_WEIGHT_LAMBDA, solved_lambda, rel_tol=1e-12)
    assert math.isclose(config.DEFAULT_CVAR_WEIGHT_GAMMA, solved_gamma, rel_tol=1e-12)


def test_objective_weights_exist_and_are_policy():
    """README §2 called lambda 'the single dial'; it previously existed nowhere in code."""
    reg = registry()
    for name in ("DEFAULT_CLV_WEIGHT_LAMBDA", "DEFAULT_CVAR_WEIGHT_GAMMA"):
        assert name in reg, f"{name} is in the objective but not in configuration"
        assert reg[name].provenance is Provenance.POLICY
        assert reg[name].requires_local_elicitation, (
            f"{name} is a repo default, and must say so rather than pass as elicited"
        )


# ---------------------------------------------------------------------------
# Sensitivity sweeps
# ---------------------------------------------------------------------------


def test_frontier_spans_the_bracket_inclusively():
    param = registry()["DEFAULT_CLV_WEIGHT_LAMBDA"]
    points = frontier(param, score=lambda v: v * 2, steps=5)
    values = [v for v, _ in points]
    assert values[0] == pytest.approx(param.sensitivity[0])
    assert values[-1] == pytest.approx(param.sensitivity[1])
    assert values == sorted(values)
    assert len(points) == 5


def test_frontier_scores_every_point():
    param = registry()["DEFAULT_CVAR_WEIGHT_GAMMA"]
    points = frontier(param, score=lambda v: round(v, 6), steps=3)
    assert [s for _, s in points] == [round(v, 6) for v, _ in points]


def test_frontier_refuses_a_parameter_with_no_bracket():
    param = Parameter(
        name="X", value=1.0, provenance=Provenance.TECHNICAL, rationale="technical"
    )
    with pytest.raises(ValueError, match="no sensitivity bracket"):
        frontier(param, score=lambda v: v)


def test_frontier_needs_at_least_two_points():
    param = registry()["DEFAULT_CVAR_ALPHA"]
    with pytest.raises(ValueError, match="steps must be >= 2"):
        frontier(param, score=lambda v: v, steps=1)


# ---------------------------------------------------------------------------
# Generator priors are quarantined from the decision path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "SYNTHETIC_ELASTICITY_MEAN",
        "SYNTHETIC_ELASTICITY_SD",
        "SYNTHETIC_REFERENCE_PRICE_HAZARD_THETA",
        "SYNTHETIC_MARGIN_RANGE",
    ],
)
def test_generator_priors_are_not_in_the_decision_path(name):
    """They parameterise the test instrument. They must never price anything."""
    assert registry()[name].in_decision_path is False


def test_elasticity_prior_carries_its_meta_analytic_source():
    param = registry()["SYNTHETIC_ELASTICITY_MEAN"]
    assert param.provenance is Provenance.LITERATURE
    assert "Tellis" in (param.citation or "")
    assert "Bijmolt" in (param.citation or "")


def test_the_unsourced_hazard_parameter_is_labelled_unsourced():
    """The honest failure. theta is invented, and says so rather than borrowing
    a loss-aversion citation that describes a different quantity."""
    param = registry()["SYNTHETIC_REFERENCE_PRICE_HAZARD_THETA"]
    assert param.provenance is Provenance.PLACEHOLDER
    assert not param.is_sourced
    assert "estimated in phase 4" in param.rationale


def test_generator_uses_the_registered_priors():
    """The constants are wired in, not duplicated as literals in the generator."""
    from prismprice.data import synthetic

    source = inspect.getsource(synthetic)
    assert "config.SYNTHETIC_ELASTICITY_MEAN" in source
    assert "config.SYNTHETIC_REFERENCE_PRICE_HAZARD_THETA" in source
    assert "rng.normal(-1.8, 0.4" not in source, "elasticity prior is duplicated as a literal"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_describe_flags_unsourced_values_visibly():
    text = describe()
    assert "UNSOURCED" in text
    assert "SYNTHETIC_REFERENCE_PRICE_HAZARD_THETA" in text


def test_describe_marks_repo_defaults_needing_local_elicitation():
    text = describe(["DEFAULT_MARGIN_FLOOR_PCT"])
    assert "re-elicit before production use" in text


def test_parameter_record_is_serialisable_for_the_decision_log():
    record = registry()["DEFAULT_CLV_WEIGHT_LAMBDA"].as_dict()
    assert record["provenance"] == "POLICY"
    assert record["sensitivity"] == [0.0, 1.0]
    assert record["elicitation"]
