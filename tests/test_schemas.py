"""
Data-contract tests: immutability, derived fields, and audit reconstructability.
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from prismprice.governance.schemas import (
    CandidateScore,
    DecisionEstimates,
    DecisionInputs,
    DecisionRecord,
    DegradationRung,
    ElasticityEstimate,
    GuardrailCode,
    GuardrailResult,
    GuardrailStatus,
    PriceRequest,
)

AS_OF = datetime(2026, 8, 17, 2, 0, tzinfo=timezone.utc)


def make_record(**overrides) -> DecisionRecord:
    base = dict(
        decision_id="11111111-1111-4111-8111-111111111111",
        sku="SKU-001",
        as_of=AS_OF,
        policy_version="1.4.2",
        model_versions={"demand": "d-2026.08.1", "causal_elasticity": "dml-2026.08.1"},
        inputs=DecisionInputs(current_price=32.00, unit_cost=15.00),
        recommended_price=30.95,
        degradation_reason_code=DegradationRung.OK_OPTIMAL,
        seed=20260817,
    )
    base.update(overrides)
    return DecisionRecord(**base)


# ---------------------------------------------------------------------------
# Degradation rung consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code,level",
    [
        (DegradationRung.OK_OPTIMAL, 1),
        (DegradationRung.WARN_STALE_COMPETITOR, 2),
        (DegradationRung.FALLBACK_POOLED_ELASTICITY, 3),
        (DegradationRung.FALLBACK_RULE_ENGINE, 4),
        (DegradationRung.CRITICAL_MAINTAIN_PREV, 5),
    ],
)
def test_rung_level_matches_the_degradation_matrix(code, level):
    assert code.level == level


def test_every_rung_has_a_level():
    assert {r.level for r in DegradationRung} == {1, 2, 3, 4, 5}


def test_record_rung_is_derived_not_stored():
    """Rung and reason code cannot drift apart, because there is only one field."""
    record = make_record(degradation_reason_code=DegradationRung.FALLBACK_RULE_ENGINE)
    assert record.degradation_rung == 4
    with pytest.raises(ValidationError):
        DecisionRecord(
            decision_id="d",
            sku="S",
            as_of=AS_OF,
            policy_version="1.0.0",
            inputs=DecisionInputs(current_price=1.0, unit_cost=0.5),
            recommended_price=1.0,
            degradation_reason_code=DegradationRung.OK_OPTIMAL,
            degradation_rung=5,
            seed=1,
        )


def test_derived_rung_survives_serialisation():
    assert make_record().model_dump()["degradation_rung"] == 1


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_decision_record_is_immutable():
    record = make_record()
    with pytest.raises(ValidationError):
        record.recommended_price = 25.00


def test_price_request_is_immutable():
    request = PriceRequest(sku="S", as_of=AS_OF, current_price=20.0, unit_cost=10.0)
    with pytest.raises(ValidationError):
        request.current_price = 21.0


def test_corrections_supersede_rather_than_mutate():
    original = make_record()
    correction = make_record(
        decision_id="22222222-2222-4222-8222-222222222222",
        recommended_price=29.95,
        supersedes_id=original.decision_id,
    )
    assert correction.supersedes_id == original.decision_id
    assert original.recommended_price == 30.95


# ---------------------------------------------------------------------------
# Audit completeness
# ---------------------------------------------------------------------------


def test_policy_version_cannot_be_defaulted():
    """An audit record that invents its own version is not reconstructible."""
    with pytest.raises(ValidationError):
        DecisionRecord(
            decision_id="d",
            sku="S",
            as_of=AS_OF,
            inputs=DecisionInputs(current_price=1.0, unit_cost=0.5),
            recommended_price=1.0,
            degradation_reason_code=DegradationRung.OK_OPTIMAL,
            seed=1,
        )


def test_seed_is_required():
    with pytest.raises(ValidationError):
        DecisionRecord(
            decision_id="d",
            sku="S",
            as_of=AS_OF,
            policy_version="1.0.0",
            inputs=DecisionInputs(current_price=1.0, unit_cost=0.5),
            recommended_price=1.0,
            degradation_reason_code=DegradationRung.OK_OPTIMAL,
        )


def test_record_round_trips_through_json():
    record = make_record(
        estimates=DecisionEstimates(
            causal_elasticity=ElasticityEstimate(
                point=-1.38, ci_low=-1.81, ci_high=-0.94, method="dml", confidence="high"
            ),
            delta_clv=-0.42,
        ),
        candidates=[CandidateScore(price=30.95, j_score=241.1, cvar=180.2)],
    )
    restored = DecisionRecord.model_validate_json(record.model_dump_json())
    assert restored.model_dump_json() == record.model_dump_json()


def test_inputs_snapshot_captures_promo_context():
    """is_promo must reach the record, or Omnibus applicability is unreconstructible."""
    request = PriceRequest(
        sku="S",
        as_of=AS_OF,
        current_price=20.0,
        unit_cost=10.0,
        is_promo=True,
        min_price_last_30d=18.0,
    )
    inputs = DecisionInputs.from_request(request)
    assert inputs.is_promo is True
    assert inputs.min_price_last_30d == 18.0


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------


def test_cvar_alpha_is_explicit_rather_than_encoded_in_a_field_name():
    score = CandidateScore(price=30.95, j_score=241.1, cvar=180.2, cvar_alpha=0.01)
    assert score.cvar_alpha == 0.01
    assert not hasattr(score, "cvar5")


@pytest.mark.parametrize("alpha", [0.0, 1.0, 1.5, -0.1])
def test_cvar_alpha_must_be_a_probability(alpha):
    with pytest.raises(ValidationError):
        CandidateScore(price=1.0, j_score=1.0, cvar=1.0, cvar_alpha=alpha)


def test_elasticity_ci_width():
    estimate = ElasticityEstimate(
        point=-1.38, ci_low=-1.81, ci_high=-0.94, method="dml", confidence="high"
    )
    assert estimate.ci_width == pytest.approx(0.87)


# ---------------------------------------------------------------------------
# Guardrail result semantics
# ---------------------------------------------------------------------------


def test_not_applicable_is_distinguishable_from_passed():
    """A missing-data pass must never read as a compliance pass in the audit log."""
    na = GuardrailResult(
        reason_code=GuardrailCode.EU_OMNIBUS,
        constraint_name="EU Omnibus Anchor",
        status=GuardrailStatus.NOT_APPLICABLE,
        detail="not a promo",
    )
    ok = GuardrailResult(
        reason_code=GuardrailCode.EU_OMNIBUS,
        constraint_name="EU Omnibus Anchor",
        status=GuardrailStatus.PASSED,
        detail="within anchor",
    )
    assert na.passed and ok.passed
    assert na.status is not ok.status
    assert not na.binding and not ok.binding


def test_failed_result_is_binding():
    result = GuardrailResult(
        reason_code=GuardrailCode.ABSOLUTE_FLOOR,
        constraint_name="Absolute Floor",
        status=GuardrailStatus.FAILED,
        detail="below cost",
        slack=-1.0,
    )
    assert result.binding
    assert not result.passed


def test_guardrail_codes_are_stable_identifiers():
    """These appear in shipped audit records; renumbering breaks history."""
    assert GuardrailCode.ABSOLUTE_FLOOR.value == "PP-G001"
    assert GuardrailCode.FAIRNESS_CHECK.value == "PP-G009"
    assert len({c.value for c in GuardrailCode}) == len(GuardrailCode)
