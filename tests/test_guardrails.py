"""
Guardrail verification.

The properties here assert *system-level guarantees* — "no feasible price is
below cost", "slack agrees with status", "the registry implements every code" —
rather than restating each predicate's own condition. A test that re-derives the
implementation's arithmetic and checks the implementation agrees with it passes
for any self-consistent code, including self-consistently wrong code.
"""

from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from pydantic import ConfigDict, ValidationError, create_model

from prismprice import config
from prismprice.governance.guardrails import (
    AbsoluteFloorGuardrail,
    ChangeFrequencyGuardrail,
    CompetitiveCeilingGuardrail,
    EUOmnibusAnchorGuardrail,
    FairnessGuardrail,
    GuardrailEngine,
    InventoryGuardGuardrail,
    LadderComplianceGuardrail,
    MarginFloorGuardrail,
    MovementCapGuardrail,
    at_least,
    default_guardrails,
)
from prismprice.governance.schemas import (
    DegradationRung,
    GuardrailCode,
    GuardrailStatus,
    PriceRequest,
)
from tests.strategies import AS_OF, money, price_requests, request_and_price


def make_request(**overrides) -> PriceRequest:
    base = dict(
        sku="SKU-001",
        as_of=AS_OF,
        current_price=20.00,
        unit_cost=10.00,
        allowed_price_endings=None,
    )
    base.update(overrides)
    return PriceRequest(**base)


# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------


def test_every_reason_code_has_an_implementation():
    """Regression: PP-G006/7/8 were documented and enumerated but never built."""
    implemented = {g.code for g in default_guardrails()}
    missing = set(GuardrailCode) - implemented
    assert not missing, (
        f"GuardrailCode members with no implementation: {sorted(c.name for c in missing)}"
    )


def test_no_duplicate_codes_in_registry():
    codes = [g.code for g in default_guardrails()]
    assert len(codes) == len(set(codes))


def test_registry_is_ordered_by_code():
    codes = [g.code.value for g in default_guardrails()]
    assert codes == sorted(codes)


# ---------------------------------------------------------------------------
# Cross-cutting properties over the whole engine
# ---------------------------------------------------------------------------


@given(request_and_price())
def test_feasible_price_is_never_below_cost(case):
    """The headline guarantee: whatever the parameters, a feasible price covers cost."""
    request, price = case
    verdict = GuardrailEngine().evaluate_candidate(price, request)
    if verdict.feasible:
        assert at_least(price, request.unit_cost)


@given(request_and_price())
def test_feasible_price_always_clears_the_margin_floor(case):
    request, price = case
    verdict = GuardrailEngine().evaluate_candidate(price, request)
    if verdict.feasible:
        assert at_least(price, request.unit_cost * (1.0 + request.margin_floor_pct))


@given(request_and_price())
def test_slack_sign_agrees_with_status(case):
    """One convention across every guardrail: slack >= 0 iff the check passed.

    Catches a builder using the wrong subtraction order in any single predicate.

    A PASSED verdict may carry a marginally negative slack, because `at_least` /
    `at_most` forgive IEEE-754 representation error. That allowance is
    *relative*, so the permitted overshoot scales with magnitude — the bound
    below is the tolerance itself, not a fixed epsilon.
    """
    request, price = case
    for result in GuardrailEngine().evaluate_candidate(price, request).results:
        if result.slack is None:
            continue
        magnitude = max(abs(result.observed or 0.0), abs(result.limit or 0.0))
        allowance = config.FLOAT_REL_TOL * magnitude + config.FLOAT_ABS_TOL
        if result.status is GuardrailStatus.FAILED:
            assert result.slack < 0
        elif result.status is GuardrailStatus.PASSED:
            assert result.slack >= -allowance


@given(request_and_price())
def test_not_applicable_never_blocks_a_candidate(case):
    request, price = case
    verdict = GuardrailEngine().evaluate_candidate(price, request)
    na_codes = {
        r.reason_code for r in verdict.results if r.status is GuardrailStatus.NOT_APPLICABLE
    }
    assert not (na_codes & set(verdict.binding_constraints))


@given(request_and_price())
def test_binding_constraints_exactly_match_failures(case):
    request, price = case
    verdict = GuardrailEngine().evaluate_candidate(price, request)
    failures = [r.reason_code for r in verdict.results if r.status is GuardrailStatus.FAILED]
    assert verdict.binding_constraints == failures
    assert verdict.feasible == (not failures)


@given(request_and_price())
def test_evaluation_is_deterministic(case):
    """Byte-for-byte reproducibility is a stated guarantee (README §5)."""
    request, price = case
    engine = GuardrailEngine()
    first = engine.evaluate_candidate(price, request).model_dump_json()
    second = engine.evaluate_candidate(price, request).model_dump_json()
    assert first == second


@given(request_and_price())
def test_every_guardrail_reports_a_result(case):
    request, price = case
    verdict = GuardrailEngine().evaluate_candidate(price, request)
    assert len(verdict.results) == len(GuardrailCode)
    assert {r.reason_code for r in verdict.results} == set(GuardrailCode)


# ---------------------------------------------------------------------------
# Monotonicity — the structural shape of floors and ceilings
# ---------------------------------------------------------------------------


@given(request=price_requests(), low=money, delta=st.floats(0.01, 500.0))
def test_floor_feasibility_is_monotone_in_price(request, low, delta):
    """If a price clears a floor, every higher price clears it too."""
    high = low + delta
    for guardrail in (AbsoluteFloorGuardrail(), MarginFloorGuardrail()):
        if guardrail.evaluate(low, request).passed:
            assert guardrail.evaluate(high, request).passed


@given(request=price_requests(), high=money, delta=st.floats(0.01, 500.0))
def test_ceiling_feasibility_is_antitone_in_price(request, high, delta):
    """If a price clears a ceiling, every lower price clears it too."""
    low = max(high - delta, 0.01)
    guardrail = CompetitiveCeilingGuardrail()
    if guardrail.evaluate(high, request).passed:
        assert guardrail.evaluate(low, request).passed


@given(request=price_requests(is_promo=True), high=money, delta=st.floats(0.01, 500.0))
def test_omnibus_is_antitone_in_price(request, high, delta):
    low = max(high - delta, 0.01)
    guardrail = EUOmnibusAnchorGuardrail()
    if guardrail.evaluate(high, request).passed:
        assert guardrail.evaluate(low, request).passed


# ---------------------------------------------------------------------------
# PP-G003 EU Omnibus — fail-closed behaviour
# ---------------------------------------------------------------------------


@given(price=money)
def test_omnibus_fails_closed_when_promo_anchor_is_missing(price):
    """Regression: a promo with no 30-day history used to pass silently."""
    request = make_request(is_promo=True, min_price_last_30d=None)
    result = EUOmnibusAnchorGuardrail().evaluate(price, request)
    assert result.status is GuardrailStatus.FAILED
    assert "failing closed" in result.detail.lower()


@given(price=money)
def test_omnibus_is_not_applicable_when_not_a_promo(price):
    request = make_request(is_promo=False, min_price_last_30d=None)
    result = EUOmnibusAnchorGuardrail().evaluate(price, request)
    assert result.status is GuardrailStatus.NOT_APPLICABLE


def test_omnibus_missing_anchor_makes_the_whole_candidate_infeasible():
    request = make_request(is_promo=True, min_price_last_30d=None, unit_cost=5.0)
    verdict = GuardrailEngine().evaluate_candidate(20.00, request)
    assert not verdict.feasible
    assert GuardrailCode.EU_OMNIBUS in verdict.binding_constraints


def test_omnibus_enforces_the_anchor_when_supplied():
    request = make_request(is_promo=True, min_price_last_30d=18.00)
    assert EUOmnibusAnchorGuardrail().evaluate(17.99, request).passed
    assert EUOmnibusAnchorGuardrail().evaluate(18.00, request).passed
    assert not EUOmnibusAnchorGuardrail().evaluate(18.01, request).passed


# ---------------------------------------------------------------------------
# PP-G009 Fairness — a real check, not a rubber stamp
# ---------------------------------------------------------------------------


def test_fairness_passes_on_the_identity_free_contract():
    result = FairnessGuardrail().evaluate(20.00, make_request())
    assert result.status is GuardrailStatus.PASSED
    assert result.observed == 0.0


def test_fairness_fails_when_an_identity_attribute_is_added():
    """The check must actually fail on a violating context, or it proves nothing."""

    class PersonalisedRequest(PriceRequest):
        customer_id: str = "cust-1"

    request = PersonalisedRequest(sku="S", as_of=AS_OF, current_price=20.0, unit_cost=10.0)
    result = FairnessGuardrail().evaluate(20.00, request)
    assert result.status is GuardrailStatus.FAILED
    assert "customer_id" in result.detail


@pytest.mark.parametrize(
    "field_name",
    ["customer_id", "user_segment", "postcode", "device_type", "loyalty_tier", "age_band"],
)
def test_fairness_rejects_each_protected_attribute_and_proxy(field_name):
    model = create_model("ProxiedRequest", __base__=PriceRequest, **{field_name: (str, "x")})
    request = model(sku="S", as_of=AS_OF, current_price=20.0, unit_cost=10.0)
    assert not FairnessGuardrail().evaluate(20.00, request).passed


def test_fairness_fails_when_the_model_stops_forbidding_extras():
    class LooseRequest(PriceRequest):
        model_config = ConfigDict(frozen=True, extra="allow")

    request = LooseRequest(sku="S", as_of=AS_OF, current_price=20.0, unit_cost=10.0)
    result = FairnessGuardrail().evaluate(20.00, request)
    assert result.status is GuardrailStatus.FAILED


def test_price_request_rejects_unknown_fields_outright():
    with pytest.raises(ValidationError):
        PriceRequest(sku="S", as_of=AS_OF, current_price=20.0, unit_cost=10.0, customer_id="c-1")


# ---------------------------------------------------------------------------
# PP-G006 Change frequency
# ---------------------------------------------------------------------------


def test_change_frequency_not_applicable_without_history():
    result = ChangeFrequencyGuardrail().evaluate(21.00, make_request())
    assert result.status is GuardrailStatus.NOT_APPLICABLE


def test_change_frequency_counts_the_prospective_change():
    request = make_request(price_changes_in_window=3, max_changes_per_window=4)
    assert ChangeFrequencyGuardrail().evaluate(21.00, request).passed

    request = make_request(price_changes_in_window=4, max_changes_per_window=4)
    result = ChangeFrequencyGuardrail().evaluate(21.00, request)
    assert not result.passed
    assert result.observed == 5.0


def test_change_frequency_allows_holding_the_current_price_at_budget():
    request = make_request(price_changes_in_window=4, max_changes_per_window=4)
    assert ChangeFrequencyGuardrail().evaluate(request.current_price, request).passed


# ---------------------------------------------------------------------------
# PP-G007 Inventory guard
# ---------------------------------------------------------------------------


def test_inventory_guard_not_applicable_without_cover_data():
    assert (
        InventoryGuardGuardrail().evaluate(20.00, make_request()).status
        is GuardrailStatus.NOT_APPLICABLE
    )


def test_inventory_shadow_price_binds_below_the_cover_threshold():
    request = make_request(
        unit_cost=10.0,
        inventory_cover_days=3.0,
        min_inventory_cover_days=14.0,
        inventory_shadow_price=5.0,
    )
    assert not InventoryGuardGuardrail().evaluate(14.00, request).passed
    assert InventoryGuardGuardrail().evaluate(15.00, request).passed


def test_inventory_shadow_price_does_not_bind_above_the_threshold():
    request = make_request(
        unit_cost=10.0,
        inventory_cover_days=40.0,
        min_inventory_cover_days=14.0,
        inventory_shadow_price=5.0,
    )
    assert InventoryGuardGuardrail().evaluate(11.00, request).passed


def test_inventory_guard_declares_itself_non_binding_without_a_shadow_price():
    request = make_request(
        inventory_cover_days=1.0, min_inventory_cover_days=14.0, inventory_shadow_price=0.0
    )
    result = InventoryGuardGuardrail().evaluate(11.00, request)
    assert result.passed
    assert "no shadow price was supplied" in result.detail


# ---------------------------------------------------------------------------
# PP-G008 Ladder compliance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "price,expected", [(19.95, True), (19.99, True), (20.00, False), (19.50, False)]
)
def test_ladder_compliance_enforces_price_endings(price, expected):
    request = make_request(allowed_price_endings=(95, 99))
    assert LadderComplianceGuardrail().evaluate(price, request).passed is expected


def test_ladder_compliance_not_applicable_when_disabled():
    request = make_request(allowed_price_endings=None)
    assert (
        LadderComplianceGuardrail().evaluate(20.00, request).status
        is GuardrailStatus.NOT_APPLICABLE
    )


@given(whole=st.integers(min_value=1, max_value=5000), ending=st.sampled_from([95, 99]))
def test_any_price_on_an_allowed_ending_passes(whole, ending):
    request = make_request(allowed_price_endings=(95, 99))
    assert LadderComplianceGuardrail().evaluate(whole + ending / 100, request).passed


# ---------------------------------------------------------------------------
# Tolerance policy — the old absolute 1e-4 slack is gone
# ---------------------------------------------------------------------------


def test_margin_floor_rejects_a_price_one_hundredth_of_a_cent_short():
    """Regression: the previous 1e-4 absolute slack admitted illegal prices."""
    request = make_request(unit_cost=10.0, margin_floor_pct=0.15)
    assert not MarginFloorGuardrail().evaluate(11.5 - 1e-4, request).passed


def test_margin_floor_accepts_exact_equality_despite_float_error():
    """10.00 * 1.15 == 11.499999999999998; a price of exactly 11.50 must pass."""
    request = make_request(unit_cost=10.0, margin_floor_pct=0.15)
    assert MarginFloorGuardrail().evaluate(11.50, request).passed


def test_absolute_floor_uses_the_same_tolerance_as_every_other_guardrail():
    request = make_request(unit_cost=10.0)
    assert AbsoluteFloorGuardrail().evaluate(10.0, request).passed
    assert not AbsoluteFloorGuardrail().evaluate(10.0 - 1e-4, request).passed


@given(cost=money)
def test_float_tolerance_is_never_wide_enough_to_matter_commercially(cost):
    """The allowance must stay at representation-error scale, not cent scale."""
    request = make_request(unit_cost=cost, margin_floor_pct=0.0)
    one_cent_short = cost - 0.01
    assume(one_cent_short > 0)
    assert not AbsoluteFloorGuardrail().evaluate(one_cent_short, request).passed


# ---------------------------------------------------------------------------
# Movement cap and competitive ceiling
# ---------------------------------------------------------------------------


def test_movement_cap_reports_the_relative_change():
    request = make_request(current_price=20.00, movement_cap_pct=0.15)

    at_the_cap = MovementCapGuardrail().evaluate(23.00, request)
    assert at_the_cap.passed, "a move of exactly the cap is permitted"
    assert at_the_cap.observed == pytest.approx(0.15)
    assert at_the_cap.limit == pytest.approx(0.15)

    assert MovementCapGuardrail().evaluate(22.50, request).passed
    assert not MovementCapGuardrail().evaluate(23.01, request).passed
    assert not MovementCapGuardrail().evaluate(16.99, request).passed, "cap is two-sided"


def test_competitive_ceiling_not_applicable_without_a_competitor():
    request = make_request(competitor_price=None)
    assert (
        CompetitiveCeilingGuardrail().evaluate(50.00, request).status
        is GuardrailStatus.NOT_APPLICABLE
    )


def test_competitive_ceiling_caps_at_the_multiplier():
    request = make_request(competitor_price=20.00, competitor_ceiling_multiplier=1.10)
    assert CompetitiveCeilingGuardrail().evaluate(22.00, request).passed
    assert not CompetitiveCeilingGuardrail().evaluate(22.01, request).passed


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------


def test_engine_accepts_a_healthy_candidate(sample_request, engine):
    verdict = engine.evaluate_candidate(20.50, sample_request)
    assert verdict.feasible, [r.detail for r in verdict.results if r.binding]
    assert verdict.binding_constraints == []


def test_engine_reports_all_binding_constraints_not_just_the_first(sample_request, engine):
    verdict = engine.evaluate_candidate(9.00, sample_request)
    assert not verdict.feasible
    assert GuardrailCode.ABSOLUTE_FLOOR in verdict.binding_constraints
    assert GuardrailCode.MARGIN_FLOOR in verdict.binding_constraints
    assert GuardrailCode.MOVEMENT_CAP in verdict.binding_constraints


def test_min_slack_identifies_the_tightest_constraint(sample_request, engine):
    verdict = engine.evaluate_candidate(20.50, sample_request)
    assert verdict.min_slack is not None
    assert verdict.min_slack >= 0


# ---------------------------------------------------------------------------
# Ladder evaluation and the empty feasible set
# ---------------------------------------------------------------------------


def test_ladder_returns_only_feasible_prices(sample_request, engine):
    ladder = [9.00, 11.50, 20.50, 22.95, 40.00]
    result = engine.evaluate_ladder(ladder, sample_request)
    assert result.feasible_prices
    for price in result.feasible_prices:
        assert engine.evaluate_candidate(price, sample_request).feasible
    for price in set(ladder) - set(result.feasible_prices):
        assert not engine.evaluate_candidate(price, sample_request).feasible


def test_empty_feasible_set_falls_back_to_the_previous_price(engine):
    request = make_request(current_price=20.00, unit_cost=100.00, margin_floor_pct=0.20)
    result = engine.evaluate_ladder([18.00, 19.00, 20.00], request)
    assert result.is_empty
    assert result.fallback_price == request.current_price
    assert result.degradation_rung is DegradationRung.FALLBACK_RULE_ENGINE
    assert result.degradation_rung.level == 4


def test_ladder_counts_binding_constraints_for_observability(engine):
    request = make_request(current_price=20.00, unit_cost=100.00)
    result = engine.evaluate_ladder([10.00, 11.00, 12.00], request)
    assert result.binding_constraint_counts[GuardrailCode.ABSOLUTE_FLOOR] == 3


def test_stale_competitor_feed_raises_the_degradation_rung(engine):
    stale = make_request(
        competitor_price=19.50,
        competitor_observed_at=AS_OF - timedelta(hours=config.COMPETITOR_STALENESS_HOURS + 1),
    )
    result = engine.evaluate_ladder([20.50], stale)
    assert result.degradation_rung is DegradationRung.WARN_STALE_COMPETITOR
    assert result.degradation_rung.level == 2


def test_fresh_competitor_feed_stays_at_rung_one(engine):
    fresh = make_request(competitor_price=19.50, competitor_observed_at=AS_OF - timedelta(hours=1))
    result = engine.evaluate_ladder([20.50], fresh)
    assert result.degradation_rung is DegradationRung.OK_OPTIMAL


def test_naive_competitor_timestamp_does_not_raise(engine):
    """Mixed naive/aware timestamps must not blow up the decision path."""
    naive = make_request(competitor_price=19.50, competitor_observed_at=datetime(2026, 8, 17, 0, 0))
    assert engine.evaluate_ladder([20.50], naive).degradation_rung is DegradationRung.OK_OPTIMAL


@given(request_and_price())
def test_ladder_feasible_set_is_a_subset_of_its_candidates(case):
    request, price = case
    ladder = [price, price * 1.05, price * 0.95]
    result = GuardrailEngine().evaluate_ladder(ladder, request)
    assert set(result.feasible_prices) <= set(ladder)
    assert (result.fallback_price is None) == bool(result.feasible_prices)


# ---------------------------------------------------------------------------
# Custom registries
# ---------------------------------------------------------------------------


def test_custom_registry_replaces_the_default():
    engine = GuardrailEngine([AbsoluteFloorGuardrail()])
    assert len(engine.guardrails) == 1
    verdict = engine.evaluate_candidate(9.00, make_request(unit_cost=10.0))
    assert verdict.binding_constraints == [GuardrailCode.ABSOLUTE_FLOOR]


def test_empty_registry_is_respected_not_silently_replaced():
    """`if not guardrails` would have swapped an explicit empty list for the default."""
    engine = GuardrailEngine([])
    assert engine.guardrails == []
    assert engine.evaluate_candidate(0.01, make_request()).feasible


def test_subclass_must_declare_code_and_name():
    from prismprice.governance.guardrails import BaseGuardrail

    with pytest.raises(TypeError, match="must declare a class-level"):

        class Broken(BaseGuardrail):
            def evaluate(self, candidate_price, request):  # pragma: no cover
                raise NotImplementedError


def test_holding_the_price_passes_with_non_negative_slack_when_overspent():
    """Regression: PP-G006 returned PASSED with slack -1.0.

    Found by Hypothesis at CI's 2,000-example depth, on a request whose change
    window was already overspent (1 change made against a cap of 0) and whose
    candidate equalled the current price. Holding consumes no change, so PASSED
    is right; reporting the raw remaining budget as slack made the verdict and
    the slack contradict each other, breaking the one convention every guardrail
    shares — and doing it on exactly the path a system takes when it has run out
    of permission to move.
    """
    request = PriceRequest(
        sku="SKU-001",
        as_of=datetime(2026, 8, 17, 2, tzinfo=timezone.utc),
        current_price=30.0,
        unit_cost=10.0,
        price_changes_in_window=1,
        max_changes_per_window=0,
    )
    results = GuardrailEngine().evaluate_candidate(30.0, request).results
    frequency = next(r for r in results if r.reason_code is GuardrailCode.CHANGE_FREQUENCY)

    assert frequency.status is GuardrailStatus.PASSED
    assert frequency.slack >= 0.0
    assert "overspent" in frequency.detail, "the exhausted budget must still be reported"
