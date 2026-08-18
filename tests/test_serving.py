"""
Serving layer tests.

The phase-7 gate has two halves: the API contracts hold, and the **chaos test**
— killing the model store — degrades cleanly to rung 5
(``CRITICAL_MAINTAIN_PREV``) instead of failing.

The chaos test kills the real dependency rather than mocking its response. A
mock that returns an error tests the mock; flipping the store to unavailable
tests what the service does when the thing it needs is genuinely gone, which is
the only version of the question worth asking.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from prismprice.decision import DecisionEngine
from prismprice.governance.schemas import DegradationRung, PriceRequest
from prismprice.serving import DecisionService, InMemoryModelStore, ModelStoreUnavailable
from prismprice.serving.app import create_app

AS_OF = datetime(2026, 8, 17, 2, tzinfo=timezone.utc)


def demand(sku: str, price: float) -> tuple[float, float, float]:
    units = 100.0 * (price / 30.0) ** -1.8
    return (units * 0.7, units, units * 1.4)


def request_payload(**overrides) -> dict:
    payload = {
        "sku": "SKU-001",
        "as_of": "2026-08-17T02:00:00Z",
        "current_price": 30.0,
        "unit_cost": 15.0,
    }
    payload.update(overrides)
    return payload


def make_request(**overrides) -> PriceRequest:
    payload = {"sku": "SKU-001", "as_of": AS_OF, "current_price": 30.0, "unit_cost": 15.0}
    payload.update(overrides)
    return PriceRequest(**payload)


@pytest.fixture
def store() -> InMemoryModelStore:
    return InMemoryModelStore(demand=demand, versions={"demand": "d-2026.08.1"})


@pytest.fixture
def service(store) -> DecisionService:
    return DecisionService(store=store, engine=DecisionEngine(policy_version="1.4.2"))


@pytest.fixture
def client(service):
    from fastapi.testclient import TestClient

    return TestClient(create_app(service))


# ---------------------------------------------------------------------------
# The chaos test
# ---------------------------------------------------------------------------


def test_dead_model_store_degrades_to_rung_five(service, store):
    """Phase-7 gate. The dependency is killed, not mocked."""
    store.available = False
    outcome = service.decide(make_request())

    assert outcome.record.degradation_reason_code is DegradationRung.CRITICAL_MAINTAIN_PREV
    assert outcome.record.degradation_rung == 5


def test_rung_five_holds_the_current_price(service, store):
    """The system stops having an opinion rather than degrading to one it
    cannot justify."""
    store.available = False
    outcome = service.decide(make_request(current_price=27.5))
    assert outcome.recommended_price == pytest.approx(27.5)


def test_rung_five_records_no_candidates(service, store):
    """None were scored. A ladder here would make the record look like a
    decision that weighed alternatives and chose to stand still."""
    store.available = False
    outcome = service.decide(make_request())
    assert outcome.record.candidates == []
    assert outcome.ladder == ()


def test_dead_store_is_still_a_200(client, store):
    """A degraded decision is a successful recommendation of 'do not move'.

    A 5xx pushes the caller into a fallback of their own, outside the audit log
    — which is the outcome the degradation matrix exists to prevent.
    """
    store.available = False
    response = client.post("/decide", json=request_payload())
    assert response.status_code == 200
    assert response.json()["degradation_reason_code"] == "CRITICAL_MAINTAIN_PREV"


def test_service_recovers_when_the_store_returns(service, store):
    store.available = False
    assert service.decide(make_request()).record.degradation_rung == 5
    store.available = True
    assert service.decide(make_request()).record.degradation_rung == 1


def test_batch_degrades_per_request_not_wholesale(service, store):
    """A catalogue refresh that returns nothing because one artefact is missing
    is how every price silently goes stale overnight."""
    requests = [make_request(sku=f"SKU-{i:03d}") for i in range(5)]
    store.available = False
    outcomes = service.decide_batch(requests)

    assert len(outcomes) == len(requests)
    assert all(o.record.degradation_rung == 5 for o in outcomes)
    assert [o.record.sku for o in outcomes] == [r.sku for r in requests]


def test_health_probes_rather_than_remembers(service, store):
    """A store that died after startup must not still read green."""
    assert service.health().model_store_available is True
    store.available = False
    assert service.health().model_store_available is False
    assert service.health().healthy is False


def test_unexpected_errors_are_not_swallowed(service, store):
    """Only a store outage is recoverable by holding the price.

    A bad prediction means the system has a *wrong* opinion, which a held price
    cannot honestly paper over, so anything other than ModelStoreUnavailable
    must stay loud.
    """
    store.demand = lambda sku, price: (_ for _ in ()).throw(ValueError("corrupt artefact"))
    with pytest.raises(ValueError, match="corrupt artefact"):
        service.decide(make_request())


def test_model_store_unavailable_is_distinct_from_a_generic_error():
    assert issubclass(ModelStoreUnavailable, RuntimeError)


# ---------------------------------------------------------------------------
# API contracts
# ---------------------------------------------------------------------------


def test_decide_returns_a_price_and_a_reason_code(client):
    response = client.post("/decide", json=request_payload())
    assert response.status_code == 200

    body = response.json()
    assert body["recommended_price"] > 0
    assert body["degradation_reason_code"] == "OK_OPTIMAL"
    assert body["degradation_rung"] == 1
    assert body["sku"] == "SKU-001"


def test_response_carries_the_whole_record(client):
    """A recommendation without its working is a number to trust on faith, and
    this system's entire claim is that nobody should have to."""
    body = client.post("/decide", json=request_payload()).json()
    record = body["record"]
    assert record["candidates"], "no candidate scores were returned"
    assert "guardrail_results" in record
    assert record["seed"] is not None
    assert record["policy_version"] == "1.4.2"
    assert record["inputs"]["unit_cost"] == 15.0


def test_every_candidate_reports_its_feasibility(client):
    for candidate in client.post("/decide", json=request_payload()).json()["record"]["candidates"]:
        assert "feasible" in candidate
        assert "j_score" in candidate
        assert "cvar_alpha" in candidate


def test_decision_ids_are_unique(client):
    first = client.post("/decide", json=request_payload()).json()["decision_id"]
    second = client.post("/decide", json=request_payload()).json()["decision_id"]
    assert first != second


def test_batch_returns_one_decision_per_request_in_order(client):
    payload = {"requests": [request_payload(sku=f"SKU-{i:03d}") for i in range(4)]}
    body = client.post("/decide/batch", json=payload).json()

    assert body["count"] == 4
    assert [d["sku"] for d in body["decisions"]] == ["SKU-000", "SKU-001", "SKU-002", "SKU-003"]


def test_empty_batch_is_rejected(client):
    assert client.post("/decide/batch", json={"requests": []}).status_code == 422


def test_malformed_request_is_a_422_not_a_500(client):
    assert client.post("/decide", json={"sku": "S"}).status_code == 422


def test_negative_price_is_rejected_by_the_contract(client):
    assert client.post("/decide", json=request_payload(current_price=-5.0)).status_code == 422


def test_customer_attributes_are_refused_at_the_boundary(client):
    """PP-G009 is structural: the request model forbids extras, so a caller
    cannot personalise a price even by accident."""
    response = client.post("/decide", json=request_payload(customer_id="C-123"))
    assert response.status_code == 422


def test_health_reports_the_compute_backends(client):
    body = client.get("/health").json()
    assert body["model_store_available"] is True
    assert "compute" in body
    assert "backends" in body["compute"]


def test_stale_competitor_surfaces_as_rung_two(client):
    stale = (AS_OF - timedelta(hours=72)).isoformat().replace("+00:00", "Z")
    body = client.post(
        "/decide",
        json=request_payload(competitor_price=31.0, competitor_observed_at=stale),
    ).json()
    assert body["degradation_reason_code"] == "WARN_STALE_COMPETITOR"
    assert body["degradation_rung"] == 2


def test_infeasible_request_surfaces_as_rung_four(client):
    body = client.post("/decide", json=request_payload(unit_cost=40.0)).json()
    assert body["degradation_reason_code"] == "FALLBACK_RULE_ENGINE"
    assert body["recommended_price"] == pytest.approx(30.0)


def test_promo_without_an_omnibus_anchor_fails_closed(client):
    """A regulatory constraint with missing data is a data failure. Passing it
    is how a non-compliant price ships."""
    body = client.post("/decide", json=request_payload(is_promo=True)).json()
    assert body["recommended_price"] == pytest.approx(30.0)
    assert body["degradation_reason_code"] == "FALLBACK_RULE_ENGINE"


def test_openapi_schema_is_served(client):
    schema = client.get("/openapi.json").json()
    assert "/decide" in schema["paths"]
    assert "/decide/batch" in schema["paths"]
    assert "/health" in schema["paths"]
