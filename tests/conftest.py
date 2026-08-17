"""
Shared fixtures and Hypothesis configuration.

Hypothesis is a hard test dependency (``pip install -e ".[dev]"``). The previous
"fallback to random iteration if unavailable" branch was removed: it silently
downgraded the guardrail proof to 250 fixed-seed draws while the docs claimed
10,000 generated cases, which is the sort of gap this repo exists to avoid.
"""

from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import HealthCheck, settings

from prismprice.governance.guardrails import GuardrailEngine
from prismprice.governance.schemas import PriceRequest

# Governance is the one layer where an edge case is a published illegal price, so
# it gets a deeper search than the Hypothesis default of 100.
settings.register_profile(
    "default",
    max_examples=1_000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile("ci", max_examples=2_000, deadline=None)
settings.register_profile("fast", max_examples=100, deadline=None)
settings.load_profile("default")


AS_OF = datetime(2026, 8, 17, 2, 0, tzinfo=timezone.utc)


@pytest.fixture
def as_of() -> datetime:
    return AS_OF


@pytest.fixture
def sample_request() -> PriceRequest:
    """A healthy, fully-populated request that a sane price should satisfy."""
    return PriceRequest(
        sku="SKU-TEST-001",
        as_of=AS_OF,
        current_price=20.00,
        unit_cost=10.00,
        is_promo=False,
        competitor_price=19.50,
        competitor_observed_at=AS_OF - timedelta(hours=2),
        min_price_last_30d=18.00,
        inventory_cover_days=30.0,
        inventory_shadow_price=0.0,
        margin_floor_pct=0.15,
        movement_cap_pct=0.15,
        competitor_ceiling_multiplier=1.10,
        price_changes_in_window=1,
        max_changes_per_window=4,
        allowed_price_endings=None,
    )


@pytest.fixture
def engine() -> GuardrailEngine:
    return GuardrailEngine()
