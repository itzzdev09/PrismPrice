"""
Hypothesis strategies for governance property tests.
"""

from datetime import datetime, timezone

from hypothesis import strategies as st

from prismprice.governance.schemas import PriceRequest

AS_OF = datetime(2026, 8, 17, 2, 0, tzinfo=timezone.utc)

money = st.floats(
    min_value=0.01, max_value=10_000.0, allow_nan=False, allow_infinity=False, width=64
)
ratio = st.floats(min_value=0.0, max_value=0.50, allow_nan=False, allow_infinity=False)


@st.composite
def price_requests(draw, is_promo=None, with_endings: bool = False) -> PriceRequest:
    """Draw a valid PriceRequest across the full parameter space.

    Price endings are disabled by default. Left on, the ladder constraint rejects
    ~98% of randomly drawn prices, which would make every "if feasible then ..."
    property vacuously true and hide real violations behind an unreachable
    feasible set.
    """
    unit_cost = draw(money)
    promo = draw(st.booleans()) if is_promo is None else is_promo
    return PriceRequest(
        sku=draw(
            st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-", min_size=1, max_size=12)
        ),
        as_of=AS_OF,
        current_price=draw(money),
        unit_cost=unit_cost,
        is_promo=promo,
        min_price_last_30d=draw(st.one_of(st.none(), money)),
        competitor_price=draw(st.one_of(st.none(), money)),
        competitor_ceiling_multiplier=draw(
            st.floats(min_value=1.0, max_value=2.0, allow_nan=False, allow_infinity=False)
        ),
        inventory_cover_days=draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=120.0))),
        min_inventory_cover_days=draw(st.floats(min_value=0.0, max_value=60.0)),
        inventory_shadow_price=draw(st.floats(min_value=0.0, max_value=100.0)),
        margin_floor_pct=draw(ratio),
        movement_cap_pct=draw(ratio),
        price_changes_in_window=draw(st.one_of(st.none(), st.integers(0, 10))),
        max_changes_per_window=draw(st.integers(0, 10)),
        allowed_price_endings=(95, 99) if with_endings else None,
    )


@st.composite
def request_and_price(draw) -> tuple[PriceRequest, float]:
    """A request plus a candidate price drawn near its feasible region.

    Sampling the candidate as a multiple of unit cost rather than independently
    keeps a useful share of draws inside the feasible set, so conditional
    properties are actually exercised instead of passing vacuously.
    """
    request = draw(price_requests())
    multiple = draw(st.floats(min_value=0.5, max_value=3.0, allow_nan=False, allow_infinity=False))
    return request, max(request.unit_cost * multiple, 0.01)
