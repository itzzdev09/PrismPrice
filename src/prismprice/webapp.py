"""Run the PrismPrice browser workbench with a clearly labelled sample model.

Start it with ``python -m prismprice.webapp`` after installing the ``serving``
extra. The built-in price response is a deterministic demonstration curve. It
is not trained on retail data; production deployments should call
``create_app`` with a :class:`DecisionService` wired to their fitted models.
"""

from __future__ import annotations

from typing import Any

from prismprice.decision import DecisionEngine
from prismprice.serving import DecisionService, InMemoryModelStore
from prismprice.serving.app import create_app


def _sample_demand(sku: str, price: float) -> tuple[float, float, float]:
    """A small, deterministic isoelastic curve for exploring the interface."""
    del sku  # A sample curve, deliberately shared by every example product.
    midpoint_price = 30.0
    midpoint_units = 48.0
    elasticity = -1.35
    median = midpoint_units * (price / midpoint_price) ** elasticity
    return median * 0.8, median, median * 1.2


def create_demo_app() -> Any:
    """Build the web app around a labelled local demonstration model."""
    store = InMemoryModelStore(
        demand=_sample_demand,
        versions={"demand": "sample-isoelastic-v1"},
    )
    service = DecisionService(
        store=store,
        engine=DecisionEngine(policy_version="demo-1.0"),
    )
    return create_app(
        service,
        allow_anonymous=True,
        trusted_hosts=("localhost", "127.0.0.1"),
        expose_docs=True,
    )


app = create_demo_app()


def main() -> None:
    """Serve locally; a real model store should use its own deployment entry."""
    import uvicorn

    uvicorn.run("prismprice.webapp:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
