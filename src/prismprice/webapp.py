"""Run the PrismPrice browser workbench with a labelled historical data demo.

Start it with ``python -m prismprice.webapp`` after installing the ``serving``
extra. The demo uses product baselines and elasticity estimates from the saved
Online Retail II pipeline run, plus an explicitly illustrative demand spread.
It is a historical walkthrough rather than a live model; production
deployments should call ``create_app`` with a :class:`DecisionService` wired to
their fitted models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prismprice.decision import DecisionEngine
from prismprice.serving import DecisionService, InMemoryModelStore
from prismprice.serving.app import create_app

_WEB_DIR = Path(__file__).resolve().parent / "web"
_DEMO_DATA_PATH = _WEB_DIR / "demo-data.json"
_DEFAULT_DEMAND = {
    "baseline_units": 48.0,
    "current_price": 30.0,
    "elasticity": -1.35,
    "spread": 0.2,
}


def _sample_demand(
    sku: str,
    price: float,
    catalog: dict[str, dict[str, Any]] | None = None,
) -> tuple[float, float, float]:
    """Return a deterministic demo curve anchored to one historical product."""
    product = catalog.get(sku) if catalog else None
    baseline = product or _DEFAULT_DEMAND
    midpoint_price = float(baseline["current_price"])
    midpoint_units = float(baseline["baseline_units"])
    elasticity = float(baseline["elasticity"])
    spread = float(baseline["spread"])
    median = midpoint_units * (price / midpoint_price) ** elasticity
    return median * max(1.0 - spread, 0.05), median, median * (1.0 + spread)


def create_demo_app() -> Any:
    """Build the demo app from its compact, precomputed dataset snapshot."""
    if not _DEMO_DATA_PATH.exists():
        raise RuntimeError(
            "The Online Retail II demo snapshot is missing. Run "
            "python scripts/build_demo_snapshot.py from the repository root."
        )
    demo_data = json.loads(_DEMO_DATA_PATH.read_text(encoding="utf-8"))
    catalog = {product["sku"]: product for product in demo_data["products"]}
    store = InMemoryModelStore(
        demand=lambda sku, price: _sample_demand(sku, price, catalog),
        versions={"demand": "uci-online-retail-ii-demo-curves-v1"},
    )
    service = DecisionService(
        store=store,
        engine=DecisionEngine(
            policy_version="demo-uci-online-retail-ii-1.0",
            model_versions={"demand": "uci-online-retail-ii-demo-curves-v1"},
        ),
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
