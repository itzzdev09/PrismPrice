"""Run the local workbench with a time-validated historical demand bundle.

Create the bundle with ``python scripts/build_demand_bundle.py`` after preparing
the local Online Retail II DuckDB panel. This remains a historical demonstration
with explicit source-data limitations, not a current-market pricing service.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from prismprice.decision import DecisionEngine
from prismprice.serving import DecisionService, UnavailableModelStore
from prismprice.serving.app import create_app
from prismprice.serving.repository import LockedPriceStoreDecisionRepository

logger = logging.getLogger(__name__)

_WEB_DIR = Path(__file__).resolve().parent / "web"

_ROOT = Path(__file__).resolve().parents[2]


def create_demo_app() -> Any:
    """Build the UI/API around a verified model bundle and local decision ledger."""
    database_path = Path(os.getenv("PRISMPRICE_DB_PATH", _ROOT / "data" / "prismprice.duckdb"))
    bundle_path = Path(
        os.getenv("PRISMPRICE_MODEL_BUNDLE", _ROOT / "artifacts" / "demand-bundle")
    )
    price_store = None
    repository = None
    model_store: Any

    if database_path.is_file():
        try:
            from prismprice.storage import PriceStore

            price_store = PriceStore(database_path)
            repository = LockedPriceStoreDecisionRepository(price_store)
        except Exception:
            logger.exception("Could not open the local PrismPrice decision store")
            price_store = None
            repository = None

    try:
        if price_store is None:
            raise RuntimeError(f"Local panel database not found: {database_path}")
        from prismprice.serving.demand_store import HistoricalDemandModelStore

        bundle_module = __import__(
            "prismprice.estimation.bundle", fromlist=["load_demand_bundle"]
        )
        model, manifest = bundle_module.load_demand_bundle(bundle_path)
        # Verify against the same full positive-sales panel that was hashed at
        # training time. The adapter filters to supported SKUs after verification.
        panel = price_store.daily_panel()
        model_store = HistoricalDemandModelStore(model=model, manifest=manifest, panel=panel)
    except Exception as exc:
        logger.warning("Historical demand model unavailable: %s", exc)
        model_store = UnavailableModelStore(
            f"{exc} Run scripts/build_demand_bundle.py after preparing the local dataset."
        )

    service = DecisionService(
        store=model_store,
        engine=DecisionEngine(
            policy_version="historical-demand-workbench-0.2.0",
            model_versions=(
                model_store.model_versions()
                if not isinstance(model_store, UnavailableModelStore)
                else {}
            ),
        ),
        repository=repository,
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
