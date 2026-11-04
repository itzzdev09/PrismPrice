"""Train, time-validate, and write the local historical demand bundle.

Run after the UCI transactions have been ingested and the daily panel rebuilt::

    python scripts/build_demand_bundle.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prismprice.estimation.bundle import build_demand_bundle
from prismprice.storage import PriceStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=Path, default=ROOT / "data" / "prismprice.duckdb"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts" / "demand-bundle"
    )
    parser.add_argument("--min-observations", type=int, default=300)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()

    if not args.database.is_file():
        parser.error(
            f"Database not found: {args.database}. Ingest transactions and rebuild the daily panel first."
        )

    with PriceStore(args.database) as store:
        panel = store.daily_panel(min_observations=args.min_observations)
    manifest = build_demand_bundle(
        panel,
        args.output,
        min_observations=args.min_observations,
        stride=args.stride,
        n_folds=args.folds,
    )
    print(
        json.dumps(
            {
                "model_name": manifest["model_name"],
                "model_version": manifest["model_version"],
                "supported_skus": len(manifest["supported_skus"]),
                "training_rows": manifest["training_rows"],
                "validation": manifest["validation"],
                "bundle_directory": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
