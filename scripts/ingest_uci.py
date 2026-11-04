"""Validate and load the UCI Online Retail II workbook into local DuckDB.

Download and extract ``online_retail_II.xlsx`` from the UCI source, then run::

    python scripts/ingest_uci.py path/to/online_retail_II.xlsx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prismprice.data.augment import normalise_uci
from prismprice.data.sources import (
    clean_uci_transactions,
    drop_non_product_lines,
    load_uci_excel,
    transaction_contract,
)
from prismprice.storage import PriceStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path, help="Extracted UCI .xlsx workbook or .csv")
    parser.add_argument(
        "--database", type=Path, default=ROOT / "data" / "prismprice.duckdb"
    )
    args = parser.parse_args()

    raw = load_uci_excel(args.workbook)
    transactions = normalise_uci(raw)
    transactions, cleaning = clean_uci_transactions(transactions)
    transactions, non_products = drop_non_product_lines(transactions)

    with PriceStore(args.database) as store:
        result = store.ingest(
            transactions,
            transaction_contract(),
            source="uci-online-retail-ii",
        )
        daily_rows = store.rebuild_daily_demand(batch_id=result.batch_id)

    print(
        json.dumps(
            {
                "database": str(args.database.resolve()),
                "ingest": result.as_dict(),
                "daily_panel_rows": daily_rows,
                "cleaning_counts": cleaning,
                "removed_non_product_lines": sum(non_products.values()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
