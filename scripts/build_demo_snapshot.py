"""Build the small, browser-ready summary for the Online Retail II demo.

The full transaction history stays in DuckDB. This script writes only
aggregated monthly, product and model-review data into the web package; it does
not export customer identifiers or transaction-level rows.

Run from the repository root after the real data pipeline has populated
``data/prismprice.duckdb`` and ``data/runs/real_run.json``::

    python scripts/build_demo_snapshot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "data" / "prismprice.duckdb"
RUN_REPORT = ROOT / "data" / "runs" / "real_run.json"
OUTPUT = ROOT / "src" / "prismprice" / "web" / "demo-data.json"

KAGGLE_URL = "https://www.kaggle.com/datasets/lakshmi25npathi/online-retail-dataset"
UCI_URL = "https://archive.ics.uci.edu/dataset/502/online+retail+ii"
UCI_CITATION = "Chen, D. (2012). Online Retail II. UCI Machine Learning Repository. DOI: 10.24432/C5CG6D."


def _clean(value: Any) -> Any:
    """Convert DuckDB/date values to JSON-safe primitives."""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, float):
        return round(value, 6)
    return value


def build_snapshot() -> dict[str, Any]:
    if not DATABASE.exists() or not RUN_REPORT.exists():
        raise FileNotFoundError(
            "Build the UCI Online Retail II data store and real pipeline report first. "
            f"Expected {DATABASE} and {RUN_REPORT}."
        )

    run = json.loads(RUN_REPORT.read_text(encoding="utf-8"))
    decision_by_sku = {row["sku"]: row for row in run["decisions"]}
    as_of = run["as_of"][:10]

    connection = duckdb.connect(str(DATABASE), read_only=True)
    try:
        connection.execute("SET TimeZone='UTC'")
        data = connection.execute(
            """
            SELECT
                count(*) AS prepared_rows,
                min(invoice_ts)::DATE AS first_date,
                max(invoice_ts)::DATE AS last_date,
                count(DISTINCT sku) AS skus,
                count(DISTINCT customer_id) AS customers,
                count(DISTINCT invoice_id) AS invoices,
                count(DISTINCT country) AS countries,
                count(*) FILTER (WHERE customer_id IS NULL) AS missing_customer_rows,
                count(*) FILTER (WHERE is_return) AS return_rows
            FROM transactions
            """
        ).fetchone()
        panel = connection.execute(
            """
            SELECT
                count(*) AS panel_rows,
                count(DISTINCT sku) AS panel_skus,
                sum(units) AS net_units,
                sum(revenue) AS net_revenue
            FROM daily_demand
            """
        ).fetchone()
        monthly = connection.execute(
            """
            SELECT
                strftime(date, '%Y-%m') AS month,
                sum(units) AS units,
                sum(revenue) AS revenue
            FROM daily_demand
            GROUP BY month
            ORDER BY month
            """
        ).fetchall()
        country_rows = connection.execute(
            """
            SELECT country, count(*) AS lines
            FROM transactions
            WHERE country IS NOT NULL
            GROUP BY country
            ORDER BY lines DESC
            LIMIT 5
            """
        ).fetchall()
        ranked_panel = connection.execute(
            """
            WITH eligible AS (
                SELECT sku, date, units, revenue, list_price
                FROM daily_demand
                WHERE date < ?::DATE
            ),
            recent_rows AS (
                SELECT *, row_number() OVER (PARTITION BY sku ORDER BY date DESC) AS recency
                FROM eligible
            ),
            recent AS (
                SELECT
                    sku,
                    avg(units) AS baseline_units,
                    stddev_samp(units) / nullif(avg(units), 0) AS relative_spread,
                    arg_max(list_price, date) AS current_price
                FROM recent_rows
                WHERE recency <= 28
                GROUP BY sku
            ),
            totals AS (
                SELECT
                    sku,
                    count(*) AS observations,
                    min(date) AS first_seen,
                    max(date) AS last_seen,
                    sum(units) AS observed_units,
                    sum(revenue) AS observed_revenue
                FROM eligible
                GROUP BY sku
            ),
            descriptions AS (
                SELECT sku, any_value(description) AS description
                FROM transactions
                WHERE description IS NOT NULL
                GROUP BY sku
            )
            SELECT
                t.sku, d.description, t.observations, t.first_seen, t.last_seen,
                t.observed_units, t.observed_revenue, r.current_price,
                r.baseline_units, r.relative_spread
            FROM totals t
            JOIN recent r USING (sku)
            LEFT JOIN descriptions d USING (sku)
            WHERE t.sku IN (SELECT unnest(?))
            ORDER BY t.observed_revenue DESC
            """,
            [as_of, list(decision_by_sku)],
        ).fetchall()
    finally:
        connection.close()

    products = []
    for row in ranked_panel:
        (
            sku,
            description,
            observations,
            first_seen,
            last_seen,
            observed_units,
            observed_revenue,
            panel_price,
            baseline_units,
            relative_spread,
        ) = row
        decision = decision_by_sku[sku]
        spread = relative_spread if relative_spread is not None else 0.4
        products.append(
            {
                "sku": sku,
                "description": description or "Unlabelled product",
                "current_price": _clean(decision["current_price"] or panel_price),
                "unit_cost": _clean(decision["unit_cost"]),
                "baseline_units": _clean(max(baseline_units or 0.1, 0.1)),
                "elasticity": _clean(decision["elasticity"]),
                "elasticity_confidence": decision["elasticity_confidence"],
                "spread": _clean(min(max(spread, 0.15), 1.2)),
                "observations": int(observations),
                "first_seen": _clean(first_seen),
                "last_seen": _clean(last_seen),
                "observed_units": _clean(observed_units),
                "observed_revenue": _clean(observed_revenue),
            }
        )

    elasticity = run["elasticity_summary"]
    high_confidence = int(round(elasticity["share_usable"] * run["n_skus_considered"]))
    movement = next(
        (item for item in run["breakers"] if item["name"] == "aggregate_movement"), None
    )
    monthly_data = [
        {"month": month, "units": _clean(units), "revenue": _clean(revenue)}
        for month, units, revenue in monthly
    ]
    november_2011 = next((item for item in monthly_data if item["month"] == "2011-11"), None)
    october_2011 = next((item for item in monthly_data if item["month"] == "2011-10"), None)
    autumn_lift = (
        november_2011["revenue"] / october_2011["revenue"] - 1
        if november_2011 and october_2011 and october_2011["revenue"]
        else None
    )

    return {
        "dataset": {
            "name": "Online Retail II",
            "source_rows": 1_067_371,
            "prepared_rows": int(data[0]),
            "first_date": _clean(data[1]),
            "last_date": _clean(data[2]),
            "skus": int(data[3]),
            "customers": int(data[4]),
            "invoices": int(data[5]),
            "countries": int(data[6]),
            "missing_customer_rows": int(data[7]),
            "return_rows": int(data[8]),
            "panel_rows": int(panel[0]),
            "panel_skus": int(panel[1]),
            "net_units": _clean(panel[2]),
            "net_revenue_gbp": _clean(panel[3]),
            "modelable_skus": int(run["n_skus_considered"]),
            "model_panel_rows": int(run["n_panel_rows"]),
            "decision_as_of": as_of,
            "currency": "GBP",
            "kaggle_url": KAGGLE_URL,
            "uci_url": UCI_URL,
            "citation": UCI_CITATION,
            "license": "CC BY 4.0 (UCI Machine Learning Repository)",
        },
        "monthly": monthly_data,
        "countries": [
            {"country": country, "lines": int(lines)} for country, lines in country_rows
        ],
        "elasticity": {
            "n": int(elasticity["n"]),
            "median": _clean(elasticity["median"]),
            "p10": _clean(elasticity["p10"]),
            "p90": _clean(elasticity["p90"]),
            "share_usable": _clean(elasticity["share_usable"]),
            "high_confidence": high_confidence,
            "low_confidence": int(run["n_skus_considered"] - high_confidence),
        },
        "pipeline": {
            "may_publish": bool(run["may_publish"]),
            "decisions": int(run["n_decisions"]),
            "aggregate_movement_pct": _clean(movement["observed"] * 100) if movement else None,
            "aggregate_movement_limit_pct": _clean(movement["limit"] * 100) if movement else None,
            "november_revenue_lift_vs_october_pct": _clean(autumn_lift * 100)
            if autumn_lift is not None
            else None,
            "notes": run["notes"],
        },
        "products": products,
        "top_products": products[:8],
    }


def main() -> None:
    snapshot = build_snapshot()
    OUTPUT.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} with {len(snapshot['products'])} modelled products.")


if __name__ == "__main__":
    main()
