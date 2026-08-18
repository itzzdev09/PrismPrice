"""
DuckDB-backed store (L0).

Until now PrismPrice had no database. Panels were built in memory, handed
between layers as DataFrames, and forgotten. That is fine for a test rig and
useless as a product: there was nowhere for real transactions to live, no record
of what had been loaded, and no way to ask what the data looked like on the day
a decision was made.

Three properties make this more than a table with a connection attached.

**Contract enforcement is at the write boundary, not beside it.** ``ingest()``
validates against the source contract and raises before touching the database,
so unvalidated rows cannot physically be in the store. A caller who wants the
data anyway has to quarantine it explicitly; there is no path that writes first
and checks later, because that path is how bad data reaches a price.

**Every row knows which batch loaded it, and every batch keeps its report.**
``ingest_batches`` holds the source, the timestamp, the row counts and the full
quality report as JSON — including for batches that *failed*. A rejected load is
evidence, and deleting it means the next person rediscovers the same broken
export. Reconstructing a decision months later needs the data as it was, which
needs to know what arrived and when.

**Ingestion is idempotent.** Loading the same file twice is a mistake people
make constantly, and a store that silently doubles every quantity turns it into
a demand spike nobody can explain. Writes are anti-joined against the primary
key, so a repeated load inserts nothing and says so.

DuckDB rather than SQLite or Postgres: the workload is analytical (scan a panel,
aggregate by SKU and date), it is a single embedded file with no server, and the
SQL is close enough to Postgres that moving a deployment there is a connection
string rather than a rewrite (README §8).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from prismprice.data.contracts import DataContractViolation, QualityReport, SourceContract

__all__ = [
    "IngestResult",
    "PriceStore",
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_batches (
    batch_id      VARCHAR PRIMARY KEY,
    source        VARCHAR NOT NULL,
    ingested_at   TIMESTAMPTZ NOT NULL,
    rows_offered  BIGINT NOT NULL,
    rows_written  BIGINT NOT NULL,
    rows_duplicate BIGINT NOT NULL,
    passed        BOOLEAN NOT NULL,
    report_json   VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS transactions (
    invoice_id  VARCHAR NOT NULL,
    sku         VARCHAR NOT NULL,
    line_number BIGINT  NOT NULL,
    quantity    BIGINT  NOT NULL,
    unit_price  DOUBLE  NOT NULL,
    invoice_ts  TIMESTAMPTZ NOT NULL,
    customer_id VARCHAR,
    country     VARCHAR,
    description VARCHAR,
    is_return   BOOLEAN NOT NULL DEFAULT FALSE,
    batch_id    VARCHAR NOT NULL,
    PRIMARY KEY (invoice_id, sku, line_number)
);

CREATE TABLE IF NOT EXISTS daily_demand (
    sku         VARCHAR NOT NULL,
    date        DATE    NOT NULL,
    units       DOUBLE  NOT NULL,
    revenue     DOUBLE  NOT NULL,
    price       DOUBLE  NOT NULL,
    list_price  DOUBLE  NOT NULL,
    n_invoices  BIGINT  NOT NULL,
    batch_id    VARCHAR NOT NULL,
    PRIMARY KEY (sku, date)
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id VARCHAR PRIMARY KEY,
    sku         VARCHAR NOT NULL,
    as_of       TIMESTAMPTZ NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL,
    record_json VARCHAR NOT NULL
);
"""


@dataclass(frozen=True)
class IngestResult:
    """Outcome of one ingest, including the parts that did nothing.

    ``rows_duplicate`` is reported rather than swallowed: "I loaded 500k rows and
    0 were written" is the signal that someone re-ran yesterday's file, and it
    only reads as a signal if the number is visible.
    """

    batch_id: str
    source: str
    rows_offered: int
    rows_written: int
    rows_duplicate: int
    report: QualityReport

    @property
    def was_noop(self) -> bool:
        return self.rows_written == 0 and self.rows_offered > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "source": self.source,
            "rows_offered": self.rows_offered,
            "rows_written": self.rows_written,
            "rows_duplicate": self.rows_duplicate,
            "passed": self.report.passed,
        }


class PriceStore:
    """Embedded analytical store for transactions, panels and decisions.

    Args:
        path: Database file. ``":memory:"`` gives an ephemeral store, which is
            what the tests use.

    Use as a context manager, or call :meth:`close`.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        import duckdb

        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = duckdb.connect(self.path)
        # DuckDB renders TIMESTAMPTZ in the session time zone, which defaults to
        # the machine's. Left alone, the same query returns 17:30 IST here and
        # 12:00 UTC on a CI runner — same instant, different rendering, and any
        # date derived from it can land on a different day. Reproducibility is a
        # stated guarantee (README §5), so the session zone is part of the
        # contract rather than the operator's locale.
        self._connection.execute("SET TimeZone='UTC'")
        self._connection.execute(_SCHEMA)

    def __enter__(self) -> PriceStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    # -- ingestion ---------------------------------------------------------

    def ingest(
        self,
        frame: pd.DataFrame,
        contract: SourceContract,
        source: str,
        as_of: datetime | None = None,
        quarantine_on_failure: bool = True,
    ) -> IngestResult:
        """Validate *frame* against *contract*, then write what is new.

        The order is the point. Validation happens first and, on failure with
        ``quarantine_on_failure``, raises before any row is written — so the
        store cannot contain rows that never passed a gate.

        Args:
            frame: Rows to load.
            contract: The contract to enforce.
            source: Human name for the batch, e.g. ``"uci-online-retail-ii"``.
            as_of: Freshness reference; defaults to now.
            quarantine_on_failure: Raise on a failing report instead of writing.
                Setting this ``False`` records the batch as failed and still
                writes nothing — it is for inspecting a bad export, not for
                forcing it through.

        Returns:
            :class:`IngestResult`.

        Raises:
            DataContractViolation: the batch failed its contract and
                ``quarantine_on_failure`` is set.
        """
        stamp = as_of or datetime.now(timezone.utc)
        report = contract.validate_frame(frame, as_of=stamp)
        batch_id = str(uuid4())

        if not report.passed:
            self._record_batch(batch_id, source, stamp, len(frame), 0, 0, report)
            if quarantine_on_failure:
                raise DataContractViolation(report)
            return IngestResult(batch_id, source, len(frame), 0, 0, report)

        written, duplicate = self._write_transactions(frame, batch_id)
        self._record_batch(batch_id, source, stamp, len(frame), written, duplicate, report)
        return IngestResult(batch_id, source, len(frame), written, duplicate, report)

    def _write_transactions(self, frame: pd.DataFrame, batch_id: str) -> tuple[int, int]:
        """Insert rows whose primary key is not already present.

        The anti-join is what makes a repeated load a no-op rather than a
        doubling. It also means an interrupted ingest can simply be re-run.
        """
        # Absent optional columns are filled with a typed default rather than
        # None. `is_return` is NOT NULL with a DEFAULT, and an explicit None
        # overrides the default rather than falling back to it — which fails the
        # constraint on any extract that simply does not carry the column.
        defaults: dict[str, Any] = {
            "customer_id": None,
            "country": None,
            "description": None,
            "is_return": False,
        }
        columns = [
            "invoice_id", "sku", "line_number", "quantity", "unit_price",
            "invoice_ts", "customer_id", "country", "description", "is_return",
        ]
        staged = frame.copy()
        for column in columns:
            if column not in staged.columns:
                staged[column] = defaults.get(column)
        staged["is_return"] = staged["is_return"].fillna(False).astype(bool)
        staged = staged[columns]
        staged["batch_id"] = batch_id

        self._connection.register("staging", staged)
        before = self._scalar("SELECT count(*) FROM transactions")
        self._connection.execute(
            """
            INSERT INTO transactions
            SELECT s.* FROM staging s
            WHERE NOT EXISTS (
                SELECT 1 FROM transactions t
                WHERE t.invoice_id = s.invoice_id
                  AND t.sku = s.sku
                  AND t.line_number = s.line_number
            )
            """
        )
        after = self._scalar("SELECT count(*) FROM transactions")
        self._connection.unregister("staging")

        written = after - before
        return written, len(staged) - written

    def _record_batch(
        self,
        batch_id: str,
        source: str,
        stamp: datetime,
        offered: int,
        written: int,
        duplicate: int,
        report: QualityReport,
    ) -> None:
        """Record the batch, passing or failing.

        Failed batches are kept deliberately. A rejected load is the evidence
        that an upstream export broke, and discarding it guarantees the next
        person rediscovers the same fault from scratch.
        """
        self._connection.execute(
            "INSERT INTO ingest_batches VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                batch_id,
                source,
                stamp,
                offered,
                written,
                duplicate,
                report.passed,
                json.dumps(report.model_dump(mode="json")),
            ],
        )

    # -- derived panel -----------------------------------------------------

    def rebuild_daily_demand(self, batch_id: str | None = None) -> int:
        """Aggregate transactions into the ``(sku, date)`` panel.

        **Two prices are recorded, because they answer different questions.**

        ``price`` is realised revenue over units — the quantity-weighted average
        actually taken. It is the right number for revenue and margin, and the
        wrong number to regress demand on. A plain unweighted mean is worse
        still: it lets a single one-unit line at an odd price move the day as
        much as a hundred-unit line at the real one.

        ``list_price`` is the highest unit price seen that day, which on this
        retailer's data is the undiscounted price — what a customer buying a
        small quantity pays. **This is the treatment variable for elasticity.**

        The distinction is not fastidiousness; it was forced by the data. This
        retailer runs a volume-discount ladder, so within a single SKU-day the
        correlation between log unit price and log line quantity is -0.67, and
        88.7% of SKU-days show it. Most of the apparent daily "price variation"
        is therefore order-size mix: a large wholesale order mechanically lowers
        the weighted price and raises units on the same day. Regressing demand
        on that recovers the discount schedule with the sign of an elasticity,
        and a pricing system would read it as "customers are extremely price
        sensitive" and cut prices on the strength of its own billing rules.
        Switching to the list price cuts the mechanical correlation from -0.47
        to -0.15.

        Returns are netted rather than dropped — a return is real information
        about demand, and discarding it inflates the series.
        """
        marker = batch_id or str(uuid4())
        self._connection.execute("DELETE FROM daily_demand")
        self._connection.execute(
            """
            INSERT INTO daily_demand
            SELECT
                sku,
                CAST(invoice_ts AS DATE) AS date,
                SUM(quantity)                        AS units,
                SUM(quantity * unit_price)           AS revenue,
                SUM(quantity * unit_price) / NULLIF(SUM(quantity), 0) AS price,
                MAX(unit_price)                      AS list_price,
                COUNT(DISTINCT invoice_id)           AS n_invoices,
                ? AS batch_id
            FROM transactions
            WHERE unit_price > 0
            GROUP BY sku, CAST(invoice_ts AS DATE)
            HAVING SUM(quantity) > 0
            """,
            [marker],
        )
        return int(self._scalar("SELECT count(*) FROM daily_demand"))

    def daily_panel(
        self,
        skus: Sequence[str] | None = None,
        as_of: datetime | None = None,
        min_observations: int = 0,
    ) -> pd.DataFrame:
        """Return the modelling panel.

        Args:
            skus: Restrict to these SKUs.
            as_of: Point-in-time cut — rows strictly *before* this timestamp.
                The strict inequality is the same leakage rule L1 enforces: a
                row stamped exactly at the decision moment is not knowable when
                the decision is made.
            min_observations: Drop SKUs with fewer than this many days. A SKU
                with four observations cannot support an elasticity, and letting
                it through produces a confident number from nothing.
        """
        query = (
            "SELECT sku, date, units, revenue, price, list_price, n_invoices "
            "FROM daily_demand"
        )
        clauses: list[str] = []
        params: list[Any] = []

        if skus is not None:
            placeholders = ", ".join("?" for _ in skus)
            clauses.append(f"sku IN ({placeholders})")
            params.extend(skus)
        if as_of is not None:
            clauses.append("date < ?")
            params.append(as_of)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY sku, date"

        frame = self._connection.execute(query, params).df()

        if min_observations > 0 and not frame.empty:
            counts = frame.groupby("sku")["date"].transform("count")
            frame = frame[counts >= min_observations].reset_index(drop=True)

        if not frame.empty:
            frame["date"] = pd.to_datetime(frame["date"], utc=True)
        return frame

    # -- decisions ---------------------------------------------------------

    def record_decision(self, record: dict[str, Any]) -> None:
        """Append an immutable decision record.

        Rejects a duplicate ``decision_id`` rather than overwriting: a decision
        is never mutated, and a correction is a new record linked by
        ``supersedes_id`` (architecture.md §2).
        """
        decision_id = record["decision_id"]
        if self._scalar(
            "SELECT count(*) FROM decisions WHERE decision_id = ?", [decision_id]
        ):
            raise ValueError(
                f"decision {decision_id} already recorded; decisions are immutable, "
                f"supersede it with a new record instead"
            )
        self._connection.execute(
            "INSERT INTO decisions VALUES (?, ?, ?, ?, ?)",
            [
                decision_id,
                record["sku"],
                record["as_of"],
                record.get("created_at", datetime.now(timezone.utc)),
                json.dumps(record, default=str),
            ],
        )

    # -- introspection -----------------------------------------------------

    def batches(self) -> pd.DataFrame:
        """Ingest history, newest first. The lineage record."""
        return self._connection.execute(
            "SELECT batch_id, source, ingested_at, rows_offered, rows_written, "
            "rows_duplicate, passed FROM ingest_batches ORDER BY ingested_at DESC"
        ).df()

    def summary(self) -> dict[str, Any]:
        """Row counts and date span, for a load to be checkable at a glance."""
        span = self._connection.execute(
            "SELECT min(invoice_ts), max(invoice_ts) FROM transactions"
        ).fetchone()
        return {
            "path": self.path,
            "transactions": self._scalar("SELECT count(*) FROM transactions"),
            "skus": self._scalar("SELECT count(DISTINCT sku) FROM transactions"),
            "customers": self._scalar(
                "SELECT count(DISTINCT customer_id) FROM transactions "
                "WHERE customer_id IS NOT NULL"
            ),
            "daily_rows": self._scalar("SELECT count(*) FROM daily_demand"),
            "batches": self._scalar("SELECT count(*) FROM ingest_batches"),
            "decisions": self._scalar("SELECT count(*) FROM decisions"),
            "first_transaction": span[0] if span else None,
            "last_transaction": span[1] if span else None,
        }

    def _scalar(self, query: str, params: Sequence[Any] | None = None) -> int:
        row = self._connection.execute(query, list(params or [])).fetchone()
        return int(row[0]) if row and row[0] is not None else 0
