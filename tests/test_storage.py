"""
Storage tests.

The store's job is not "hold rows" — any table does that. It is to make three
things impossible: writing data that never passed a contract, losing the record
of what was loaded, and silently doubling a dataset when someone re-runs
yesterday's file. Those are what the tests below assert, in that order.

Fixtures build transactions from the synthetic generator rather than a real
extract, so the ingestion path is exercised end to end without a download.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from prismprice.data.contracts import DataContractViolation
from prismprice.data.sources import (
    UCI_ONLINE_RETAIL_II,
    drop_non_product_lines,
    load_uci_excel,
    transaction_contract,
)
from prismprice.data.synthetic import generate_panel, transactions_from_panel
from prismprice.storage import PriceStore


@pytest.fixture(scope="module")
def transactions() -> pd.DataFrame:
    return transactions_from_panel(generate_panel(n_skus=6, n_days=200, seed=3))


@pytest.fixture
def db():
    with PriceStore(":memory:") as instance:
        yield instance


# ---------------------------------------------------------------------------
# Unvalidated data cannot physically be in the store
# ---------------------------------------------------------------------------


def test_failing_batch_is_quarantined_before_any_write(db, transactions):
    """The order matters: validate, then write. Never write, then check."""
    broken = transactions.copy()
    broken.loc[broken.index[:5], "unit_price"] = -1.0

    with pytest.raises(DataContractViolation):
        db.ingest(broken, transaction_contract(), source="broken")

    assert db.summary()["transactions"] == 0, "a failing batch wrote rows"


def test_failed_batch_is_still_recorded(db, transactions):
    """A rejected load is evidence that an upstream export broke.

    Discarding it guarantees the next person rediscovers the same fault.
    """
    broken = transactions.copy()
    broken.loc[broken.index[:5], "unit_price"] = -1.0
    with pytest.raises(DataContractViolation):
        db.ingest(broken, transaction_contract(), source="broken")

    batches = db.batches()
    assert len(batches) == 1
    assert not bool(batches.iloc[0]["passed"])
    assert int(batches.iloc[0]["rows_written"]) == 0


def test_inspection_mode_still_writes_nothing(db, transactions):
    """``quarantine_on_failure=False`` is for looking, not for forcing through."""
    broken = transactions.copy()
    broken.loc[broken.index[:5], "unit_price"] = -1.0
    result = db.ingest(
        broken, transaction_contract(), source="broken", quarantine_on_failure=False
    )
    assert not result.report.passed
    assert result.rows_written == 0
    assert db.summary()["transactions"] == 0


def test_clean_batch_is_written(db, transactions):
    result = db.ingest(transactions, transaction_contract(), source="synthetic")
    assert result.report.passed
    assert result.rows_written == len(transactions)
    assert db.summary()["transactions"] == len(transactions)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_reingesting_the_same_file_writes_nothing(db, transactions):
    """Re-running yesterday's load is a mistake people make constantly.

    A store that doubles every quantity turns it into a demand spike nobody can
    explain, and the model downstream has no way to tell it from a real one.
    """
    db.ingest(transactions, transaction_contract(), source="first")
    second = db.ingest(transactions, transaction_contract(), source="second")

    assert second.rows_written == 0
    assert second.rows_duplicate == len(transactions)
    assert second.was_noop
    assert db.summary()["transactions"] == len(transactions)


def test_overlapping_batch_writes_only_the_new_rows(db, transactions):
    """A partial re-load is the common case, not a full one."""
    head = transactions.iloc[:400]
    db.ingest(head, transaction_contract(), source="head")
    result = db.ingest(transactions, transaction_contract(), source="full")

    assert result.rows_written == len(transactions) - len(head)
    assert db.summary()["transactions"] == len(transactions)


def test_noop_is_distinguishable_from_an_empty_file(db, transactions):
    """'0 written from 1181 offered' and '0 written from 0 offered' are
    different problems, so they must not report identically."""
    db.ingest(transactions, transaction_contract(), source="first")
    repeat = db.ingest(transactions, transaction_contract(), source="repeat")
    assert repeat.was_noop and repeat.rows_offered > 0


# ---------------------------------------------------------------------------
# The derived panel
# ---------------------------------------------------------------------------


def test_daily_panel_aggregates_transactions(db, transactions):
    db.ingest(transactions, transaction_contract(), source="synthetic")
    rows = db.rebuild_daily_demand()
    assert rows > 0

    panel = db.daily_panel()
    assert set(panel.columns) == {
        "sku", "date", "units", "revenue", "price", "list_price", "n_invoices"
    }
    assert (panel["units"] > 0).all()
    assert (panel["price"] > 0).all()


def test_panel_price_is_quantity_weighted(db):
    """A plain mean lets a one-unit line at an odd price move the daily price as
    much as a hundred-unit line at the real one."""
    frame = pd.DataFrame(
        {
            "invoice_id": ["A", "B"],
            "sku": ["S1", "S1"],
            "line_number": [0, 0],
            "quantity": [100, 1],
            "unit_price": [10.0, 100.0],
            "invoice_ts": [datetime(2024, 1, 1, 9, tzinfo=timezone.utc)] * 2,
            "customer_id": ["c1", "c2"],
        }
    )
    db.ingest(frame, transaction_contract(), source="weighted")
    db.rebuild_daily_demand()
    row = db.daily_panel().iloc[0]
    assert float(row["price"]) == pytest.approx((100 * 10.0 + 1 * 100.0) / 101)
    assert float(row["price"]) < 15.0, "an unweighted mean would report 55.0 here"


def test_list_price_is_the_undiscounted_price(db):
    """The treatment variable for elasticity, kept separate from realised price.

    This retailer runs a volume-discount ladder: within one SKU-day, log unit
    price and log line quantity correlate -0.67 across 88.7% of SKU-days. So the
    quantity-weighted price falls mechanically whenever a large order lands, and
    demand regressed on it recovers the discount schedule wearing the sign of an
    elasticity. The list price is what a small order pays, and switching to it
    cuts that mechanical correlation from -0.47 to -0.15 on the real data.
    """
    frame = pd.DataFrame(
        {
            "invoice_id": ["A", "B"],
            "sku": ["S1", "S1"],
            "line_number": [0, 0],
            "quantity": [100, 1],
            "unit_price": [10.0, 12.0],
            "invoice_ts": [datetime(2024, 1, 1, 9, tzinfo=timezone.utc)] * 2,
            "customer_id": ["c1", "c2"],
        }
    )
    db.ingest(frame, transaction_contract(), source="list")
    db.rebuild_daily_demand()
    row = db.daily_panel().iloc[0]
    assert float(row["list_price"]) == pytest.approx(12.0)
    assert float(row["price"]) < float(row["list_price"]), (
        "the bulk order must drag realised price below list, which is exactly "
        "the contamination list_price exists to avoid"
    )


def test_panel_as_of_is_strictly_before(db, transactions):
    """Same leakage rule L1 enforces: a row stamped at the decision moment is
    not knowable when the decision is made."""
    db.ingest(transactions, transaction_contract(), source="synthetic")
    db.rebuild_daily_demand()

    full = db.daily_panel()
    cut = full["date"].max()
    trimmed = db.daily_panel(as_of=cut)
    assert trimmed["date"].max() < cut


def test_panel_min_observations_drops_thin_skus(db, transactions):
    """A SKU with four observations cannot support an elasticity."""
    db.ingest(transactions, transaction_contract(), source="synthetic")
    db.rebuild_daily_demand()
    counts = db.daily_panel(min_observations=150).groupby("sku").size()
    assert (counts >= 150).all()


def test_panel_can_be_restricted_to_skus(db, transactions):
    db.ingest(transactions, transaction_contract(), source="synthetic")
    db.rebuild_daily_demand()
    panel = db.daily_panel(skus=["SKU-000", "SKU-001"])
    assert set(panel["sku"].unique()) <= {"SKU-000", "SKU-001"}


def test_timestamps_are_utc_not_machine_local(db, transactions):
    """The session time zone is pinned, so the same query renders identically on
    any machine. Without it a derived date can land on a different day."""
    db.ingest(transactions, transaction_contract(), source="synthetic")
    first = db.summary()["first_transaction"]
    assert first.utcoffset() == timedelta(0), f"expected UTC, got {first.tzinfo}"


# ---------------------------------------------------------------------------
# Decisions are immutable
# ---------------------------------------------------------------------------


def _decision(decision_id: str = "d-1") -> dict:
    return {
        "decision_id": decision_id,
        "sku": "SKU-000",
        "as_of": datetime(2026, 8, 17, 2, tzinfo=timezone.utc),
        "created_at": datetime(2026, 8, 17, 2, 0, 3, tzinfo=timezone.utc),
        "recommended_price": 30.95,
    }


def test_decision_is_recorded(db):
    db.record_decision(_decision())
    assert db.summary()["decisions"] == 1


def test_duplicate_decision_id_is_refused(db):
    """Corrections are new records linked by supersedes_id, never overwrites."""
    db.record_decision(_decision())
    with pytest.raises(ValueError, match="immutable"):
        db.record_decision(_decision())


def test_superseding_decision_is_a_new_row(db):
    db.record_decision(_decision("d-1"))
    db.record_decision(_decision("d-2"))
    assert db.summary()["decisions"] == 2


# ---------------------------------------------------------------------------
# Source handling
# ---------------------------------------------------------------------------


def test_non_product_lines_are_removed_and_counted():
    """A filter that removes a chunk of a dataset should have to say so."""
    frame = pd.DataFrame({"sku": ["85123A", "POST", "M", "22423", "post"]})
    kept, removed = drop_non_product_lines(frame)
    assert set(kept["sku"]) == {"85123A", "22423"}
    assert sum(removed.values()) == 3


def test_missing_dataset_explains_how_to_get_it():
    """A bare FileNotFoundError tells the reader nothing about a file that is
    deliberately not in the repository."""
    with pytest.raises(FileNotFoundError, match=r"archive\.ics\.uci\.edu"):
        load_uci_excel("does-not-exist.xlsx")


def test_source_spec_states_size_and_licence_before_download():
    text = UCI_ONLINE_RETAIL_II.describe()
    assert "44 MB" in text
    assert "CC BY 4.0" in text
    assert UCI_ONLINE_RETAIL_II.url.startswith("https://")


@pytest.mark.parametrize(
    ("release", "columns"),
    [
        (
            "online-retail-2011",
            ["InvoiceNo", "StockCode", "Description", "Quantity",
             "InvoiceDate", "UnitPrice", "CustomerID", "Country"],
        ),
        (
            "online-retail-II",
            ["Invoice", "StockCode", "Description", "Quantity",
             "InvoiceDate", "Price", "Customer ID", "Country"],
        ),
    ],
)
def test_both_uci_releases_normalise(release, columns):
    """The two releases name the same fields differently; handling only one
    fails silently on the other."""
    from prismprice.data.augment import normalise_uci

    raw = pd.DataFrame(
        [["A1", "85123A", "MUG", 6, "2010-12-01 08:26:00", 2.55, "17850", "UK"]],
        columns=columns,
    )
    out = normalise_uci(raw)
    assert out.loc[0, "invoice_id"] == "A1"
    assert out.loc[0, "unit_price"] == pytest.approx(2.55)
    assert out.loc[0, "customer_id"] == "17850"


def test_normalise_names_the_columns_it_could_not_find():
    from prismprice.data.augment import normalise_uci

    with pytest.raises(ValueError, match="unit_price"):
        normalise_uci(pd.DataFrame({"Invoice": ["A"], "StockCode": ["X"],
                                    "Quantity": [1], "InvoiceDate": ["2010-01-01"]}))


def test_guest_checkouts_keep_a_null_customer_id():
    """Collapsing them to a sentinel would create one synthetic mega-customer
    and corrupt every retention curve built on top."""
    from prismprice.data.augment import normalise_uci

    raw = pd.DataFrame(
        [["A1", "85123A", "MUG", 6, "2010-12-01 08:26:00", 2.55, None, "UK"]],
        columns=["Invoice", "StockCode", "Description", "Quantity",
                 "InvoiceDate", "Price", "Customer ID", "Country"],
    )
    assert pd.isna(normalise_uci(raw).loc[0, "customer_id"])


def test_cancellations_are_flagged_as_returns():
    from prismprice.data.augment import normalise_uci

    raw = pd.DataFrame(
        [["C536379", "85123A", "MUG", -6, "2010-12-01 08:26:00", 2.55, "17850", "UK"]],
        columns=["Invoice", "StockCode", "Description", "Quantity",
                 "InvoiceDate", "Price", "Customer ID", "Country"],
    )
    assert bool(normalise_uci(raw).loc[0, "is_return"])


def test_store_persists_to_disk(tmp_path, transactions):
    """An in-memory store proves nothing about a store you can come back to."""
    path = tmp_path / "nested" / "prism.duckdb"
    with PriceStore(path) as writer:
        writer.ingest(transactions, transaction_contract(), source="synthetic")
        writer.rebuild_daily_demand()

    with PriceStore(path) as reader:
        assert reader.summary()["transactions"] == len(transactions)
        assert not reader.daily_panel().empty
