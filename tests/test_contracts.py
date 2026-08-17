"""
Data contract and quality gate tests (L0).

The gate's job is to fail loudly at the boundary. These tests are mostly about
proving it *does* fail — a validator that passes everything is indistinguishable
from no validator until the day it matters.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from prismprice.data.contracts import (
    CHECK_REGISTRY,
    TRANSACTIONS_CONTRACT,
    ColumnContract,
    DataContractViolation,
    DType,
    Severity,
    SourceContract,
    register_check,
    validate,
)
from prismprice.data.contracts import ViolationCode as VC

AS_OF = datetime(2026, 8, 17, 2, 0, tzinfo=timezone.utc)


def make_transactions(rows: int = 3, **overrides) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "invoice_id": pd.Series([f"INV-{i}" for i in range(rows)], dtype="string"),
            "line_number": pd.Series(range(rows), dtype="int64"),
            "sku": pd.Series([f"SKU-{i:03d}" for i in range(rows)], dtype="string"),
            "quantity": pd.Series([2] * rows, dtype="int64"),
            "unit_price": pd.Series([9.99] * rows, dtype="float64"),
            "invoice_ts": pd.Series([AS_OF - timedelta(hours=1)] * rows),
            "customer_id": pd.Series(["C-1"] * rows, dtype="string"),
        }
    )
    for key, value in overrides.items():
        df[key] = value
    return df


def codes(report) -> set:
    return {v.code for v in report.violations}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_clean_batch_passes():
    report = validate(make_transactions(), TRANSACTIONS_CONTRACT, as_of=AS_OF)
    assert report.passed, report.summary()
    assert report.violations == ()
    assert report.rows == 3


def test_enforce_returns_the_frame_when_clean():
    df = make_transactions()
    assert TRANSACTIONS_CONTRACT.enforce(df, as_of=AS_OF) is df


def test_guest_checkouts_are_allowed_to_have_no_customer():
    """Nullable customer_id is deliberate — see the contract's comment."""
    df = make_transactions()
    df["customer_id"] = pd.Series([None, None, "C-3"], dtype="string")
    assert validate(df, TRANSACTIONS_CONTRACT, as_of=AS_OF).passed


# ---------------------------------------------------------------------------
# Each failure mode fires, and fires distinguishably
# ---------------------------------------------------------------------------


def test_missing_column_is_reported():
    df = make_transactions().drop(columns=["unit_price"])
    report = validate(df, TRANSACTIONS_CONTRACT, as_of=AS_OF)
    assert VC.MISSING_COLUMN in codes(report)
    assert not report.passed


def test_wrong_dtype_is_reported():
    df = make_transactions()
    df["quantity"] = df["quantity"].astype(str)
    assert VC.WRONG_DTYPE in codes(validate(df, TRANSACTIONS_CONTRACT, as_of=AS_OF))


def test_null_in_non_nullable_column_is_reported():
    df = make_transactions()
    df.loc[0, "sku"] = None
    assert VC.UNEXPECTED_NULL in codes(validate(df, TRANSACTIONS_CONTRACT, as_of=AS_OF))


def test_out_of_range_values_are_reported_with_a_sample():
    df = make_transactions()
    df.loc[0, "quantity"] = 99_999
    report = validate(df, TRANSACTIONS_CONTRACT, as_of=AS_OF)
    violation = next(v for v in report.violations if v.code is VC.OUT_OF_RANGE)
    assert violation.row_count == 1
    assert violation.sample  # the offending value travels with the report


def test_duplicate_primary_key_is_reported():
    df = make_transactions(rows=2)
    df.loc[1, ["invoice_id", "line_number", "sku"]] = df.loc[
        0, ["invoice_id", "line_number", "sku"]
    ].to_numpy()
    assert VC.DUPLICATE_PRIMARY_KEY in codes(validate(df, TRANSACTIONS_CONTRACT, as_of=AS_OF))


def test_future_timestamp_is_an_error_not_a_curiosity():
    """A row stamped after as_of is a leakage vector, so it fails the gate."""
    df = make_transactions()
    df.loc[0, "invoice_ts"] = AS_OF + timedelta(days=1)
    report = validate(df, TRANSACTIONS_CONTRACT, as_of=AS_OF)
    assert VC.FUTURE_TIMESTAMP in codes(report)
    assert not report.passed


def test_stale_source_breaches_the_freshness_sla():
    df = make_transactions()
    df["invoice_ts"] = AS_OF - timedelta(hours=48)
    report = validate(df, TRANSACTIONS_CONTRACT, as_of=AS_OF)
    assert VC.STALE_SOURCE in codes(report)


def test_fresh_source_within_sla_passes():
    df = make_transactions()
    df["invoice_ts"] = AS_OF - timedelta(hours=12)
    assert validate(df, TRANSACTIONS_CONTRACT, as_of=AS_OF).passed


def test_empty_batch_is_an_error_by_default():
    empty = make_transactions().iloc[0:0]
    assert VC.EMPTY_BATCH in codes(validate(empty, TRANSACTIONS_CONTRACT, as_of=AS_OF))


def test_empty_batch_can_be_permitted_explicitly():
    contract = TRANSACTIONS_CONTRACT.model_copy(update={"allow_empty": True})
    empty = make_transactions().iloc[0:0]
    assert validate(empty, contract, as_of=AS_OF).passed


# ---------------------------------------------------------------------------
# Row checks
# ---------------------------------------------------------------------------


def test_returns_must_be_credit_notes():
    df = make_transactions()
    df.loc[0, "quantity"] = -5  # negative quantity on a non-credit invoice
    report = validate(df, TRANSACTIONS_CONTRACT, as_of=AS_OF)
    assert VC.ROW_CHECK_FAILED in codes(report)


def test_negative_quantity_on_a_credit_note_is_fine():
    df = make_transactions()
    df.loc[0, "quantity"] = -5
    df.loc[0, "invoice_id"] = "C-INV-0"
    assert validate(df, TRANSACTIONS_CONTRACT, as_of=AS_OF).passed


def test_unregistered_row_check_fails_loudly():
    """A typo'd check name must not silently mean 'no check'."""
    contract = TRANSACTIONS_CONTRACT.model_copy(update={"row_checks": ("does_not_exist",)})
    report = validate(make_transactions(), contract, as_of=AS_OF)
    assert VC.ROW_CHECK_FAILED in codes(report)
    assert "not registered" in report.violations[0].detail


def test_row_check_needing_an_absent_column_reports_rather_than_crashing():
    contract = SourceContract(
        source="partial",
        columns=(ColumnContract(name="a", dtype=DType.INT),),
        row_checks=("price_is_positive",),
    )
    report = validate(pd.DataFrame({"a": [1]}), contract, as_of=AS_OF)
    assert VC.ROW_CHECK_FAILED in codes(report)


def test_warn_only_checks_do_not_gate_the_batch():
    contract = TRANSACTIONS_CONTRACT.model_copy(
        update={"row_checks": (), "warn_only_checks": ("price_is_positive",)}
    )
    df = make_transactions()
    df.loc[0, "unit_price"] = 0.0
    report = validate(df, contract, as_of=AS_OF)
    assert report.warnings
    assert report.passed, "warnings must not quarantine a batch"


def test_duplicate_check_registration_is_rejected():
    with pytest.raises(ValueError, match="already registered"):

        @register_check("price_is_positive")
        def _dupe(df):  # pragma: no cover
            return df["x"] > 0


def test_registry_contains_the_documented_checks():
    assert {"returns_are_negative_quantity", "price_is_positive"} <= set(CHECK_REGISTRY)


# ---------------------------------------------------------------------------
# Foreign keys and enforcement
# ---------------------------------------------------------------------------


def test_orphan_foreign_key_is_reported_when_references_are_supplied():
    df = make_transactions()
    report = validate(
        df, TRANSACTIONS_CONTRACT, as_of=AS_OF, references={"products.sku": ["SKU-000"]}
    )
    assert VC.ORPHAN_FOREIGN_KEY in codes(report)


def test_foreign_key_is_skipped_when_no_reference_set_is_given():
    """Absent references means unchecked, not failed — the FK is another system's."""
    assert validate(make_transactions(), TRANSACTIONS_CONTRACT, as_of=AS_OF).passed


def test_enforce_raises_and_carries_the_report():
    df = make_transactions()
    df.loc[0, "unit_price"] = -1.0
    with pytest.raises(DataContractViolation) as excinfo:
        TRANSACTIONS_CONTRACT.enforce(df, as_of=AS_OF)
    assert excinfo.value.report.errors
    assert "transactions" in str(excinfo.value)


def test_all_violations_are_reported_not_just_the_first():
    """One bad column must not mask another; the batch is diagnosed in one pass."""
    df = make_transactions()
    df.loc[0, "unit_price"] = -1.0
    df.loc[1, "quantity"] = 50_000
    df["invoice_ts"] = AS_OF - timedelta(hours=100)
    report = validate(df, TRANSACTIONS_CONTRACT, as_of=AS_OF)
    assert {VC.OUT_OF_RANGE, VC.STALE_SOURCE, VC.ROW_CHECK_FAILED} <= codes(report)


def test_report_is_serialisable_for_quarantine():
    df = make_transactions()
    df.loc[0, "unit_price"] = -1.0
    report = validate(df, TRANSACTIONS_CONTRACT, as_of=AS_OF)
    assert "DQ-" in report.model_dump_json()


def test_severity_split_is_exhaustive():
    df = make_transactions()
    df.loc[0, "quantity"] = -5
    report = validate(df, TRANSACTIONS_CONTRACT, as_of=AS_OF)
    assert len(report.errors) + len(report.warnings) == len(report.violations)
    assert all(v.severity in Severity for v in report.violations)
