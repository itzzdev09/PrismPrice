"""
Schema contracts and quality gates (L0).

Every source declares a contract that is enforced at the boundary. A violation
quarantines the batch; bad data never flows downstream to become a wrong price.

Row-level checks are **named callables from a registry**, not evaluated
expression strings. The YAML in ``docs/architecture.md`` writes checks as
``expr: "quantity < 0 implies invoice_id LIKE 'C%'"``; resolving that at runtime
would mean either `eval` on config (an injection surface on the one code path
whose entire job is to distrust its input) or a bespoke expression parser. The
registry keeps checks in Python where they are typed, testable and importable,
and the YAML references them by name.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "CHECK_REGISTRY",
    "ColumnContract",
    "DType",
    "DataContractViolation",
    "QualityReport",
    "Severity",
    "SourceContract",
    "Violation",
    "register_check",
    "validate",
]


class DType(str, Enum):
    """Logical column types, mapped onto pandas dtypes at validation time."""

    STRING = "string"
    INT = "int"
    FLOAT = "float"
    DECIMAL = "decimal"
    TIMESTAMP = "timestamp"
    BOOL = "bool"


class Severity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"


class ViolationCode(str, Enum):
    """Stable codes so a quality report can be aggregated across runs."""

    MISSING_COLUMN = "DQ-001"
    WRONG_DTYPE = "DQ-002"
    UNEXPECTED_NULL = "DQ-003"
    OUT_OF_RANGE = "DQ-004"
    NOT_IN_ALLOWED_VALUES = "DQ-005"
    DUPLICATE_PRIMARY_KEY = "DQ-006"
    NULL_PRIMARY_KEY = "DQ-007"
    STALE_SOURCE = "DQ-008"
    FUTURE_TIMESTAMP = "DQ-009"
    ROW_CHECK_FAILED = "DQ-010"
    ORPHAN_FOREIGN_KEY = "DQ-011"
    EMPTY_BATCH = "DQ-012"


class ColumnContract(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    dtype: DType
    nullable: bool = False
    min: float | None = None
    max: float | None = None
    allowed_values: tuple[str, ...] | None = None
    foreign_key: str | None = Field(
        None, description="Reference as 'table.column'; checked when a reference set is supplied"
    )


class Violation(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: ViolationCode
    severity: Severity
    column: str | None
    detail: str
    row_count: int = 0
    sample: tuple[str, ...] = ()

    def __str__(self) -> str:
        where = f" [{self.column}]" if self.column else ""
        return f"{self.code.value} {self.severity.value}{where}: {self.detail}"


class QualityReport(BaseModel):
    """The artefact Phase 1's gate produces. Serialisable, so a batch can be
    quarantined with the evidence attached rather than just a log line."""

    model_config = ConfigDict(frozen=True)

    source: str
    rows: int
    evaluated_at: datetime
    violations: tuple[Violation, ...] = ()

    @property
    def errors(self) -> tuple[Violation, ...]:
        return tuple(v for v in self.violations if v.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[Violation, ...]:
        return tuple(v for v in self.violations if v.severity is Severity.WARNING)

    @property
    def passed(self) -> bool:
        """A batch passes when nothing error-severity fired. Warnings do not gate."""
        return not self.errors

    def summary(self) -> str:
        if self.passed and not self.warnings:
            return f"{self.source}: {self.rows} rows, clean"
        lines = [
            f"{self.source}: {self.rows} rows, "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        ]
        lines.extend(f"  {v}" for v in self.violations)
        return "\n".join(lines)


class DataContractViolation(Exception):
    """Raised by :meth:`SourceContract.enforce` when a batch fails its gate."""

    def __init__(self, report: QualityReport) -> None:
        super().__init__(report.summary())
        self.report = report


# ---------------------------------------------------------------------------
# Row-check registry
# ---------------------------------------------------------------------------

#: A row check receives the frame and returns a boolean mask of *failing* rows.
RowCheck = Callable[[pd.DataFrame], "pd.Series[bool]"]

CHECK_REGISTRY: dict[str, RowCheck] = {}


def register_check(name: str) -> Callable[[RowCheck], RowCheck]:
    """Register a named row-level check that contracts can reference by name."""

    def decorator(func: RowCheck) -> RowCheck:
        if name in CHECK_REGISTRY:
            raise ValueError(f"Row check {name!r} is already registered")
        CHECK_REGISTRY[name] = func
        return func

    return decorator


@register_check("returns_are_negative_quantity")
def _returns_are_negative_quantity(df: pd.DataFrame) -> pd.Series[bool]:
    """Negative quantities must belong to a credit note (invoice id starting 'C').

    Fails rows where quantity < 0 and the invoice is not a credit note. Returns
    booked as ordinary sales silently deflate demand for the SKU.
    """
    negative = df["quantity"] < 0
    is_credit = df["invoice_id"].astype("string").str.startswith("C", na=False)
    return negative & ~is_credit


@register_check("price_is_positive")
def _price_is_positive(df: pd.DataFrame) -> pd.Series[bool]:
    return df["unit_price"] <= 0


@register_check("cost_below_price")
def _cost_below_price(df: pd.DataFrame) -> pd.Series[bool]:
    """Warns on inverted cost/price, which usually means a currency mismatch."""
    return df["unit_cost"] >= df["unit_price"]


# ---------------------------------------------------------------------------
# Source contract
# ---------------------------------------------------------------------------


class SourceContract(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    columns: tuple[ColumnContract, ...]
    primary_key: tuple[str, ...] = ()
    timestamp_column: str | None = None
    freshness_sla_hours: float | None = None
    row_checks: tuple[str, ...] = ()
    warn_only_checks: tuple[str, ...] = ()
    allow_empty: bool = False

    def column(self, name: str) -> ColumnContract | None:
        return next((c for c in self.columns if c.name == name), None)

    # -- loading ----------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> SourceContract:
        """Load a contract from the YAML form used in ``docs/architecture.md``.

        ``checks`` entries are names resolved against :data:`CHECK_REGISTRY`; an
        unknown name is an error at load time, not a silently skipped check.
        """
        import yaml

        raw: Mapping[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        columns = tuple(
            ColumnContract(name=name, **spec) for name, spec in raw.get("columns", {}).items()
        )
        checks = tuple(raw.get("checks", ()))
        unknown = [c for c in checks if c not in CHECK_REGISTRY]
        if unknown:
            raise ValueError(
                f"Contract {raw.get('source')!r} references unregistered checks {unknown}. "
                f"Known checks: {sorted(CHECK_REGISTRY)}"
            )
        return cls(
            source=raw["source"],
            columns=columns,
            primary_key=tuple(raw.get("primary_key", ())),
            timestamp_column=raw.get("timestamp_column"),
            freshness_sla_hours=raw.get("freshness_sla_hours"),
            row_checks=checks,
            warn_only_checks=tuple(raw.get("warn_only_checks", ())),
        )

    # -- validation -------------------------------------------------------

    def validate_frame(
        self,
        df: pd.DataFrame,
        as_of: datetime | None = None,
        references: Mapping[str, Iterable[Any]] | None = None,
    ) -> QualityReport:
        return validate(df, self, as_of=as_of, references=references)

    def enforce(
        self,
        df: pd.DataFrame,
        as_of: datetime | None = None,
        references: Mapping[str, Iterable[Any]] | None = None,
    ) -> pd.DataFrame:
        """Validate and return the frame, or raise :class:`DataContractViolation`.

        This is the boundary call. Failing loudly here is the whole design: a
        quarantined batch is recoverable, a wrong published price is not.
        """
        report = self.validate_frame(df, as_of=as_of, references=references)
        if not report.passed:
            raise DataContractViolation(report)
        return df


# ---------------------------------------------------------------------------
# Validation engine
# ---------------------------------------------------------------------------

_DTYPE_PREDICATES: dict[DType, Callable[[Any], bool]] = {
    DType.STRING: lambda dt: pd.api.types.is_string_dtype(dt) or pd.api.types.is_object_dtype(dt),
    DType.INT: lambda dt: pd.api.types.is_integer_dtype(dt),
    DType.FLOAT: lambda dt: pd.api.types.is_float_dtype(dt) or pd.api.types.is_integer_dtype(dt),
    DType.DECIMAL: lambda dt: pd.api.types.is_float_dtype(dt) or pd.api.types.is_integer_dtype(dt),
    DType.TIMESTAMP: lambda dt: pd.api.types.is_datetime64_any_dtype(dt),
    DType.BOOL: lambda dt: pd.api.types.is_bool_dtype(dt),
}


def _sample(values: Sequence[Any], limit: int = 3) -> tuple[str, ...]:
    return tuple(str(v) for v in list(values)[:limit])


def validate(
    df: pd.DataFrame,
    contract: SourceContract,
    as_of: datetime | None = None,
    references: Mapping[str, Iterable[Any]] | None = None,
) -> QualityReport:
    """Check *df* against *contract* and return a :class:`QualityReport`.

    Checks are ordered cheapest-first and each is independent, so one failure
    never masks another — the report lists everything wrong with the batch, not
    just the first thing found.
    """
    evaluated_at = as_of or datetime.now(timezone.utc)
    violations: list[Violation] = []

    if df.empty and not contract.allow_empty:
        violations.append(
            Violation(
                code=ViolationCode.EMPTY_BATCH,
                severity=Severity.ERROR,
                column=None,
                detail="Batch contains no rows",
            )
        )

    present = set(df.columns)

    for col in contract.columns:
        if col.name not in present:
            violations.append(
                Violation(
                    code=ViolationCode.MISSING_COLUMN,
                    severity=Severity.ERROR,
                    column=col.name,
                    detail=f"Required column {col.name!r} is absent",
                )
            )
            continue

        series = df[col.name]

        predicate = _DTYPE_PREDICATES[col.dtype]
        if len(series) and not predicate(series.dtype):
            violations.append(
                Violation(
                    code=ViolationCode.WRONG_DTYPE,
                    severity=Severity.ERROR,
                    column=col.name,
                    detail=f"Expected {col.dtype.value}, found pandas dtype {series.dtype}",
                )
            )
            continue

        if not col.nullable:
            null_count = int(series.isna().sum())
            if null_count:
                violations.append(
                    Violation(
                        code=ViolationCode.UNEXPECTED_NULL,
                        severity=Severity.ERROR,
                        column=col.name,
                        detail=f"{null_count} null value(s) in a non-nullable column",
                        row_count=null_count,
                    )
                )

        non_null = series.dropna()

        if col.min is not None and len(non_null):
            below = non_null[non_null < col.min]
            if len(below):
                violations.append(
                    Violation(
                        code=ViolationCode.OUT_OF_RANGE,
                        severity=Severity.ERROR,
                        column=col.name,
                        detail=f"{len(below)} value(s) below minimum {col.min}",
                        row_count=len(below),
                        sample=_sample(below.tolist()),
                    )
                )

        if col.max is not None and len(non_null):
            above = non_null[non_null > col.max]
            if len(above):
                violations.append(
                    Violation(
                        code=ViolationCode.OUT_OF_RANGE,
                        severity=Severity.ERROR,
                        column=col.name,
                        detail=f"{len(above)} value(s) above maximum {col.max}",
                        row_count=len(above),
                        sample=_sample(above.tolist()),
                    )
                )

        if col.allowed_values is not None and len(non_null):
            disallowed = non_null[~non_null.isin(col.allowed_values)]
            if len(disallowed):
                violations.append(
                    Violation(
                        code=ViolationCode.NOT_IN_ALLOWED_VALUES,
                        severity=Severity.ERROR,
                        column=col.name,
                        detail=f"{len(disallowed)} value(s) outside {col.allowed_values}",
                        row_count=len(disallowed),
                        sample=_sample(disallowed.unique().tolist()),
                    )
                )

        if col.foreign_key and references and col.foreign_key in references:
            valid = set(references[col.foreign_key])
            orphans = non_null[~non_null.isin(valid)]
            if len(orphans):
                violations.append(
                    Violation(
                        code=ViolationCode.ORPHAN_FOREIGN_KEY,
                        severity=Severity.ERROR,
                        column=col.name,
                        detail=(f"{len(orphans)} value(s) with no match in {col.foreign_key}"),
                        row_count=len(orphans),
                        sample=_sample(orphans.unique().tolist()),
                    )
                )

    violations.extend(_check_primary_key(df, contract))
    violations.extend(_check_freshness(df, contract, evaluated_at))
    violations.extend(_check_rows(df, contract))

    return QualityReport(
        source=contract.source,
        rows=len(df),
        evaluated_at=evaluated_at,
        violations=tuple(violations),
    )


def _check_primary_key(df: pd.DataFrame, contract: SourceContract) -> list[Violation]:
    if not contract.primary_key or not set(contract.primary_key) <= set(df.columns):
        return []

    violations: list[Violation] = []
    key = df[list(contract.primary_key)]

    null_rows = int(key.isna().any(axis=1).sum())
    if null_rows:
        violations.append(
            Violation(
                code=ViolationCode.NULL_PRIMARY_KEY,
                severity=Severity.ERROR,
                column=",".join(contract.primary_key),
                detail=f"{null_rows} row(s) with a null primary-key component",
                row_count=null_rows,
            )
        )

    duplicated = key.duplicated(keep=False)
    dup_count = int(duplicated.sum())
    if dup_count:
        violations.append(
            Violation(
                code=ViolationCode.DUPLICATE_PRIMARY_KEY,
                severity=Severity.ERROR,
                column=",".join(contract.primary_key),
                detail=f"{dup_count} row(s) share a duplicated primary key",
                row_count=dup_count,
                sample=_sample(key[duplicated].astype(str).agg("|".join, axis=1).unique().tolist()),
            )
        )
    return violations


def _check_freshness(
    df: pd.DataFrame, contract: SourceContract, as_of: datetime
) -> list[Violation]:
    column = contract.timestamp_column
    if not column or column not in df.columns or df[column].isna().all():
        return []

    violations: list[Violation] = []
    stamps = pd.to_datetime(df[column], errors="coerce", utc=True).dropna()
    if stamps.empty:
        return []

    cutoff = pd.Timestamp(as_of).tz_localize("UTC") if as_of.tzinfo is None else pd.Timestamp(as_of)

    future = stamps[stamps > cutoff]
    if len(future):
        violations.append(
            Violation(
                code=ViolationCode.FUTURE_TIMESTAMP,
                severity=Severity.ERROR,
                column=column,
                detail=(
                    f"{len(future)} row(s) timestamped after as_of {cutoff.isoformat()}; "
                    "this is a leakage risk, not a curiosity"
                ),
                row_count=len(future),
                sample=_sample([t.isoformat() for t in future]),
            )
        )

    if contract.freshness_sla_hours is not None:
        age_hours = (cutoff - stamps.max()).total_seconds() / 3600.0
        if age_hours > contract.freshness_sla_hours:
            violations.append(
                Violation(
                    code=ViolationCode.STALE_SOURCE,
                    severity=Severity.ERROR,
                    column=column,
                    detail=(
                        f"Newest row is {age_hours:.1f}h old, SLA is "
                        f"{contract.freshness_sla_hours:.1f}h"
                    ),
                )
            )
    return violations


def _check_rows(df: pd.DataFrame, contract: SourceContract) -> list[Violation]:
    violations: list[Violation] = []
    for name in (*contract.row_checks, *contract.warn_only_checks):
        check = CHECK_REGISTRY.get(name)
        if check is None:
            violations.append(
                Violation(
                    code=ViolationCode.ROW_CHECK_FAILED,
                    severity=Severity.ERROR,
                    column=None,
                    detail=f"Row check {name!r} is not registered",
                )
            )
            continue
        try:
            failing = check(df)
        except KeyError as exc:
            violations.append(
                Violation(
                    code=ViolationCode.ROW_CHECK_FAILED,
                    severity=Severity.ERROR,
                    column=None,
                    detail=f"Row check {name!r} needs a column the batch lacks: {exc}",
                )
            )
            continue
        except (TypeError, ValueError) as exc:
            # A row check operating on a wrongly-typed column raises rather than
            # returning a mask. That is still a data-quality finding, not a
            # validator crash — the batch already has a WRONG_DTYPE violation and
            # deserves to see this one too rather than losing the whole report.
            violations.append(
                Violation(
                    code=ViolationCode.ROW_CHECK_FAILED,
                    severity=Severity.ERROR,
                    column=None,
                    detail=f"Row check {name!r} could not run on this batch's types: {exc}",
                )
            )
            continue

        count = int(failing.sum())
        if count:
            violations.append(
                Violation(
                    code=ViolationCode.ROW_CHECK_FAILED,
                    severity=(
                        Severity.WARNING if name in contract.warn_only_checks else Severity.ERROR
                    ),
                    column=None,
                    detail=f"Row check {name!r} failed on {count} row(s)",
                    row_count=count,
                )
            )
    return violations


# ---------------------------------------------------------------------------
# Built-in contracts
# ---------------------------------------------------------------------------

TRANSACTIONS_CONTRACT = SourceContract(
    source="transactions",
    primary_key=("invoice_id", "sku", "line_number"),
    timestamp_column="invoice_ts",
    freshness_sla_hours=24.0,
    columns=(
        ColumnContract(name="invoice_id", dtype=DType.STRING),
        ColumnContract(name="line_number", dtype=DType.INT, min=0),
        ColumnContract(name="sku", dtype=DType.STRING, foreign_key="products.sku"),
        ColumnContract(name="quantity", dtype=DType.INT, min=-10_000, max=10_000),
        ColumnContract(name="unit_price", dtype=DType.DECIMAL, min=0),
        ColumnContract(name="invoice_ts", dtype=DType.TIMESTAMP),
        # Guest checkouts are a large share of real transaction logs. Nullable
        # here is deliberate: retention modelling must exclude them explicitly
        # rather than silently treating them as one synthetic mega-customer.
        ColumnContract(name="customer_id", dtype=DType.STRING, nullable=True),
    ),
    row_checks=("returns_are_negative_quantity", "price_is_positive"),
)

DAILY_DEMAND_CONTRACT = SourceContract(
    source="daily_demand",
    primary_key=("sku", "date"),
    timestamp_column="date",
    freshness_sla_hours=36.0,
    columns=(
        ColumnContract(name="sku", dtype=DType.STRING),
        ColumnContract(name="date", dtype=DType.TIMESTAMP),
        ColumnContract(name="units", dtype=DType.FLOAT, min=0),
        ColumnContract(name="price", dtype=DType.FLOAT, min=0),
        ColumnContract(name="unit_cost", dtype=DType.FLOAT, min=0),
        ColumnContract(name="inventory_on_hand", dtype=DType.FLOAT, min=0),
        ColumnContract(name="stockout", dtype=DType.BOOL),
    ),
)

COMPETITOR_CONTRACT = SourceContract(
    source="competitor_observations",
    primary_key=("sku", "competitor", "observed_at"),
    timestamp_column="observed_at",
    freshness_sla_hours=24.0,
    columns=(
        ColumnContract(name="sku", dtype=DType.STRING),
        ColumnContract(name="competitor", dtype=DType.STRING),
        ColumnContract(name="observed_at", dtype=DType.TIMESTAMP),
        ColumnContract(name="price", dtype=DType.FLOAT, min=0),
        ColumnContract(name="in_stock", dtype=DType.BOOL),
    ),
)
