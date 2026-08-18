"""
Real data sources (L0).

Everything measured in this repository so far was measured on the synthetic
generator. That is the right way to prove an estimator — on real transactions
nobody knows the true elasticity, so there is nothing to score against — but a
system whose only data is its own test rig is not a system. This module is the
other half: the path real transactions take to get in.

The one source wired up is **UCI Online Retail II**: roughly a million real
invoice lines from a UK online giftware retailer, December 2009 to December
2011. It is chosen over the larger grocery benchmarks for one reason — it
carries a customer identifier, and is therefore the only freely available
transaction log that can support the retention and CLV layer at all. Its
weakness is the mirror of that strength: prices are fairly stable, so it
supports demand modelling far better than it supports elasticity. M5 is the
right complement there and is not wired up yet.

**What is real and what is assumed.** The invoice lines, quantities, prices,
timestamps and customer identifiers are observed. Cost, inventory and competitor
prices are **not in the dataset and not in any comparable public one**, so any
margin computed downstream rests on :class:`~prismprice.data.augment.AssumptionSet`.
"Real data" here means real demand and real prices, not real margins, and the
distinction is worth keeping sharp because every profit figure inherits it.

**Known defects in this dataset**, which the contract catches rather than the
reader discovering later: cancellations appear as invoices prefixed ``C`` with
negative quantities; roughly a quarter of lines have no customer id (guest
checkout); there are non-product stock codes (``POST``, ``M``, ``BANK CHARGES``,
``DOT``) that are postage and adjustments rather than things with a demand
curve; and a handful of lines carry zero or negative prices.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from prismprice.data.contracts import ColumnContract, DType, SourceContract

__all__ = [
    "UCI_ONLINE_RETAIL_II",
    "SourceSpec",
    "clean_uci_transactions",
    "drop_non_product_lines",
    "load_uci_excel",
    "transaction_contract",
]


@dataclass(frozen=True)
class SourceSpec:
    """A downloadable public dataset, described before it is fetched."""

    name: str
    url: str
    filename: str
    approx_mb: float
    description: str
    licence: str

    def describe(self) -> str:
        return (
            f"{self.name}\n"
            f"  url      : {self.url}\n"
            f"  file     : {self.filename} (~{self.approx_mb:.0f} MB)\n"
            f"  contents : {self.description}\n"
            f"  licence  : {self.licence}"
        )


UCI_ONLINE_RETAIL_II = SourceSpec(
    name="UCI Online Retail II",
    url="https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
    filename="online_retail_II.xlsx",
    approx_mb=44.0,
    description=(
        "~1,067,371 invoice lines from a UK online giftware retailer, "
        "2009-12-01 to 2011-12-09, with customer identifiers"
    ),
    licence="CC BY 4.0 (UCI Machine Learning Repository)",
)

#: Column aliases across the two UCI releases. The 2011 "Online Retail" set and
#: the 2009-2011 "Online Retail II" set describe the same fields under different
#: names, and code that handles only one silently fails on the other.
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "invoice_id": ("Invoice", "InvoiceNo"),
    "sku": ("StockCode",),
    "description": ("Description",),
    "quantity": ("Quantity",),
    "invoice_ts": ("InvoiceDate",),
    "unit_price": ("Price", "UnitPrice"),
    "customer_id": ("Customer ID", "CustomerID"),
    "country": ("Country",),
}

#: Stock codes that are not products. They have no demand curve — pricing
#: postage as if buyers chose it on elasticity would be a category error — so
#: they are removed before modelling rather than left to distort a category mean.
_NON_PRODUCT_CODES = frozenset(
    {
        "POST", "DOT", "C2", "M", "BANK CHARGES", "AMAZONFEE", "B",
        "CRUK", "PADS", "S", "gift_0001_10", "gift_0001_20",
        "gift_0001_30", "gift_0001_40", "gift_0001_50",
    }
)


def transaction_contract() -> SourceContract:
    """Contract for a normalised transaction extract.

    Matches the YAML in ``docs/architecture.md`` §3. ``customer_id`` is nullable
    on purpose: guest checkouts are a large share of real transaction logs, and
    collapsing them to a sentinel would create one synthetic mega-customer and
    corrupt every retention curve built on top.

    ``quantity`` permits negatives because a cancellation is a negative
    quantity, and the ``returns_are_negative_quantity`` row check is what
    enforces that they only appear where they should.
    """
    return SourceContract(
        source="transactions",
        columns=(
            ColumnContract(name="invoice_id", dtype=DType.STRING, nullable=False),
            ColumnContract(name="sku", dtype=DType.STRING, nullable=False),
            ColumnContract(name="line_number", dtype=DType.INT, nullable=False),
            ColumnContract(
                name="quantity", dtype=DType.INT, nullable=False, min=-100_000, max=100_000
            ),
            ColumnContract(name="unit_price", dtype=DType.DECIMAL, nullable=False, min=0.0),
            ColumnContract(name="invoice_ts", dtype=DType.TIMESTAMP, nullable=False),
            ColumnContract(name="customer_id", dtype=DType.STRING, nullable=True),
        ),
        primary_key=("invoice_id", "sku", "line_number"),
        timestamp_column="invoice_ts",
        row_checks=("returns_are_negative_quantity", "price_is_positive"),
    )


def load_uci_excel(path: str | Path, sheets: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Read a UCI Online Retail workbook into one raw frame.

    Online Retail II ships as a two-sheet workbook (2009-2010 and 2010-2011).
    Reading only the first sheet silently halves the dataset and shifts its date
    range by a year, which is the kind of mistake that produces a plausible
    model of the wrong period, so both are read and concatenated by default.

    Args:
        path: ``.xlsx`` workbook, or a ``.csv`` extract.
        sheets: Restrict to these sheet names. ``None`` reads every sheet.

    Returns:
        The raw frame, columns exactly as the file names them. Normalisation is
        :func:`~prismprice.data.augment.normalise_uci`'s job.

    Raises:
        FileNotFoundError: with the download instructions, rather than a bare
            path error — the file is not in the repository and is not fetched
            automatically.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"{source} not found. This dataset is not bundled with the repository.\n"
            f"{UCI_ONLINE_RETAIL_II.describe()}"
        )

    if source.suffix.lower() == ".csv":
        return pd.read_csv(source, encoding="latin-1")

    workbook = pd.read_excel(source, sheet_name=list(sheets) if sheets else None)
    if isinstance(workbook, pd.DataFrame):
        return workbook
    return pd.concat(workbook.values(), ignore_index=True)


def drop_non_product_lines(
    transactions: pd.DataFrame, sku_column: str = "sku"
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove postage, fees and adjustments from a normalised extract.

    These carry a price and a quantity and are not products: nobody weighs the
    elasticity of postage. Left in, they join the catalogue as SKUs with strange
    price paths and drag category aggregates around.

    Returns:
        ``(kept, removed_counts)`` — the counts are returned rather than logged
        so a caller can assert on them; a filter that quietly removes 40% of a
        dataset should have to say so.
    """
    codes = transactions[sku_column].astype("string").str.strip().str.upper()
    target = {c.upper() for c in _NON_PRODUCT_CODES}
    mask = codes.isin(target)

    removed = (
        transactions.loc[mask, sku_column].astype("string").value_counts().to_dict()
        if mask.any()
        else {}
    )
    return transactions.loc[~mask].reset_index(drop=True), {
        str(k): int(v) for k, v in removed.items()
    }


def clean_uci_transactions(
    transactions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove ledger adjustments that are not sales.

    The raw extract fails its own contract, and the two failures turn out to be
    the same fault seen twice. Inspecting the 6,163 zero-price rows and the 3,457
    negative-quantity-without-cancellation rows shows they are inventory
    adjustments, not transactions: every one of the negative-quantity rows also
    carries a zero price and no customer id, and the descriptions are stockroom
    notes — ``CHECK``, ``DAMAGED``, ``DAMAGES``, ``?``, ``MISSING``,
    ``FOUND``, ``THROWN AWAY``.

    Keeping them would feed a demand model a write-off of 240 units as if
    customers had returned them, which is reading a stockroom ledger as customer
    behaviour. Dropping them silently would be worse, so the counts come back to
    the caller and the ingest records them.

    **Cancellations are kept.** A ``C``-prefixed invoice with a negative quantity
    and a real price is a genuine return by a real customer, and
    :func:`~prismprice.data.augment.to_daily_demand` nets it off rather than
    discarding it — a return is real information about demand, and dropping it
    inflates the series.

    Args:
        transactions: Normalised extract.

    Returns:
        ``(clean, removed)`` where ``removed`` counts each rule that fired.
    """
    counts: dict[str, int] = {}

    priced = transactions["unit_price"] > 0
    counts["zero_or_negative_price"] = int((~priced).sum())
    clean = transactions.loc[priced]

    is_return = clean["is_return"] if "is_return" in clean.columns else False
    stray_negative = (clean["quantity"] < 0) & (~is_return)
    counts["negative_quantity_without_cancellation"] = int(stray_negative.sum())
    clean = clean.loc[~stray_negative]

    missing_key = clean["invoice_id"].isna() | clean["sku"].isna() | clean["invoice_ts"].isna()
    counts["null_primary_key_or_timestamp"] = int(missing_key.sum())
    clean = clean.loc[~missing_key]

    return clean.reset_index(drop=True), counts


def file_digest(path: str | Path, algorithm: str = "sha256") -> str:
    """Hash a downloaded file, so a run can name the exact bytes it used.

    Reproducibility (README §5) means a decision can be rebuilt months later.
    That needs the input pinned by content, not by filename — public datasets do
    get silently re-published.
    """
    digest = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return f"{algorithm}:{digest.hexdigest()}"
