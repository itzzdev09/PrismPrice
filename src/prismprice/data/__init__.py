"""
Data foundation (L0): schema contracts, quality gates, synthetic ground truth,
and deterministic augmentation of the public UCI retail dataset.
"""

from prismprice.data.augment import (
    AssumptionSet,
    build_augmented_panel,
    normalise_uci,
    to_daily_demand,
)
from prismprice.data.contracts import (
    COMPETITOR_CONTRACT,
    DAILY_DEMAND_CONTRACT,
    TRANSACTIONS_CONTRACT,
    ColumnContract,
    DataContractViolation,
    DType,
    QualityReport,
    Severity,
    SourceContract,
    Violation,
    register_check,
    validate,
)
from prismprice.data.synthetic import (
    GroundTruth,
    SyntheticPanel,
    generate_panel,
    transactions_from_panel,
)

__all__ = [
    "COMPETITOR_CONTRACT",
    "DAILY_DEMAND_CONTRACT",
    "TRANSACTIONS_CONTRACT",
    "AssumptionSet",
    "ColumnContract",
    "DType",
    "DataContractViolation",
    "GroundTruth",
    "QualityReport",
    "Severity",
    "SourceContract",
    "SyntheticPanel",
    "Violation",
    "build_augmented_panel",
    "generate_panel",
    "normalise_uci",
    "register_check",
    "to_daily_demand",
    "transactions_from_panel",
    "validate",
]
