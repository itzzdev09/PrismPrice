"""Thread-safe adapter that records scored decisions in the local DuckDB ledger."""

from __future__ import annotations

from threading import RLock
from typing import Any

from prismprice.storage import PriceStore

__all__ = ["LockedPriceStoreDecisionRepository"]


class LockedPriceStoreDecisionRepository:
    """Serialise writes to the shared DuckDB connection used by the local app."""

    def __init__(self, store: PriceStore) -> None:
        self.store = store
        self._lock = RLock()

    def record_decision(self, record: dict[str, Any]) -> None:
        with self._lock:
            self.store.record_decision(record)
