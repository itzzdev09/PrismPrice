"""
Central configuration: numeric tolerances, governance defaults, and policy dials.

Everything here is a *declared assumption*. Nothing in the decision path may
hardcode a threshold; it must be read from here or from the request payload so
that a decision can be reconstructed from configuration plus inputs alone.
"""

from __future__ import annotations

import os
from typing import Final

# ---------------------------------------------------------------------------
# Numeric comparison policy
# ---------------------------------------------------------------------------
#
# Guardrails are HARD constraints, so comparisons must not carry business slack.
# The only tolerance permitted is an allowance for IEEE-754 representation error
# (e.g. 10.00 * 1.15 == 11.499999999999998, which should be treated as 11.50).
#
# The previous implementation used an absolute 1e-4 slack, which silently
# admitted marginally-illegal prices and scaled inconsistently across price
# magnitudes. These tolerances are ~5 orders of magnitude tighter and relative.

FLOAT_REL_TOL: Final[float] = 1e-9
FLOAT_ABS_TOL: Final[float] = 1e-12

# ---------------------------------------------------------------------------
# Governance defaults (overridable per request / per category policy)
# ---------------------------------------------------------------------------

DEFAULT_MARGIN_FLOOR_PCT: Final[float] = 0.15
DEFAULT_MOVEMENT_CAP_PCT: Final[float] = 0.15
DEFAULT_COMPETITOR_CEILING_MULTIPLIER: Final[float] = 1.10
DEFAULT_MAX_CHANGES_PER_WINDOW: Final[int] = 4
DEFAULT_CHANGE_WINDOW_DAYS: Final[int] = 28
DEFAULT_MIN_INVENTORY_COVER_DAYS: Final[float] = 14.0

# Age beyond which a competitor observation is treated as stale, triggering
# degradation rung 2 (WARN_STALE_COMPETITOR).
COMPETITOR_STALENESS_HOURS: Final[float] = 24.0

# Psychological price endings, expressed in whole cents (see README §L3).
DEFAULT_ALLOWED_PRICE_ENDINGS: Final[tuple[int, ...]] = (95, 99)

# ---------------------------------------------------------------------------
# Decision / simulation defaults
# ---------------------------------------------------------------------------

DEFAULT_MONTE_CARLO_DRAWS: Final[int] = 2_000
DEFAULT_CVAR_ALPHA: Final[float] = 0.05
DEFAULT_CLV_HORIZON_PERIODS: Final[int] = 12
DEFAULT_CLV_DISCOUNT_RATE: Final[float] = 0.10
DEFAULT_SEED: Final[int] = 20260817

# ---------------------------------------------------------------------------
# Compute policy
# ---------------------------------------------------------------------------
#
# PrismPrice trains and scores its ML models on GPU only. See prismprice.compute.
# The escape hatch exists solely so that CI and the guardrail unit tests — which
# touch no ML code at all — can run on CPU-only runners.

ALLOW_CPU_ENV_VAR: Final[str] = "PRISMPRICE_ALLOW_CPU"


def cpu_fallback_allowed() -> bool:
    """True only when the CPU escape hatch is explicitly set (CI/test use)."""
    return os.environ.get(ALLOW_CPU_ENV_VAR, "").strip().lower() in {"1", "true", "yes"}
