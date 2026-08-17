"""
Pydantic data contracts for governance, pricing requests, and audit records.

Two properties are load-bearing here:

1. :class:`PriceRequest` is ``frozen`` and ``extra="forbid"``. Forbidding extras
   is what makes :class:`~prismprice.governance.guardrails.FairnessGuardrail` a
   real check rather than a rubber stamp — a caller cannot smuggle
   ``customer_id`` into the decision context, because construction fails.
2. :class:`DecisionRecord` is ``frozen`` and carries every input, estimate,
   model version and seed needed to rebuild the number months later. Corrections
   are new records linked by ``supersedes_id``; nothing is ever mutated.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from prismprice import config

__all__ = [
    "PROTECTED_ATTRIBUTE_TOKENS",
    "CandidateEvaluation",
    "CandidateScore",
    "DecisionEstimates",
    "DecisionInputs",
    "DecisionRecord",
    "DegradationRung",
    "DemandQuantiles",
    "ElasticityEstimate",
    "ExplorationRecord",
    "GuardrailCode",
    "GuardrailResult",
    "GuardrailStatus",
    "LadderEvaluation",
    "PriceRequest",
]


# ---------------------------------------------------------------------------
# Fairness policy surface
# ---------------------------------------------------------------------------

#: Tokens that must never appear in a decision-context field name. Covers direct
#: identity attributes and the proxies most likely to reconstruct them.
#:
#: Matching is prefix-of-token-part, not raw substring: a field name is split on
#: underscores and each part is tested with ``part.startswith(token)``. Raw
#: substring matching produces false positives that erode trust in the check
#: (``storage`` contains ``age``, ``multiplier`` contains ``ip``), and a fairness
#: guardrail nobody believes is one that gets disabled.
PROTECTED_ATTRIBUTE_TOKENS: frozenset[str] = frozenset(
    {
        # direct identity
        "customer",
        "user",
        "account",
        "email",
        "phone",
        "name",
        "address",
        "ip",
        "device",
        "browser",
        # protected characteristics
        "gender",
        "sex",
        "age",
        "birth",
        "dob",
        "race",
        "ethnic",
        "religion",
        "disability",
        "pregnan",
        "marital",
        "nationality",
        # geographic and socioeconomic proxies
        "postcode",
        "postal",
        "zip",
        "geo",
        "income",
        "affluence",
        # behavioural proxies for willingness to pay
        "loyalty",
        "segment",
        "cohort",
        "propensity",
        "willingness",
    }
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class DegradationRung(str, Enum):
    """Degradation matrix states (README §5). ``level`` is the single source of
    truth for the numeric rung, so the code and the number cannot disagree."""

    OK_OPTIMAL = "OK_OPTIMAL"
    WARN_STALE_COMPETITOR = "WARN_STALE_COMPETITOR"
    FALLBACK_POOLED_ELASTICITY = "FALLBACK_POOLED_ELASTICITY"
    FALLBACK_RULE_ENGINE = "FALLBACK_RULE_ENGINE"
    CRITICAL_MAINTAIN_PREV = "CRITICAL_MAINTAIN_PREV"

    @property
    def level(self) -> int:
        """Numeric rung, 1 (healthy) through 5 (critical)."""
        return _RUNG_LEVELS[self]


_RUNG_LEVELS: dict[DegradationRung, int] = {
    DegradationRung.OK_OPTIMAL: 1,
    DegradationRung.WARN_STALE_COMPETITOR: 2,
    DegradationRung.FALLBACK_POOLED_ELASTICITY: 3,
    DegradationRung.FALLBACK_RULE_ENGINE: 4,
    DegradationRung.CRITICAL_MAINTAIN_PREV: 5,
}


class GuardrailCode(str, Enum):
    """Stable reason codes. These appear in audit records and dashboards, so a
    value is never reused or renumbered once shipped."""

    ABSOLUTE_FLOOR = "PP-G001"
    MARGIN_FLOOR = "PP-G002"
    EU_OMNIBUS = "PP-G003"
    COMPETITIVE_CEILING = "PP-G004"
    MOVEMENT_CAP = "PP-G005"
    CHANGE_FREQUENCY = "PP-G006"
    INVENTORY_GUARD = "PP-G007"
    LADDER_COMPLIANCE = "PP-G008"
    FAIRNESS_CHECK = "PP-G009"


class GuardrailStatus(str, Enum):
    """Outcome of one predicate.

    ``NOT_APPLICABLE`` is deliberately distinct from ``PASSED``: "we checked and
    it is fine" and "there was nothing to check" must be distinguishable in the
    audit log, otherwise a missing-data pass reads as a compliance pass.
    """

    PASSED = "PASSED"
    FAILED = "FAILED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


# ---------------------------------------------------------------------------
# Guardrail evaluation
# ---------------------------------------------------------------------------


class GuardrailResult(BaseModel):
    """Result of evaluating a single guardrail predicate.

    ``observed`` / ``limit`` / ``slack`` are the machine-readable form; ``detail``
    is the human rendering of the same facts. Dashboards and audit queries read
    the numbers, never the prose.
    """

    model_config = ConfigDict(frozen=True)

    reason_code: GuardrailCode
    constraint_name: str
    status: GuardrailStatus
    detail: str
    observed: float | None = Field(
        None, description="The quantity under test, e.g. the candidate price"
    )
    limit: float | None = Field(None, description="The threshold it was tested against")
    slack: float | None = Field(
        None,
        description="Signed distance to the constraint. >= 0 is feasible; "
        "negative is the size of the violation. Uniform across floors and ceilings.",
    )

    @property
    def passed(self) -> bool:
        """True unless the predicate actively failed."""
        return self.status is not GuardrailStatus.FAILED

    @property
    def binding(self) -> bool:
        """True when this constraint is what makes the candidate infeasible."""
        return self.status is GuardrailStatus.FAILED


# ---------------------------------------------------------------------------
# Request payload
# ---------------------------------------------------------------------------


class PriceRequest(BaseModel):
    """Input payload for a single-SKU recommendation.

    Deliberately carries no customer-level attribute of any kind. ``extra`` is
    forbidden so that this is enforced at construction, not by convention.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sku: str = Field(..., min_length=1, description="Unique product SKU identifier")
    as_of: datetime = Field(..., description="Point-in-time timestamp of the decision")
    current_price: float = Field(..., gt=0, description="Current published item price")
    unit_cost: float = Field(
        ..., gt=0, description="Fully-loaded unit cost (COGS + landed + fulfilment)"
    )

    # --- promotional context -------------------------------------------------
    is_promo: bool = Field(
        False,
        description="Whether this decision sets a promotional price. Part of the "
        "decision context and recorded in the audit log, because EU Omnibus "
        "applicability cannot be reconstructed without it.",
    )
    min_price_last_30d: float | None = Field(
        None,
        gt=0,
        description="Minimum published price over the trailing 30 days (EU Omnibus "
        "anchor). Required when is_promo is True.",
    )

    # --- market --------------------------------------------------------------
    competitor_price: float | None = Field(
        None, gt=0, description="Observed market competitor price"
    )
    competitor_observed_at: datetime | None = Field(
        None, description="Observation timestamp, used for staleness / rung 2"
    )
    competitor_ceiling_multiplier: float = Field(
        config.DEFAULT_COMPETITOR_CEILING_MULTIPLIER,
        ge=1.0,
        description="Maximum ceiling as a multiple of competitor price",
    )

    # --- inventory -----------------------------------------------------------
    inventory_cover_days: float | None = Field(
        None, ge=0, description="Days of stock cover remaining"
    )
    min_inventory_cover_days: float = Field(
        config.DEFAULT_MIN_INVENTORY_COVER_DAYS,
        ge=0,
        description="Cover threshold below which the inventory shadow price binds",
    )
    inventory_shadow_price: float = Field(
        0.0,
        ge=0,
        description="Inventory opportunity cost nu_i, added to the floor when cover "
        "is below threshold so scarce stock is not sold too cheaply",
    )

    # --- policy dials --------------------------------------------------------
    margin_floor_pct: float = Field(
        config.DEFAULT_MARGIN_FLOOR_PCT, ge=0, description="Minimum required margin over cost"
    )
    movement_cap_pct: float = Field(
        config.DEFAULT_MOVEMENT_CAP_PCT,
        ge=0,
        le=0.50,
        description="Maximum relative price change per step",
    )
    price_changes_in_window: int | None = Field(
        None,
        ge=0,
        description="Price changes already made in the rolling window. None means "
        "the change history was not supplied.",
    )
    max_changes_per_window: int = Field(
        config.DEFAULT_MAX_CHANGES_PER_WINDOW,
        ge=0,
        description="Maximum permitted changes per rolling window",
    )
    change_window_days: int = Field(
        config.DEFAULT_CHANGE_WINDOW_DAYS, gt=0, description="Length of the rolling window"
    )
    allowed_price_endings: tuple[int, ...] | None = Field(
        config.DEFAULT_ALLOWED_PRICE_ENDINGS,
        description="Permitted price endings in whole cents, e.g. (95, 99). None "
        "disables ladder snapping entirely.",
    )


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------


class CandidateScore(BaseModel):
    """Objective scores for one candidate price on the ladder."""

    model_config = ConfigDict(frozen=True)

    price: float = Field(..., gt=0)
    j_score: float = Field(..., description="Objective value J(p)")
    cvar: float = Field(..., description="Conditional Value at Risk at cvar_alpha")
    cvar_alpha: float = Field(
        config.DEFAULT_CVAR_ALPHA,
        gt=0,
        lt=1,
        description="Tail probability for the CVaR estimate. Explicit because the "
        "objective parameterises it; do not assume 5%.",
    )
    feasible: bool = True
    binding_constraints: list[GuardrailCode] = Field(default_factory=list)


class CandidateEvaluation(BaseModel):
    """Full guardrail verdict for one candidate price."""

    model_config = ConfigDict(frozen=True)

    price: float = Field(..., gt=0)
    feasible: bool
    results: list[GuardrailResult] = Field(default_factory=list)
    binding_constraints: list[GuardrailCode] = Field(default_factory=list)

    @property
    def min_slack(self) -> float | None:
        """Tightest constraint's slack — how close this price ran to the edge."""
        slacks = [r.slack for r in self.results if r.slack is not None]
        return min(slacks) if slacks else None


class LadderEvaluation(BaseModel):
    """Result of filtering a candidate ladder down to its feasible set.

    When the feasible set is empty the engine does not raise and does not invent
    a price: it returns the previous price with an explicit degradation rung, per
    the reliability contract in README §5.
    """

    model_config = ConfigDict(frozen=True)

    evaluations: list[CandidateEvaluation] = Field(default_factory=list)
    feasible_prices: list[float] = Field(default_factory=list)
    binding_constraint_counts: dict[GuardrailCode, int] = Field(default_factory=dict)
    fallback_price: float | None = None
    degradation_rung: DegradationRung = DegradationRung.OK_OPTIMAL

    @property
    def is_empty(self) -> bool:
        """True when no candidate on the ladder satisfied every guardrail."""
        return not self.feasible_prices


# ---------------------------------------------------------------------------
# Audit record
# ---------------------------------------------------------------------------


class DemandQuantiles(BaseModel):
    model_config = ConfigDict(frozen=True)

    p10: float
    p50: float
    p90: float


class ElasticityEstimate(BaseModel):
    model_config = ConfigDict(frozen=True)

    point: float
    ci_low: float
    ci_high: float
    method: str = Field(..., description="e.g. 'dml-orthogonal-r-learner'")
    confidence: str = Field(..., description="'high' | 'low'")

    @property
    def ci_width(self) -> float:
        return self.ci_high - self.ci_low


class DecisionInputs(BaseModel):
    """The input facts, snapshotted. Copied rather than referenced so the record
    stays readable when upstream feature values are later revised."""

    model_config = ConfigDict(frozen=True)

    current_price: float
    unit_cost: float
    is_promo: bool = False
    competitor_price: float | None = None
    inventory_cover_days: float | None = None
    inventory_shadow_price: float = 0.0
    min_price_last_30d: float | None = None

    @classmethod
    def from_request(cls, request: PriceRequest) -> DecisionInputs:
        return cls(
            current_price=request.current_price,
            unit_cost=request.unit_cost,
            is_promo=request.is_promo,
            competitor_price=request.competitor_price,
            inventory_cover_days=request.inventory_cover_days,
            inventory_shadow_price=request.inventory_shadow_price,
            min_price_last_30d=request.min_price_last_30d,
        )


class DecisionEstimates(BaseModel):
    model_config = ConfigDict(frozen=True)

    demand_quantiles: DemandQuantiles | None = None
    causal_elasticity: ElasticityEstimate | None = None
    delta_clv: float | None = None


class ExplorationRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    is_exploratory: bool = False
    propensity: float | None = Field(None, gt=0, le=1)


class DecisionRecord(BaseModel):
    """Immutable audit record for one recommendation.

    ``degradation_rung`` is derived from ``degradation_reason_code``. Supplying
    it is allowed — records must round-trip through JSON — but a value that
    contradicts the reason code is rejected rather than silently accepted, so the
    number and the code can never drift apart.

    ``policy_version`` and ``seed`` are required: an audit record that defaults
    its own version cannot be reconstructed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_id: str
    sku: str
    as_of: datetime
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    policy_version: str = Field(..., min_length=1)
    model_versions: dict[str, str] = Field(default_factory=dict)
    inputs: DecisionInputs
    estimates: DecisionEstimates = Field(default_factory=DecisionEstimates)
    candidates: list[CandidateScore] = Field(default_factory=list)
    recommended_price: float = Field(..., gt=0)
    binding_constraints: list[GuardrailCode] = Field(default_factory=list)
    guardrail_results: list[GuardrailResult] = Field(default_factory=list)
    degradation_reason_code: DegradationRung
    degradation_rung: int = Field(
        ...,
        ge=1,
        le=5,
        description="Numeric rung 1-5. Derived from degradation_reason_code; "
        "supply it only when round-tripping a serialised record.",
    )
    exploration: ExplorationRecord = Field(default_factory=ExplorationRecord)
    seed: int
    approver: str | None = None
    supersedes_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _derive_rung(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        raw_code = data.get("degradation_reason_code")
        if raw_code is None:
            return data
        try:
            derived = DegradationRung(raw_code).level
        except ValueError:
            return data  # let normal field validation report the bad code
        supplied = data.get("degradation_rung")
        if supplied is not None and int(supplied) != derived:
            raise ValueError(
                f"degradation_rung {supplied} contradicts degradation_reason_code "
                f"{DegradationRung(raw_code).value} (rung {derived})"
            )
        return {**data, "degradation_rung": derived}
