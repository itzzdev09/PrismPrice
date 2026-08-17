"""
Governance & guardrail subsystem (L4).

Hard feasibility constraints applied *after* scoring, so no model output can
produce an illegal price. Public surface:

    from prismprice.governance import GuardrailEngine, PriceRequest

    engine = GuardrailEngine()
    verdict = engine.evaluate_candidate(20.95, request)
"""

from prismprice.governance.guardrails import (
    AbsoluteFloorGuardrail,
    BaseGuardrail,
    ChangeFrequencyGuardrail,
    CompetitiveCeilingGuardrail,
    EUOmnibusAnchorGuardrail,
    FairnessGuardrail,
    GuardrailEngine,
    InventoryGuardGuardrail,
    LadderComplianceGuardrail,
    MarginFloorGuardrail,
    MovementCapGuardrail,
    at_least,
    at_most,
    default_guardrails,
)
from prismprice.governance.schemas import (
    CandidateEvaluation,
    CandidateScore,
    DecisionEstimates,
    DecisionInputs,
    DecisionRecord,
    DegradationRung,
    ElasticityEstimate,
    ExplorationRecord,
    GuardrailCode,
    GuardrailResult,
    GuardrailStatus,
    LadderEvaluation,
    PriceRequest,
)

__all__ = [
    "AbsoluteFloorGuardrail",
    "BaseGuardrail",
    "CandidateEvaluation",
    "CandidateScore",
    "ChangeFrequencyGuardrail",
    "CompetitiveCeilingGuardrail",
    "DecisionEstimates",
    "DecisionInputs",
    "DecisionRecord",
    "DegradationRung",
    "EUOmnibusAnchorGuardrail",
    "ElasticityEstimate",
    "ExplorationRecord",
    "FairnessGuardrail",
    "GuardrailCode",
    "GuardrailEngine",
    "GuardrailResult",
    "GuardrailStatus",
    "InventoryGuardGuardrail",
    "LadderComplianceGuardrail",
    "LadderEvaluation",
    "MarginFloorGuardrail",
    "MovementCapGuardrail",
    "PriceRequest",
    "at_least",
    "at_most",
    "default_guardrails",
]
