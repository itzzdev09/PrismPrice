"""Decision layer (L3): candidate ladder, objective function, and the engine."""

from prismprice.decision.engine import (
    DecisionEngine,
    DecisionOutcome,
    competitor_match_price,
    cost_plus_price,
)
from prismprice.decision.ladder import generate_ladder, snap_to_ending
from prismprice.decision.markdown import (
    MarkdownPolicy,
    MarkdownProblem,
    simulate_policy,
    solve_markdown,
)
from prismprice.decision.objective import (
    CandidateOutcome,
    ObjectiveWeights,
    sample_demand,
    score_candidate,
)

__all__ = [
    "CandidateOutcome",
    "DecisionEngine",
    "DecisionOutcome",
    "MarkdownPolicy",
    "MarkdownProblem",
    "ObjectiveWeights",
    "competitor_match_price",
    "cost_plus_price",
    "generate_ladder",
    "sample_demand",
    "score_candidate",
    "simulate_policy",
    "snap_to_ending",
    "solve_markdown",
]
