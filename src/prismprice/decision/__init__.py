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
from prismprice.decision.robust_markdown import (
    RegretRobustPolicy,
    RobustMarkdownPolicy,
    RobustMarkdownProblem,
    exact_policy_value,
    simulate_robust_policy,
    solve_regret_robust_markdown,
    solve_robust_markdown,
)

__all__ = [
    "CandidateOutcome",
    "DecisionEngine",
    "DecisionOutcome",
    "MarkdownPolicy",
    "MarkdownProblem",
    "ObjectiveWeights",
    "RegretRobustPolicy",
    "RobustMarkdownPolicy",
    "RobustMarkdownProblem",
    "competitor_match_price",
    "cost_plus_price",
    "exact_policy_value",
    "generate_ladder",
    "sample_demand",
    "score_candidate",
    "simulate_policy",
    "simulate_robust_policy",
    "snap_to_ending",
    "solve_markdown",
    "solve_regret_robust_markdown",
    "solve_robust_markdown",
]
