"""
Benchmark: does propagating the elasticity interval into the markdown DP pay?

Three experiments, deliberately answering three different questions, because no
single one of them is enough on its own.

**Synthetic (§1)** is the only setting where the question can be answered
outright, because it is the only one with a known true elasticity. DML is fitted
to a generated panel, its confidence interval is fed to the robust operator, and
every policy is then scored against the elasticity the generator actually used.
That yields *regret against an oracle* — the profit given up relative to a policy
that was told the truth — which is the quantity the whole method is trying to
reduce. It cannot be computed on real data by anyone, ever.

**Real (§2)** uses the 443-SKU UCI Online Retail II panel already in this repo,
where no true elasticity exists. Regret against an oracle is therefore
unavailable, and inventing one would mean scoring the method on a truth of our
own choosing. The question that *can* be answered honestly is decision-theoretic:
across the set of elasticities the data cannot distinguish between, how does
each policy behave? Mean and worst case over the identified set are reported,
which is a claim about the estimator's own uncertainty rather than about
realised profit.

**Multi-SKU (§3)** asks whether the idea survives the loss of exact
solvability. The joint state cannot be enumerated, so the tail cannot be taken
inside a Bellman recursion; domain randomisation over the interval is what
remains. That is a weaker instrument and the section is written to show the
difference rather than to blur it.

Every comparison uses :func:`~prismprice.decision.robust_markdown.exact_policy_value`
rather than simulation. The policies differ by a few percent, several hundred
simulated seasons carry a standard error of the same order, and a benchmark
whose noise is as large as its effect decides nothing. The quantity wanted is an
expectation over an enumerable state space, so it is computed rather than
sampled. Simulation appears once, in the floor-calibration table, where the
*distribution* of outcomes is the point rather than its mean.

Run::

    python scripts/evaluate_robust_markdown.py

Results land in ``data/runs/robust_markdown_benchmark.json``.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prismprice import config
from prismprice.data.synthetic import generate_panel
from prismprice.decision.markdown import solve_markdown
from prismprice.decision.robust_markdown import (
    RobustMarkdownProblem,
    exact_policy_value,
    simulate_robust_policy,
    solve_regret_robust_markdown,
    solve_robust_markdown,
)
from prismprice.estimation.elasticity import DoubleMLElasticity

REPO = Path(__file__).resolve().parent.parent
OUTPUT = REPO / "data" / "runs" / "robust_markdown_benchmark.json"
REAL_RUN = REPO / "data" / "runs" / "real_run.json"

# --- scenario assumptions, named because they are not in any dataset ---------
#
# No public retail panel carries a season length, an opening buy, or a salvage
# value, and UCI carries no COGS either (real_run.json already records the 45%
# gross-margin assumption standing in for it). These are the markdown scenario
# wrapped around the real demand and cost figures, and they are stated here
# rather than buried so that a reader can see exactly which numbers are the
# data's and which are ours.
SEASON_PERIODS = 40
#: Opening stock as a multiple of what the season would sell at full price.
#: Above 1.0 by construction: a season that clears itself without a markdown is
#: not a markdown problem, and every policy would tie on it.
STOCK_COVER = 2.0
#: Salvage as a fraction of unit cost — a clearance channel that recovers little.
SALVAGE_FRACTION_OF_COST = 0.10
#: A true markdown ladder: down from full price, never above it.
LADDER_LOW, LADDER_HIGH, LADDER_RUNGS = 0.50, 1.00, 14

#: Gross margin assumed for the markdown scenario, overriding the cost figure
#: carried by either dataset.
#:
#: This is the single most consequential assumption in the benchmark and it is
#: not cosmetic, so it is stated rather than tuned quietly. Sequential markdown
#: only has an interior solution at high margin, and the arithmetic is
#: unforgiving: discounting multiplies units by ``(p/p_full)**e`` and multiplies
#: unit margin by ``(p - c)/(p_full - c)``. When cost is 55% of price — the 45%
#: margin that real_run.json assumes for UCI, and roughly what the synthetic
#: generator draws — a 34% price cut at elasticity -2.2 raises volume 2.5x while
#: cutting unit margin to 0.15x, destroying 62% of contribution. The DP
#: correctly refuses to mark down at all, every policy holds full price, and
#: every comparison in this file would tie at exactly zero.
#:
#: That is a real property of low-margin retail, not a defect to be engineered
#: around, and §1 reports it directly as a scope finding. But it means a
#: benchmark run at grocery margins measures nothing about the operator. Markdown
#: optimisation is a fashion, seasonal and end-of-life discipline, and those
#: categories carry initial margins in the 60-70% range — hence 65%. Both
#: datasets' own cost columns are set aside here for that reason, and the
#: sensitivity of the whole result to this number is measured explicitly by
#: ``margin_sensitivity`` rather than left for a reader to worry about.
MARKDOWN_GROSS_MARGIN = 0.65

#: Demand is rescaled so no season exceeds this many opening units. The markdown
#: DP is very nearly invariant to a joint rescaling of demand and inventory —
#: only the Poisson coefficient of variation moves — so this changes runtime
#: rather than economics, and regret is reported as a percentage in any case.
#: ``scale_invariance_check`` measures the residual effect instead of assuming
#: it away.
MAX_OPENING_UNITS = 400

#: Elasticities at or above this are treated as unusable. A demand curve that
#: barely responds to price has no interior markdown solution, and an interval
#: crossing zero contains curves that slope the wrong way.
MAX_USABLE_ELASTICITY = -0.15

#: Candidate truths swept across each real SKU's interval. Coarser than
#: DEFAULT_ROBUST_GRID_SIZE because §2 evaluates a full policy set at every
#: point and the extra resolution changes no reported figure.
REAL_TRUTH_GRID = 9


@dataclass(frozen=True)
class Scenario:
    """One SKU's markdown season, plus what is known about its elasticity."""

    sku: str
    problem: RobustMarkdownProblem
    true_elasticity: float | None
    """None on real data, where nobody knows it."""
    ci_width: float
    confidence: str


def build_scenario(
    sku: str,
    reference_price: float,
    base_demand: float,
    point: float,
    ci_low: float,
    ci_high: float,
    true_elasticity: float | None = None,
    confidence: str = "high",
    gross_margin: float = MARKDOWN_GROSS_MARGIN,
) -> Scenario | None:
    """Wrap estimated demand economics into a solvable markdown season.

    Returns ``None`` rather than raising when the SKU cannot support the
    problem — an elasticity too close to zero, a demand level of zero, a ladder
    that collapses to one rung. Skipping is the honest outcome for those:
    forcing a degenerate instance into the benchmark would pad the sample with
    SKUs on which every policy ties by construction, which would drag every
    reported mean toward zero and make the method look weaker *and* safer than
    it is.
    """
    ci_low, ci_high = min(ci_low, ci_high), max(ci_low, ci_high)
    # An interval reaching toward zero is clipped, not discarded: the usable part
    # of it still describes a real demand curve. A *point estimate* that weak is
    # discarded, because then nothing in the interval is trustworthy.
    ci_high = min(ci_high, MAX_USABLE_ELASTICITY)
    ci_low = min(ci_low, ci_high)
    point = float(np.clip(point, ci_low, ci_high))
    if point >= MAX_USABLE_ELASTICITY or not np.isfinite([point, ci_low, ci_high]).all():
        return None
    if not np.isfinite([reference_price, base_demand]).all():
        return None
    if base_demand <= 0 or reference_price <= 0:
        return None

    unit_cost = reference_price * (1.0 - gross_margin)

    prices = tuple(
        float(np.round(reference_price * f, 2))
        for f in np.linspace(LADDER_LOW, LADDER_HIGH, LADDER_RUNGS)
    )
    prices = tuple(sorted(set(prices)))
    if len(prices) < 2 or prices[0] <= unit_cost:
        return None

    salvage = SALVAGE_FRACTION_OF_COST * unit_cost
    if salvage >= prices[0]:
        return None

    # Rescale demand and inventory together so the DP stays cheap. The pair is
    # what sets the problem; scaling both leaves the optimal policy essentially
    # unchanged (see MAX_OPENING_UNITS).
    inventory = SEASON_PERIODS * base_demand * STOCK_COVER
    if inventory > MAX_OPENING_UNITS:
        base_demand *= MAX_OPENING_UNITS / inventory
        inventory = float(MAX_OPENING_UNITS)
    inventory = round(inventory)
    if not 40 <= inventory <= MAX_OPENING_UNITS:
        return None

    try:
        problem = RobustMarkdownProblem(
            prices=prices,
            base_price=float(reference_price),
            base_demand=float(base_demand),
            elasticity_low=float(ci_low),
            elasticity_point=float(point),
            elasticity_high=float(ci_high),
            unit_cost=float(unit_cost),
            salvage_value=float(salvage),
            horizon=SEASON_PERIODS,
            initial_inventory=inventory,
        )
    except ValueError:
        return None

    return Scenario(
        sku=sku,
        problem=problem,
        true_elasticity=true_elasticity,
        ci_width=float(ci_high - ci_low),
        confidence=confidence,
    )


def policies_for(scenario: Scenario) -> dict[str, Any]:
    """The five price rules under comparison.

    ``oracle`` is available only on synthetic data and is not a competitor — it
    is the ceiling, the profit a policy would make if somebody handed it the
    true elasticity. Every other number is read as a shortfall from it.

    ``pessimistic_endpoint`` is the obvious alternative to this whole module:
    take the cautious end of the confidence interval and solve the ordinary DP
    there. It is in the table because it is what most people would try, and
    because the argument that it does not work is more convincing measured than
    asserted.

    The two ``value_robust_*`` entries and the two ``regret_robust_*`` entries
    are the two robust operators, and the contrast between them is the point of
    the whole benchmark: one protects the season's value across the ambiguity
    set, the other protects how much of the achievable profit is given up. They
    are both reported because the second was built in response to the first
    losing, and hiding the loser would make the comparison worthless.
    """
    problem = scenario.problem
    rules: dict[str, Any] = {
        "certainty_equivalent": solve_markdown(
            problem.at_elasticity(problem.elasticity_point)
        ).price_at,
        "pessimistic_endpoint": solve_markdown(
            problem.at_elasticity(problem.elasticity_low)
        ).price_at,
        "value_robust_half": solve_robust_markdown(problem, robustness_level=0.5).price_at,
        "value_robust_full": solve_robust_markdown(problem, robustness_level=1.0).price_at,
        # alpha at the shipped default is close to minimax over the grid; alpha
        # at 1 minimises mean regret instead. Both are reported because the
        # trade between them is the operator's only real dial.
        "regret_robust_cvar": solve_regret_robust_markdown(problem).price_at,
        "regret_robust_mean": solve_regret_robust_markdown(problem, cvar_alpha=1.0).price_at,
    }
    if scenario.true_elasticity is not None:
        rules["oracle"] = solve_markdown(problem.at_elasticity(scenario.true_elasticity)).price_at
    return rules


def score_at_truth(
    scenario: Scenario,
    truth: float,
    rules: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Exact expected profit of every policy, under one assumed true elasticity.

    ``rules`` is accepted so a caller sweeping many candidate truths solves the
    policies once instead of once per truth. The policies are built from the
    *estimate* and do not depend on the truth being scored against — re-solving
    them per candidate is not merely slow, it invites the mistake of accidentally
    letting the assumed truth leak into the policy it is supposed to be testing.
    """
    truth_problem = scenario.problem.at_elasticity(truth)
    opening = (truth_problem.horizon, truth_problem.initial_inventory)
    return {
        name: float(exact_policy_value(truth_problem, rule)[opening])
        for name, rule in (rules or policies_for(scenario)).items()
    }


# ---------------------------------------------------------------------------
# §1 Synthetic — the only place regret against a known truth is computable
# ---------------------------------------------------------------------------


def synthetic_experiment(
    day_counts: tuple[int, ...] = (120, 200, 320, 540),
    n_skus: int = 8,
    seeds: tuple[int, ...] = (1, 2, 3, 4),
) -> dict[str, Any]:
    """Fit DML on generated panels, price against its interval, score at the truth.

    Panel length is swept because it is the lever that moves confidence-interval
    width without touching anything else: the same demand process observed for
    120 days and for 540 days yields the same true elasticity and a very
    different amount of knowledge about it. If the method has any value it must
    appear as a function of that width — helping where the estimate is vague,
    and costing nothing where it is sharp. A method that helped uniformly would
    be suspicious, not impressive.
    """
    rows: list[dict[str, Any]] = []

    for n_days in day_counts:
        for seed in seeds:
            panel = generate_panel(n_skus=n_skus, n_days=n_days, n_customers=200, seed=seed)
            estimator = DoubleMLElasticity(n_repeats=2, n_folds=4, seed=seed)

            for index, sku in enumerate(panel.truth.skus):
                frame = panel.daily[panel.daily["sku"] == sku]
                truth = float(panel.truth.beta[index])
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        estimate = estimator.pooled(frame)
                except Exception:
                    continue
                if estimate.point is None or not np.isfinite(estimate.point):
                    continue

                scenario = build_scenario(
                    sku=f"{sku}-d{n_days}-s{seed}",
                    reference_price=float(panel.truth.reference_price[index]),
                    base_demand=float(frame["units"].median()),
                    point=float(estimate.point),
                    ci_low=float(estimate.ci_low),
                    ci_high=float(estimate.ci_high),
                    true_elasticity=truth,
                    confidence=str(estimate.confidence),
                )
                if scenario is None:
                    continue

                rules = policies_for(scenario)
                scored = score_at_truth(scenario, truth, rules)
                oracle = scored["oracle"]
                if oracle <= 0:
                    continue

                # The same worst-case-over-interval metric §2 reports, computed
                # here too so the two sections are commensurable. They answer
                # different questions and the difference matters: regret at the
                # single true elasticity rewards a good point estimate, while
                # regret across the interval is what a robust operator is
                # actually built to control. Reporting only the first would
                # understate the method; reporting only the second would dodge
                # the question of what it costs when the estimate was fine.
                interval_grid = np.linspace(
                    scenario.problem.elasticity_low,
                    scenario.problem.elasticity_high,
                    REAL_TRUTH_GRID,
                )
                sweep = pd.DataFrame(
                    [score_at_truth(scenario, float(e), rules) for e in interval_grid]
                )
                interval_oracle = np.array(
                    [
                        solve_markdown(scenario.problem.at_elasticity(float(e))).expected_profit
                        for e in interval_grid
                    ]
                )

                row: dict[str, Any] = {
                    "sku": scenario.sku,
                    "n_days": n_days,
                    "seed": seed,
                    "true_elasticity": truth,
                    "point_estimate": float(estimate.point),
                    "ci_low": scenario.problem.elasticity_low,
                    "ci_high": scenario.problem.elasticity_high,
                    "ci_width": scenario.ci_width,
                    "ci_covers_truth": bool(
                        scenario.problem.elasticity_low <= truth <= scenario.problem.elasticity_high
                    ),
                    "estimate_error": float(estimate.point) - truth,
                    "confidence": scenario.confidence,
                    "oracle_profit": oracle,
                }
                for name, value in scored.items():
                    if name == "oracle":
                        continue
                    row[f"profit_{name}"] = value
                    # Regret as a share of what perfect knowledge would have
                    # earned. Normalised because SKUs differ by an order of
                    # magnitude in season value, and an unnormalised mean would
                    # simply report the largest SKU's result.
                    row[f"regret_{name}"] = (oracle - value) / oracle
                    if name in sweep.columns:
                        across = (interval_oracle - sweep[name].to_numpy(dtype=float)) / (
                            interval_oracle
                        )
                        row[f"interval_worst_{name}"] = float(np.max(across))
                        row[f"interval_mean_{name}"] = float(np.mean(across))
                rows.append(row)

    frame = pd.DataFrame(rows)
    return {
        "n_scenarios": len(frame),
        "assumptions": {
            "season_periods": SEASON_PERIODS,
            "stock_cover": STOCK_COVER,
            "salvage_fraction_of_cost": SALVAGE_FRACTION_OF_COST,
        },
        "coverage_rate": float(frame["ci_covers_truth"].mean()) if len(frame) else None,
        "overall": _regret_summary(frame),
        "by_panel_length": {
            str(days): _regret_summary(frame[frame["n_days"] == days]) for days in day_counts
        },
        "by_interval_width": _by_width(frame),
        "rows": rows,
    }


def _regret_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Mean and median regret per policy, plus how often each policy wins."""
    if frame.empty:
        return {}
    policies = [c[len("regret_") :] for c in frame.columns if c.startswith("regret_")]
    summary: dict[str, Any] = {"n": len(frame)}
    for name in policies:
        column = frame[f"regret_{name}"]
        summary[name] = {
            "mean_regret_pct": float(column.mean() * 100.0),
            "median_regret_pct": float(column.median() * 100.0),
            "worst_regret_pct": float(column.max() * 100.0),
        }
    # The same policies scored on worst case *across the interval*, which is the
    # quantity the robust operators optimise. Kept separate from the
    # single-truth regret above rather than replacing it.
    for name in policies:
        column = f"interval_worst_{name}"
        if column in frame.columns:
            summary[name]["interval_worst_regret_pct"] = float(frame[column].mean() * 100.0)
            summary[name]["interval_mean_regret_pct"] = float(
                frame[f"interval_mean_{name}"].mean() * 100.0
            )

    # Head-to-head, which a mean can hide: a policy can win on average because
    # of a few large SKUs while losing on most of them.
    ce, robust = frame["regret_certainty_equivalent"], frame["regret_regret_robust_cvar"]
    summary["regret_robust_beats_certainty_equivalent"] = {
        "win_rate": float((robust < ce - 1e-12).mean()),
        "tie_rate": float((np.abs(robust - ce) <= 1e-12).mean()),
        "mean_regret_reduction_pp": float((ce - robust).mean() * 100.0),
    }
    if "interval_worst_regret_robust_cvar" in frame.columns:
        ce_w = frame["interval_worst_certainty_equivalent"]
        rr_w = frame["interval_worst_regret_robust_cvar"]
        summary["regret_robust_beats_certainty_equivalent_on_interval_worst_case"] = {
            "win_rate": float((rr_w < ce_w - 1e-12).mean()),
            "tie_rate": float((np.abs(rr_w - ce_w) <= 1e-12).mean()),
            "mean_improvement_pp": float((ce_w - rr_w).mean() * 100.0),
        }
    return summary


def _by_width(frame: pd.DataFrame, n_bins: int = 3) -> dict[str, Any]:
    """Regret bucketed by confidence-interval width — the paper's core claim.

    The method should be inert on sharp estimates and helpful on vague ones. If
    the buckets do not separate, the operator is not responding to the
    estimator's uncertainty and the whole construction is decorative.
    """
    if len(frame) < n_bins * 2:
        return {}
    labels = ["narrow", "medium", "wide"][:n_bins]
    bucketed = frame.assign(
        bucket=pd.qcut(frame["ci_width"], n_bins, labels=labels, duplicates="drop")
    )
    out: dict[str, Any] = {}
    for label, group in bucketed.groupby("bucket", observed=True):
        out[str(label)] = {
            "n": len(group),
            "mean_ci_width": float(group["ci_width"].mean()),
            **_regret_summary(group),
        }
    return out


def margin_sensitivity(
    margins: tuple[float, ...] = (0.35, 0.45, 0.55, 0.65, 0.75),
) -> dict[str, Any]:
    """How much of the headline result is the gross-margin assumption?

    ``MARKDOWN_GROSS_MARGIN`` is set to 65% on the argument that markdown is a
    high-margin discipline, and that argument deserves to be checked rather than
    trusted. A fixed synthetic season is priced under each policy at a range of
    margins, and the regret of a wrong elasticity is reported at each.

    The expected shape is that the method is *inert* at low margin — where the
    DP declines to mark down at all and every policy holds full price — and
    matters increasingly as margin rises. That is a scope statement for the
    method, and it is more useful stated with numbers than hedged in prose.
    """
    reference, demand = 40.0, 10.0
    rows: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    for margin in margins:
        scenario = build_scenario(
            sku=f"margin-{margin:.2f}",
            reference_price=reference,
            base_demand=demand,
            point=-2.2,
            ci_low=-3.0,
            ci_high=-1.5,
            true_elasticity=-1.5,
            gross_margin=margin,
        )
        if scenario is None:
            # Recorded rather than dropped. At these margins the ladder floor
            # sits below unit cost, so the season has no discountable range at
            # all — the strongest form of the inertness finding, and it would be
            # invisible if the row simply went missing.
            refused.append(
                {
                    "gross_margin": margin,
                    "unit_cost": reference * (1.0 - margin),
                    "ladder_floor": reference * LADDER_LOW,
                    "reason": "ladder floor is at or below unit cost; nothing to discount into",
                }
            )
            continue
        scored = score_at_truth(scenario, -1.5)
        oracle = scored["oracle"]
        opening_ce = solve_markdown(
            scenario.problem.at_elasticity(scenario.problem.elasticity_point)
        )
        rows.append(
            {
                "gross_margin": margin,
                "oracle_profit": oracle,
                "marks_down_at_all": bool(
                    opening_ce.price_at(SEASON_PERIODS, scenario.problem.initial_inventory)
                    < max(scenario.problem.prices) - 1e-9
                    or opening_ce.price_at(5, scenario.problem.initial_inventory // 3)
                    < max(scenario.problem.prices) - 1e-9
                ),
                **{
                    f"regret_{name}_pct": float((oracle - value) / oracle * 100.0)
                    for name, value in scored.items()
                    if name != "oracle"
                },
            }
        )
    return {
        "note": (
            "regret of each policy when the truth is -1.5 and the point estimate is "
            "-2.2, as gross margin varies; the method is inert where the DP declines "
            "to mark down at all"
        ),
        "shipped_assumption": MARKDOWN_GROSS_MARGIN,
        "rows": rows,
        "refused_margins": refused,
    }


def scale_invariance_check() -> dict[str, Any]:
    """Does rescaling demand and inventory together change the answer?

    ``MAX_OPENING_UNITS`` rescales large SKUs to keep the DP cheap, on the claim
    that the markdown problem is very nearly invariant to it. 'Very nearly' is
    doing real work in that sentence — demand is Poisson, so halving the mean
    raises its coefficient of variation — so the residual is measured rather
    than asserted.
    """
    rows = []
    # Chosen to stay *below* MAX_OPENING_UNITS, so the seasons genuinely differ
    # in size. An earlier version swept 5-40 units/period, every one of which
    # rescaled onto the same 400-unit cap, and the check reported a spread of
    # exactly zero — it was measuring the clamp, not the invariance.
    for demand in (1.25, 2.5, 5.0):
        scenario = build_scenario(
            sku=f"scale-{demand}",
            reference_price=40.0,
            base_demand=demand,
            point=-2.2,
            ci_low=-3.0,
            ci_high=-1.5,
            true_elasticity=-1.5,
        )
        if scenario is None:
            continue
        scored = score_at_truth(scenario, -1.5)
        oracle = scored["oracle"]
        rows.append(
            {
                "base_demand": demand,
                "opening_units": scenario.problem.initial_inventory,
                "regret_certainty_equivalent_pct": float(
                    (oracle - scored["certainty_equivalent"]) / oracle * 100.0
                ),
                "regret_robust_full_pct": float(
                    (oracle - scored["regret_robust_cvar"]) / oracle * 100.0
                ),
            }
        )
    spread = None
    if rows:
        values = [r["regret_certainty_equivalent_pct"] for r in rows]
        spread = float(max(values) - min(values))
    return {
        "note": "regret in pp across a 8x range of season size; small spread supports the rescaling",
        "regret_spread_pp": spread,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# §2 Real UCI panel — no true elasticity, so a different question
# ---------------------------------------------------------------------------


def _interval_is_wholly_inelastic(scenario: Scenario) -> bool:
    """Is every elasticity in the interval above -1, in magnitude?

    An analytic shortcut past the DP, and an exact one. Cutting price on
    inelastic demand loses on both factors at once — fewer pounds per unit, and
    not enough extra units to make it back — so full price is optimal at every
    elasticity in such an interval, and every policy in this benchmark emits the
    same ladder. ``elasticity_low`` is the most negative point in the set, so
    testing it alone settles the whole interval.

    Worth having because it is most of the panel: screening these by solving
    four dynamic programmes each would dominate the runtime of §2 to establish
    something already true by inspection.
    """
    return scenario.problem.elasticity_low >= -1.0


def _policies_disagree(scenario: Scenario) -> bool:
    """Does the elasticity interval change what this SKU should be priced at?

    The operational test: does the *optimal* ladder vary across the interval?
    If a policy told the truth would price identically whatever the truth turns
    out to be, then every policy here — robust or not — picks that same ladder,
    every regret is zero, and the SKU carries no information about robustness.

    Screening on the oracle rather than on the robust operators is both cheaper
    and the right question. It also avoids a circularity: deciding which SKUs
    count as evidence by consulting the very operators under test would let a
    quiet bug in one of them determine its own sample.
    """
    problem = scenario.problem
    states = [
        (t, i)
        for t in (problem.horizon, problem.horizon // 2, 3)
        for i in (
            problem.initial_inventory,
            max(problem.initial_inventory // 2, 1),
            max(problem.initial_inventory // 8, 1),
        )
    ]
    ladders = set()
    for elasticity in np.linspace(problem.elasticity_low, problem.elasticity_high, 5):
        solved = solve_markdown(problem.at_elasticity(float(elasticity)))
        ladders.add(tuple(round(solved.price_at(t, i), 6) for t, i in states))
        if len(ladders) > 1:
            return True
    return False


def real_data_experiment(max_skus: int = 40) -> dict[str, Any]:
    """Score policies across each SKU's own identified set.

    On real data the true elasticity is unknown and unknowable, so there is no
    oracle and no regret. What there is instead is the interval the estimator
    produced: a set of elasticities the data cannot tell apart. A policy can
    therefore be scored by how it behaves *across that set* — its mean over the
    interval, and its worst case within it.

    This is a weaker claim than §1 and is labelled as one. It says nothing about
    profit this retailer would have realised; it says how each policy behaves
    under the uncertainty the estimator actually reported on real transactions.
    """
    if not REAL_RUN.exists():
        return {"skipped": f"{REAL_RUN.name} not present; run the pipeline first"}

    run = json.loads(REAL_RUN.read_text())
    elasticities = run.get("elasticities", {})
    decisions = {str(d["sku"]): d for d in run.get("decisions", [])}

    scenarios: list[Scenario] = []
    for sku, record in elasticities.items():
        if record.get("confidence") != "high":
            continue
        decision = decisions.get(str(sku))
        if decision is None:
            continue
        if record.get("point") is None or record.get("ci_low") is None:
            continue
        units = decision.get("expected_units")
        price = decision.get("current_price")
        if not all(isinstance(v, (int, float)) for v in (units, price)):
            continue

        scenario = build_scenario(
            sku=str(sku),
            reference_price=float(price),
            base_demand=float(units),
            point=float(record["point"]),
            ci_low=float(record["ci_low"]),
            ci_high=float(record["ci_high"]),
            true_elasticity=None,
            confidence="high",
        )
        if scenario is not None:
            scenarios.append(scenario)

    scenarios.sort(key=lambda s: s.sku)

    # Split the panel before comparing anything. On a SKU whose whole interval
    # is inelastic the markdown DP holds full price at every elasticity in it —
    # correctly, since cutting price on inelastic demand loses money twice — so
    # every policy here emits an identical ladder and every regret is exactly
    # zero. Those SKUs are not evidence that the operator is safe; they are
    # evidence that no decision was at stake. Pooling them with the live ones
    # would divide the real effect by six and report the result as "no
    # difference", which is the most misleading thing this benchmark could do.
    live, degenerate, inelastic = [], [], []
    for scenario in scenarios:
        if _interval_is_wholly_inelastic(scenario):
            inelastic.append(scenario)
        elif _policies_disagree(scenario):
            live.append(scenario)
        else:
            degenerate.append(scenario)

    # Deterministic subsample of the live set: spread across the sorted SKU ids
    # rather than taking the first N, which would sample whatever the retailer's
    # numbering happens to correlate with.
    sampled = live
    if len(sampled) > max_skus:
        picks = np.linspace(0, len(sampled) - 1, max_skus).astype(int)
        sampled = [sampled[i] for i in dict.fromkeys(picks)]

    rows: list[dict[str, Any]] = []
    for scenario in sampled:
        grid = np.linspace(
            scenario.problem.elasticity_low,
            scenario.problem.elasticity_high,
            REAL_TRUTH_GRID,
        )
        # Solved once, from the estimate, then held fixed across every candidate
        # truth — which is what the policies would be in deployment.
        rules = policies_for(scenario)
        # Each column is one policy; each row one candidate truth in the set.
        table = pd.DataFrame([score_at_truth(scenario, float(e), rules) for e in grid])
        # The oracle here is per-candidate-truth: the best any policy could do
        # if that particular elasticity were the real one. It is computable
        # because it is defined relative to an assumed truth, not a known one.
        oracle = np.array(
            [solve_markdown(scenario.problem.at_elasticity(float(e))).expected_profit for e in grid]
        )
        if not np.all(oracle > 0):
            continue

        row: dict[str, Any] = {
            "sku": scenario.sku,
            "ci_width": scenario.ci_width,
            "elasticity_point": scenario.problem.elasticity_point,
            "initial_inventory": scenario.problem.initial_inventory,
        }
        for name in table.columns:
            values = table[name].to_numpy(dtype=float)
            regret = (oracle - values) / oracle
            row[f"mean_regret_{name}"] = float(regret.mean())
            row[f"worst_regret_{name}"] = float(regret.max())
        rows.append(row)

    total = len(scenarios) or 1
    degenerate_summary = {
        "n_wholly_inelastic": len(inelastic),
        "n_elastic_but_policies_agree": len(degenerate),
        "n_decision_relevant": len(live),
        "share_decision_relevant": len(live) / total,
        "median_point_elasticity_inelastic": (
            float(np.median([s.problem.elasticity_point for s in inelastic])) if inelastic else None
        ),
        "median_point_elasticity_live": (
            float(np.median([s.problem.elasticity_point for s in live])) if live else None
        ),
        "why": (
            "on a wholly inelastic interval, cutting price loses on margin and does not "
            "make it back on volume, so full price is optimal at every elasticity in the "
            "set; all four policies emit the same ladder and every regret is identically "
            "zero. This is the correct behaviour of the DP, not a limitation of the "
            "robust operator — there is simply no decision for robustness to protect."
        ),
    }

    frame = pd.DataFrame(rows)
    if frame.empty:
        return {
            "n_skus": 0,
            "source": "UCI Online Retail II via data/runs/real_run.json",
            "n_high_confidence_scenarios": len(scenarios),
            "decision_degenerate": degenerate_summary,
            "skipped": "no SKU on this panel had a decision-relevant elasticity interval",
        }

    policies = [c[len("mean_regret_") :] for c in frame.columns if c.startswith("mean_regret_")]
    summary = {
        name: {
            "mean_regret_over_interval_pct": float(frame[f"mean_regret_{name}"].mean() * 100.0),
            "worst_regret_over_interval_pct": float(frame[f"worst_regret_{name}"].mean() * 100.0),
            "max_worst_regret_pct": float(frame[f"worst_regret_{name}"].max() * 100.0),
        }
        for name in policies
    }
    ce = frame["worst_regret_certainty_equivalent"]
    robust = frame["worst_regret_regret_robust_cvar"]
    return {
        "n_skus": len(frame),
        "source": "UCI Online Retail II via data/runs/real_run.json",
        "n_high_confidence_scenarios": len(scenarios),
        "n_decision_relevant": len(live),
        "decision_degenerate": degenerate_summary,
        "caveat": (
            "No true elasticity exists on real data. These are regrets across each "
            "SKU's own DML confidence interval, not realised profit. They also cover "
            "only the decision-relevant subset; see decision_degenerate for the rest."
        ),
        "inherited_assumptions": run.get("notes", []),
        "scenario_assumptions": {
            "season_periods": SEASON_PERIODS,
            "stock_cover": STOCK_COVER,
            "salvage_fraction_of_cost": SALVAGE_FRACTION_OF_COST,
            "note": "UCI has no season, opening buy, or salvage; these are ours.",
        },
        "median_ci_width": float(frame["ci_width"].median()),
        "summary": summary,
        "regret_robust_beats_certainty_equivalent_on_worst_case": {
            "win_rate": float((robust < ce - 1e-12).mean()),
            "mean_worst_case_improvement_pp": float((ce - robust).mean() * 100.0),
        },
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# §3 Floor calibration — the one place simulation is the right instrument
# ---------------------------------------------------------------------------


def floor_calibration_experiment(n_seasons: int = 1500) -> dict[str, Any]:
    """Is the certificate honest, and by how much is it conservative?

    Two distinct questions, run together because they share the setup.

    The certificate is a statement about *parameter* error: if the elasticity
    sits in the bad tail of the interval, the season still makes this much. That
    is checked by running the policy at every elasticity in its ambiguity set,
    taking the realised tail, and comparing. The floor should come in *under*
    the realised tail — it is a nested CVaR, applied afresh at every period with
    its own tail as the continuation, and nesting is known to dominate the
    static measure. The size of that gap is worth reporting rather than
    asserting, because it is the difference between a useful certificate and a
    vacuous one.

    The certificate is *not* a statement about demand noise, and the second
    number makes that explicit: the share of individual simulated seasons that
    finish below the floor even when the elasticity is exactly as assumed.
    """
    # Built through build_scenario so this table describes the same kind of
    # season as §1 and §2. An earlier hand-written instance carried a 59% margin
    # on a ladder that stopped well above cost, which left it barely
    # elasticity-sensitive: the floor moved by 8 currency units across a 5x
    # change in interval width, and the table looked like evidence that
    # ambiguity does not matter when it was evidence that *that instance* had
    # no markdown decision in it.
    reference = 40.0
    scenario = build_scenario(
        sku="floor-calibration",
        reference_price=reference,
        base_demand=8.0,
        point=-2.4,
        ci_low=-3.0,
        ci_high=-1.8,
    )
    assert scenario is not None
    base = scenario.problem.at_elasticity(-2.4)

    results = []
    for half_width in (0.3, 0.9, 1.5):
        problem = RobustMarkdownProblem.from_markdown_problem(
            base, base.elasticity - half_width, base.elasticity + half_width
        )
        solved = solve_robust_markdown(problem, robustness_level=1.0)

        # Exact expected profit at each elasticity in the ambiguity set.
        exact = np.array(
            [
                float(
                    exact_policy_value(problem.at_elasticity(float(e)), solved.price_at)[
                        problem.horizon, problem.initial_inventory
                    ]
                )
                for e in solved.elasticity_grid
            ]
        )
        k = max(1, int(np.ceil(solved.cvar_alpha * len(exact))))
        realised_tail = float(np.mean(np.sort(exact)[:k]))

        # And the season-level distribution, which the floor deliberately does
        # not cover.
        sampled = simulate_robust_policy(
            problem,
            solved.price_at,
            true_elasticity=problem.elasticity_point,
            certified_floor=solved.certified_profit_floor,
            n_seasons=n_seasons,
            seed=config.DEFAULT_SEED,
        )

        results.append(
            {
                "interval_half_width": half_width,
                "certified_floor": solved.certified_profit_floor,
                "certainty_equivalent_profit": solved.certainty_equivalent_profit,
                "robustness_cost": solved.robustness_cost,
                "realised_tail_over_ambiguity_set": realised_tail,
                # Signed, and named for what it is. Negative means the
                # certificate promised slightly more than the policy delivers
                # across the tail of the set, which is the usual direction —
                # see the module docstring in robust_markdown.py. It is reported
                # as a magnitude to be judged rather than a boolean that would
                # read as a passed check.
                "certificate_error_pct": float(
                    (realised_tail - solved.certified_profit_floor)
                    / solved.certified_profit_floor
                    * 100.0
                ),
                "season_level_floor_violation_rate": sampled["floor_violation_rate"],
                "mean_profit_at_point_estimate": sampled["mean_profit"],
            }
        )

    return {
        "cvar_alpha": config.DEFAULT_CVAR_ALPHA,
        "grid_size": config.DEFAULT_ROBUST_GRID_SIZE,
        "n_seasons_simulated": n_seasons,
        "note": (
            "certificate covers parameter error, not demand noise; the season-level "
            "violation rate is reported so the distinction is not left to be assumed"
        ),
        "results": results,
    }


# ---------------------------------------------------------------------------
# §4 Multi-SKU — where exactness is lost and only randomisation remains
# ---------------------------------------------------------------------------


def joint_experiment(episodes: int = 3000, n_seasons: int = 400) -> dict[str, Any]:
    """Does domain randomisation over the interval help the learned policy?

    The honest framing matters here. In the single-SKU case the tail is taken
    inside the recursion and the result carries a certificate. Here the joint
    state cannot be enumerated, there is no recursion to take a tail inside, and
    what is left is training the policy on seasons drawn from across the
    interval. That optimises the average over the interval rather than its tail,
    so no certificate is available and none is claimed.

    Both policies are then evaluated at a true elasticity neither was told,
    chosen at the elastic end of the interval — the case where a point estimate
    is most wrong.
    """
    from dataclasses import replace

    from prismprice.decision.joint_markdown import (
        JointMarkdownProblem,
        NeuralMarkdownPolicy,
        evaluate_joint_policy,
        independent_dp_policy,
    )

    # Same markdown economics as §1 — 65% gross margin, a ladder reaching half
    # of full price, and enough opening stock that the season cannot clear
    # itself. The first version of this instance carried a 58% margin on a
    # shallow ladder, and every policy including the per-SKU DP held full price
    # for the whole season and spent none of the budget: three identical rows of
    # 2224.0. That is the low-margin degeneracy §1 documents, reproduced by
    # accident in the one section least able to show it.
    point, ci_low, ci_high = -2.0, -3.0, -1.2
    full = 20.0
    cost = full * (1.0 - MARKDOWN_GROSS_MARGIN)
    problem = JointMarkdownProblem(
        prices=tuple(float(np.round(p, 2)) for p in np.linspace(LADDER_LOW * full, full, 6)),
        full_prices=(full,) * 3,
        base_demands=(5.0, 4.0, 6.0),
        elasticity=point,
        unit_costs=(cost,) * 3,
        salvage_values=(SALVAGE_FRACTION_OF_COST * cost,) * 3,
        horizon=14,
        inventories=(120, 95, 145),
        markdown_budget=600.0,
    )

    fixed = NeuralMarkdownPolicy(episodes=episodes, seed=config.DEFAULT_SEED).fit(problem)
    robust = NeuralMarkdownPolicy(episodes=episodes, seed=config.DEFAULT_SEED).fit(
        problem, elasticity_sampler=lambda rng: float(rng.uniform(ci_low, ci_high))
    )

    out: dict[str, Any] = {
        "elasticity_point": point,
        "elasticity_interval": [ci_low, ci_high],
        "episodes": episodes,
        "state_space_size": problem.state_space_size(),
        "caveat": (
            "domain randomisation optimises the mean over the interval, not its tail; "
            "no certificate is available here, unlike the single-SKU operator"
        ),
        "by_true_elasticity": {},
    }

    for truth in (ci_low, point, ci_high):
        world = replace(problem, elasticity=truth)
        out["by_true_elasticity"][f"{truth:+.2f}"] = {
            "fixed_training": evaluate_joint_policy(
                world, fixed.price_at, n_seasons=n_seasons, seed=config.DEFAULT_SEED
            ),
            "robust_training": evaluate_joint_policy(
                world, robust.price_at, n_seasons=n_seasons, seed=config.DEFAULT_SEED
            ),
            "independent_dp": evaluate_joint_policy(
                world,
                independent_dp_policy(world),
                n_seasons=n_seasons,
                seed=config.DEFAULT_SEED,
            ),
        }
    return out


# ---------------------------------------------------------------------------


def main() -> None:
    started = time.time()
    report: dict[str, Any] = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "defaults": {
            "robustness_level": config.DEFAULT_ROBUSTNESS_LEVEL,
            "cvar_alpha": config.DEFAULT_CVAR_ALPHA,
            "grid_size": config.DEFAULT_ROBUST_GRID_SIZE,
        },
    }

    print("§1 synthetic (known truth) ...", flush=True)
    report["synthetic"] = synthetic_experiment()
    print(f"   {report['synthetic']['n_scenarios']} scenarios", flush=True)

    print("§1b assumption sensitivity ...", flush=True)
    report["margin_sensitivity"] = margin_sensitivity()
    report["scale_invariance"] = scale_invariance_check()

    print("§2 real UCI panel (identified set) ...", flush=True)
    report["real"] = real_data_experiment()
    print(f"   {report['real'].get('n_skus', 0)} SKUs", flush=True)

    print("§3 floor calibration ...", flush=True)
    report["floor_calibration"] = floor_calibration_experiment()

    print("§4 multi-SKU learned policy ...", flush=True)
    try:
        report["joint"] = joint_experiment()
    except Exception as error:
        report["joint"] = {"failed": f"{type(error).__name__}: {error}"}
        print(f"   failed: {error}", flush=True)

    report["runtime_seconds"] = round(time.time() - started, 1)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {OUTPUT.relative_to(REPO)} in {report['runtime_seconds']}s")


if __name__ == "__main__":
    main()
