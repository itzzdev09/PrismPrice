"""
Central configuration: numeric tolerances, governance defaults, and policy dials.

Everything here is a *declared assumption*. Nothing in the decision path may
hardcode a threshold; it must be read from here or from the request payload so
that a decision can be reconstructed from configuration plus inputs alone.

**Every constant carries a provenance record.** A bare float here used to be
unfalsifiable — 0.15 and 0.15 look identical whether one was elicited from a
category owner and the other typed to make a test pass. Each value is now
declared through :func:`prismprice.provenance.register`, which returns the value
and files the evidence, so the number appears exactly once and cannot drift from
its justification. ``tests/test_provenance.py`` fails if a constant defined here
has no record, and CI fails if any decision-path constant is a ``PLACEHOLDER``.

Three classes of number are kept apart on purpose, because conflating them is
how a preference gets defended as if it were a measurement:

* **Policy** dials (``lambda``, ``gamma``, margin floor, movement cap) encode
  what the business wants. They cannot be estimated from data at any sample
  size, so they are *solved for* from a stated trade — see
  :func:`prismprice.provenance.lambda_from_tradeoff` — and ship with a
  sensitivity bracket rather than as a point estimate.
* **Technical** constants follow from a machine or statistical constraint.
* **Literature** values come from published research and carry a citation.

Generator priors (§ Synthetic generator) are *not* in the decision path: they
parameterise the synthetic panel used to test estimators. Their job is to be
recoverable and representative, not to price anything.
"""

from __future__ import annotations

import os
from typing import Final

from prismprice.provenance import (
    Parameter,
    Provenance,
    gamma_from_tail_tradeoff,
    lambda_from_tradeoff,
    register,
)

# ---------------------------------------------------------------------------
# Numeric comparison policy
# ---------------------------------------------------------------------------

FLOAT_REL_TOL: Final[float] = register(
    Parameter(
        name="FLOAT_REL_TOL",
        value=1e-9,
        provenance=Provenance.TECHNICAL,
        rationale=(
            "Guardrails are hard constraints, so the only tolerance permitted is an "
            "allowance for IEEE-754 representation error (10.00 * 1.15 == 11.499999999999998, "
            "which must read as 11.50). float64 carries ~15-16 significant decimal digits; "
            "1e-9 sits ~6 orders above machine epsilon, which absorbs error accumulated over "
            "a short chain of arithmetic while remaining ~5 orders tighter than the absolute "
            "1e-4 slack it replaced. Relative, so it does not scale inconsistently with price "
            "magnitude the way an absolute slack does."
        ),
    )
)

FLOAT_ABS_TOL: Final[float] = register(
    Parameter(
        name="FLOAT_ABS_TOL",
        value=1e-12,
        provenance=Provenance.TECHNICAL,
        rationale=(
            "Companion floor to FLOAT_REL_TOL for comparisons against zero, where a relative "
            "tolerance degenerates. Set well below the smallest monetary quantum (1e-2) so it "
            "can never admit a business-meaningful difference."
        ),
    )
)

# ---------------------------------------------------------------------------
# Governance defaults (overridable per request / per category policy)
# ---------------------------------------------------------------------------

DEFAULT_MARGIN_FLOOR_PCT: Final[float] = register(
    Parameter(
        name="DEFAULT_MARGIN_FLOOR_PCT",
        value=0.15,
        provenance=Provenance.POLICY,
        rationale=(
            "The minimum contribution margin the business will accept on a recommended "
            "price. Not derivable from data: a loss-leading price can be optimal for the "
            "objective and still be refused as a matter of policy, which is precisely what "
            "this floor exists to express."
        ),
        owner="Category commercial owner",
        elicitation=(
            "Below what gross margin would you decline a price even if the model showed it "
            "was profit-maximising over the horizon?"
        ),
        sensitivity=(0.05, 0.40),
        requires_local_elicitation=True,
    )
)

DEFAULT_MOVEMENT_CAP_PCT: Final[float] = register(
    Parameter(
        name="DEFAULT_MOVEMENT_CAP_PCT",
        value=0.15,
        provenance=Provenance.POLICY,
        rationale=(
            "Caps single-step price movement. Trades optimisation speed against customer "
            "trust and reference-price damage: a large jump is both more visible and more "
            "likely to be acting on a model error, since the estimator has no evidence at "
            "prices far from those observed."
        ),
        owner="Pricing policy owner",
        elicitation=(
            "What single-day price change would a regular customer notice and resent, and "
            "how far from observed prices are you willing to extrapolate?"
        ),
        sensitivity=(0.02, 0.30),
        requires_local_elicitation=True,
    )
)

DEFAULT_COMPETITOR_CEILING_MULTIPLIER: Final[float] = register(
    Parameter(
        name="DEFAULT_COMPETITOR_CEILING_MULTIPLIER",
        value=1.10,
        provenance=Provenance.POLICY,
        rationale=(
            "How far above the observed competitor price the recommendation may sit. A "
            "positioning statement, not a measurement — a brand with genuine differentiation "
            "supports a higher multiplier and a commodity does not."
        ),
        owner="Category commercial owner",
        elicitation=(
            "How much more than the cheapest tracked competitor can we charge before a "
            "shopper who is comparing walks away?"
        ),
        sensitivity=(1.00, 1.25),
        requires_local_elicitation=True,
    )
)

DEFAULT_MAX_CHANGES_PER_WINDOW: Final[int] = register(
    Parameter(
        name="DEFAULT_MAX_CHANGES_PER_WINDOW",
        value=4,
        provenance=Provenance.POLICY,
        rationale=(
            "Limits price churn within DEFAULT_CHANGE_WINDOW_DAYS. Protects against "
            "customer-visible thrash and against the system chasing noise in its own "
            "demand estimates."
        ),
        owner="Pricing policy owner",
        elicitation=(
            "How often can this price change in a month before customers perceive it as "
            "unstable, and before operations cannot keep labels in sync?"
        ),
        sensitivity=(1.0, 14.0),
        requires_local_elicitation=True,
    )
)

DEFAULT_CHANGE_WINDOW_DAYS: Final[int] = register(
    Parameter(
        name="DEFAULT_CHANGE_WINDOW_DAYS",
        value=28,
        provenance=Provenance.POLICY,
        rationale=(
            "Rolling window for the change-frequency guardrail. 28 rather than 30 so the "
            "window holds a whole number of weeks and does not alias against weekly "
            "demand seasonality, which would make the effective cap depend on start day."
        ),
        owner="Pricing policy owner",
        elicitation="Over what period should price stability be judged?",
        sensitivity=(7.0, 90.0),
        requires_local_elicitation=True,
    )
)

DEFAULT_MIN_INVENTORY_COVER_DAYS: Final[float] = register(
    Parameter(
        name="DEFAULT_MIN_INVENTORY_COVER_DAYS",
        value=14.0,
        provenance=Provenance.POLICY,
        rationale=(
            "Days of cover below which the inventory guard binds and the shadow price nu "
            "starts protecting scarce stock. Should be set from the actual replenishment "
            "lead time plus a safety buffer; 14.0 assumes a fortnightly cycle and is a "
            "starting point, not a measurement of any real supply chain."
        ),
        owner="Supply chain / inventory owner",
        elicitation=(
            "At what days-of-cover does selling a unit cheaply start costing you a sale "
            "you cannot replace before replenishment lands?"
        ),
        sensitivity=(7.0, 30.0),
        requires_local_elicitation=True,
    )
)

COMPETITOR_STALENESS_HOURS: Final[float] = register(
    Parameter(
        name="COMPETITOR_STALENESS_HOURS",
        value=24.0,
        provenance=Provenance.POLICY,
        rationale=(
            "Age beyond which a competitor observation triggers degradation rung 2 "
            "(WARN_STALE_COMPETITOR). Should track the scrape cadence: set below it and "
            "every decision degrades, set far above it and the ceiling guardrail is "
            "enforced against a price that no longer exists."
        ),
        owner="Pricing policy owner",
        elicitation=(
            "How old can a competitor price be before you would rather widen the movement "
            "bound than trust it?"
        ),
        sensitivity=(1.0, 72.0),
        requires_local_elicitation=True,
    )
)

DEFAULT_ALLOWED_PRICE_ENDINGS: Final[tuple[int, ...]] = register(
    Parameter(
        name="DEFAULT_ALLOWED_PRICE_ENDINGS",
        value=(95, 99),
        provenance=Provenance.LITERATURE,
        rationale=(
            "Psychological price endings, in whole cents. Field-experimental evidence finds "
            "a 9-ending raises unit sales relative to a nearby higher *or lower* price, so "
            "the ladder snaps to these endings rather than to arbitrary cents. The effect is "
            "context-dependent and strongest where the ending also signals a discount, so "
            "this is a defensible default rather than a universal law."
        ),
        citation=(
            "Anderson, E. T. & Simester, D. I. (2003), 'Effects of $9 Price Endings on Retail "
            "Sales: Evidence from Field Experiments', Quantitative Marketing and Economics "
            "1(1), 93-110."
        ),
    )
)

# ---------------------------------------------------------------------------
# Objective weights
# ---------------------------------------------------------------------------
#
# lambda and gamma are the two dials that decide what "a good price" means. They
# were previously described in README §2 and defined nowhere, which meant the
# objective had no weights at all. Neither can be estimated: both are exchange
# rates between quantities the business values differently, so each is solved
# from a stated trade rather than chosen as a digit.

#: Margin the operator will give up per 1.00 of modelled ΔCLV. See below.
_LAMBDA_MARGIN_SACRIFICED: Final[float] = 0.30
_LAMBDA_CLV_GAINED: Final[float] = 1.00

DEFAULT_CLV_WEIGHT_LAMBDA: Final[float] = register(
    Parameter(
        name="DEFAULT_CLV_WEIGHT_LAMBDA",
        value=lambda_from_tradeoff(_LAMBDA_MARGIN_SACRIFICED, _LAMBDA_CLV_GAINED),
        provenance=Provenance.POLICY,
        rationale=(
            "Exchange rate between contribution margin banked today and one unit of "
            "*modelled* future customer value. Solved from the stated trade rather than "
            "picked, via provenance.lambda_from_tradeoff(0.30, 1.00). The value being well "
            "below 1.0 is deliberate and carries information: ΔCLV is a modelled quantity "
            "over a 12-period horizon and is sensitive to both the discount rate and the "
            "survival specification, so it earns a haircut against cash. lambda = 0 is pure "
            "margin (clearance); a high lambda protects retention (acquisition phase)."
        ),
        owner="Head of pricing / commercial strategy",
        elicitation=(
            "How much contribution margin per unit would you give up today for 1.00 of "
            "modelled incremental customer lifetime value? Answer in currency, not as a "
            "weight; lambda is the ratio."
        ),
        sensitivity=(0.0, 1.0),
        requires_local_elicitation=True,
    )
)

_GAMMA_EXPECTED_SACRIFICED: Final[float] = 0.20
_GAMMA_CVAR_REDUCED: Final[float] = 1.00

DEFAULT_CVAR_WEIGHT_GAMMA: Final[float] = register(
    Parameter(
        name="DEFAULT_CVAR_WEIGHT_GAMMA",
        value=gamma_from_tail_tradeoff(_GAMMA_EXPECTED_SACRIFICED, _GAMMA_CVAR_REDUCED),
        provenance=Provenance.POLICY,
        rationale=(
            "Weight on the CVaR penalty: expected contribution the operator will give up to "
            "remove one unit of expected loss in the worst DEFAULT_CVAR_ALPHA tail. Solved "
            "via provenance.gamma_from_tail_tradeoff(0.20, 1.00). gamma = 0 is risk neutral, "
            "which is the wrong default for a system that can publish a price across a whole "
            "category at once: the tail outcome is correlated across SKUs, so the portfolio "
            "does not diversify it away."
        ),
        owner="Head of pricing / commercial strategy",
        elicitation=(
            "How much expected contribution would you give up to cut the worst-5% outcome by 1.00?"
        ),
        sensitivity=(0.0, 1.0),
        requires_local_elicitation=True,
    )
)

# ---------------------------------------------------------------------------
# Decision / simulation defaults
# ---------------------------------------------------------------------------

DEFAULT_MONTE_CARLO_DRAWS: Final[int] = register(
    Parameter(
        name="DEFAULT_MONTE_CARLO_DRAWS",
        value=2_000,
        provenance=Provenance.TECHNICAL,
        rationale=(
            "Monte-Carlo error on a mean falls as 1/sqrt(N), so 2,000 draws give a relative "
            "standard error near 2.2% — small against the spread between adjacent ladder "
            "rungs, which is what the simulation must resolve. CVaR at alpha=0.05 is the "
            "binding consideration rather than the mean: it is estimated from the worst ~100 "
            "draws, which is enough for a stable tail mean but not for a tail quantile. "
            "Raising alpha or tightening the ladder requires raising this."
        ),
    )
)

DEFAULT_CVAR_ALPHA: Final[float] = register(
    Parameter(
        name="DEFAULT_CVAR_ALPHA",
        value=0.05,
        provenance=Provenance.POLICY,
        rationale=(
            "Tail probability defining 'a bad outcome'. Recorded explicitly on every "
            "decision (cvar_alpha, not a cvar5 field name) because a record that assumes "
            "5% cannot describe a run that used 1%. Interacts with "
            "DEFAULT_MONTE_CARLO_DRAWS: at 2,000 draws, alpha below ~0.01 estimates the "
            "tail from too few samples to be stable."
        ),
        owner="Head of pricing / commercial strategy",
        elicitation="How rare does an outcome have to be before you stop budgeting for it?",
        sensitivity=(0.01, 0.10),
        requires_local_elicitation=True,
    )
)

DEFAULT_CLV_HORIZON_PERIODS: Final[int] = register(
    Parameter(
        name="DEFAULT_CLV_HORIZON_PERIODS",
        value=12,
        provenance=Provenance.POLICY,
        rationale=(
            "Horizon over which ΔCLV is accumulated. A finite horizon is a truncation, and "
            "a longer one is not automatically better: beyond the range where the survival "
            "model has observed data it extrapolates a baseline hazard, so extending the "
            "horizon adds modelled value rather than measured value."
        ),
        owner="Head of pricing / commercial strategy",
        elicitation=(
            "Over how many periods do you hold someone accountable for a pricing decision, "
            "and over how many periods do you actually have repurchase data?"
        ),
        sensitivity=(4.0, 36.0),
        requires_local_elicitation=True,
    )
)

DEFAULT_CLV_DISCOUNT_RATE: Final[float] = register(
    Parameter(
        name="DEFAULT_CLV_DISCOUNT_RATE",
        value=0.10,
        provenance=Provenance.POLICY,
        rationale=(
            "Per-period discount applied to future margin in ΔCLV. Should be the firm's "
            "own cost of capital, which is a finance input rather than anything this system "
            "can observe. ΔCLV is materially sensitive to it, which is why the dashboards "
            "are specified to show a sensitivity band instead of a single number."
        ),
        owner="Finance",
        elicitation="What is the firm's per-period cost of capital?",
        sensitivity=(0.05, 0.20),
        requires_local_elicitation=True,
    )
)

DEFAULT_SEED: Final[int] = register(
    Parameter(
        name="DEFAULT_SEED",
        value=20260817,
        provenance=Provenance.TECHNICAL,
        rationale=(
            "Fixed RNG seed. The specific value is arbitrary and that is the point: what "
            "matters is that it is pinned and recorded on every decision, because "
            "reconstructing a recommendation months later requires reproducing its Monte-Carlo "
            "draws exactly. Any decision whose seed varies is not reproducible."
        ),
    )
)

# ---------------------------------------------------------------------------
# Synthetic generator priors — NOT in the decision path
# ---------------------------------------------------------------------------
#
# These parameterise the synthetic panel that estimators are validated against.
# They never set a price. Their job is to be *recoverable* (so a failed recovery
# indicts the estimator, not the data) and *representative* (so recovery on
# synthetic data is evidence about real data). The second is why they carry
# citations rather than being convenient round numbers.

SYNTHETIC_ELASTICITY_MEAN: Final[float] = register(
    Parameter(
        name="SYNTHETIC_ELASTICITY_MEAN",
        value=-1.8,
        provenance=Provenance.LITERATURE,
        rationale=(
            "Mean true own-price elasticity in the generator. Sits essentially on the Tellis "
            "(1988) meta-analytic mean of -1.76 and is deliberately conservative against the "
            "larger Bijmolt et al. (2005) mean of -2.62, whose authors note sales elasticities "
            "grew in magnitude over the four decades their studies span. A conservative "
            "magnitude is the harder test: smaller true elasticities are easier for a naive "
            "estimator to confuse with confounding, so recovery here is stronger evidence."
        ),
        citation=(
            "Tellis, G. J. (1988), 'The Price Elasticity of Selective Demand: A Meta-Analysis "
            "of Econometric Models of Sales', Journal of Marketing Research 25(4), 331-341 "
            "(mean -1.76). Bijmolt, T. H. A., van Heerde, H. J. & Pieters, R. G. M. (2005), "
            "'New Empirical Generalizations on the Determinants of Price Elasticity', Journal "
            "of Marketing Research 42(2), 141-156 (mean -2.62 over 1,851 elasticities from 81 "
            "studies)."
        ),
        in_decision_path=False,
    )
)

SYNTHETIC_ELASTICITY_SD: Final[float] = register(
    Parameter(
        name="SYNTHETIC_ELASTICITY_SD",
        value=0.4,
        provenance=Provenance.LITERATURE,
        rationale=(
            "Cross-SKU dispersion of true elasticity. Deliberately narrower than the "
            "between-study dispersion in the meta-analyses, which mixes genuine heterogeneity "
            "with method variance (model specification alone is a large moderator in Bijmolt "
            "et al.). Widening this makes cold-start pooling look worse and per-SKU estimation "
            "look better, so it should be swept, not trusted: it is the single generator "
            "parameter most able to flatter the cold-start layer."
        ),
        citation=(
            "Bijmolt, van Heerde & Pieters (2005), JMR 42(2), 141-156 — moderator analysis of "
            "1,851 elasticities; method choices explain a substantial share of observed spread."
        ),
        in_decision_path=False,
    )
)

SYNTHETIC_REFERENCE_PRICE_HAZARD_THETA: Final[float] = register(
    Parameter(
        name="SYNTHETIC_REFERENCE_PRICE_HAZARD_THETA",
        value=-1.2,
        provenance=Provenance.PLACEHOLDER,
        rationale=(
            "Sensitivity of the repurchase hazard to the paid-price ratio (p / p_ref - 1). "
            "UNSOURCED, and deliberately not dressed up. The obvious literature to reach for "
            "is loss aversion, whose meta-analytic asymmetry coefficient sits nearer 1.25-1.45 "
            "than the folk value of 2 — but that coefficient describes the relative weight of "
            "losses and gains in a choice utility, which is a different quantity from a "
            "proportional hazard multiplier on repurchase timing. Citing it here would be "
            "false precision. This value is identifiable from real repurchase panels and "
            "should be estimated in phase 4 (retention) rather than assumed; until then, any "
            "ΔCLV computed on synthetic data inherits an invented number. Kept out of the "
            "decision path so the audit passes honestly rather than by relabelling."
        ),
        in_decision_path=False,
    )
)

SYNTHETIC_MARGIN_RANGE: Final[tuple[float, float]] = register(
    Parameter(
        name="SYNTHETIC_MARGIN_RANGE",
        value=(0.40, 0.55),
        provenance=Provenance.PLACEHOLDER,
        rationale=(
            "Category gross-margin range used to invent unit costs, both in the generator and "
            "in the UCI augmentation, since neither source carries COGS. UNSOURCED: real "
            "gross margins vary enormously by category and retailer, and this range was chosen "
            "as plausible for general merchandise rather than measured. Every margin figure "
            "downstream inherits it, which is why AssumptionSet travels with the augmented "
            "panel. Not in the decision path — it generates test data — but it is the "
            "assumption most likely to be mistaken for a measurement in a demo."
        ),
        in_decision_path=False,
    )
)

# ---------------------------------------------------------------------------
# Compute policy
# ---------------------------------------------------------------------------
#
# PrismPrice trains and scores its ML models on GPU where the library supports
# it. See prismprice.compute. The escape hatch exists so that CI and the
# pure-Python governance tests — which touch no ML code at all — can run on
# CPU-only runners.

ALLOW_CPU_ENV_VAR: Final[str] = "PRISMPRICE_ALLOW_CPU"


def cpu_fallback_allowed() -> bool:
    """True only when the CPU escape hatch is explicitly set (CI/test use)."""
    return os.environ.get(ALLOW_CPU_ENV_VAR, "").strip().lower() in {"1", "true", "yes"}
