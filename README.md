# PrismPrice

[![CI](https://github.com/itzzdev09/PrismPrice/actions/workflows/ci.yml/badge.svg)](https://github.com/itzzdev09/PrismPrice/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: BUSL-1.1](https://img.shields.io/badge/license-BUSL--1.1-lightgrey.svg)](LICENSE)

**Dynamic pricing that optimises for the customer you keep, not just the sale you close.**

PrismPrice is a decision-support system for retail and e-commerce pricing. For every SKU it is designed to forecast demand, estimate price sensitivity, predict the effect of a price on repeat-purchase behaviour, and recommend the price that maximises **long-run contribution profit** — subject to hard guardrails on margin, competitiveness, inventory, fairness and price stability.

> Recommendations are decision support, not auto-published prices. Every recommendation ships with its assumptions, its uncertainty, the constraints that bound it, and a reason code explaining why it landed where it did.

---

## 0. Status

This document specifies the full system. **Three layers of it are built.** The table says which, so you can tell the design from the code before you clone it.

| Layer | Status | What exists |
| --- | --- | --- |
| **L0 Data foundation** | 🟢 **Built & tested** | Schema contracts with 12 quality-gate codes, quarantine on failure, synthetic panel generator with known ground truth, deterministic UCI augmentation with a declared assumption set |
| **L1 Features** | 🟢 **Built & tested** | Point-in-time feature assembly with a proven leakage guarantee, right-censored Tobit demand un-censoring (removes 78% of censoring error on synthetic truth), cold-start elasticity priors from embedding neighbours |
| **L4 Governance** | 🟢 **Built & tested** | All 9 guardrails, reason codes, structured verdicts, ladder feasibility filtering, degradation rungs 1/2/4, immutable decision-record contract |
| L3 Decision | 🟡 Partial | Feasibility filtering and the fallback path are built; the objective, ladder generation and Monte-Carlo simulation are not |
| L2 Estimation | ⚪ Spec only | Demand / DML elasticity / survival CLV / competitor game |
| L5 Learning | ⚪ Spec only | Experiments, safe bandits, OPE |
| L6 Serving | ⚪ Spec only | FastAPI service, batch scoring, price feed |
| L7 Observability | ⚪ Spec only | KPI definitions in [docs/metrics.md](docs/metrics.md) |

201 tests, property-based where the guarantee is universal and scored against known ground truth where it is statistical. Everything marked *spec only* is a design that has been thought through and written down, not code that runs. The roadmap in §9 is the build order.

**Why governance first.** It is the layer where a defect is unrecoverable — a bad price publishes, transacts, and cannot be recalled — and it is the only layer that can be proven correct without any data at all.

**Why the synthetic generator second.** It is the measuring instrument for everything after it. On real data a wrong elasticity and a right one look identical; against a known true parameter they do not. The generator confounds prices on purpose (a naive estimator is off by ~1.4) and censors demand on purpose, so each estimator has something real to prove.

---

## Quickstart

```bash
pip install -e ".[dev]"
```

```bash
pytest
```

Evaluate a candidate price against every guardrail:

```python
from datetime import datetime, timezone
from prismprice.governance import GuardrailEngine, PriceRequest

request = PriceRequest(
    sku="SKU-001",
    as_of=datetime.now(timezone.utc),
    current_price=32.00,
    unit_cost=15.00,
    competitor_price=31.50,
    inventory_cover_days=21.0,
    margin_floor_pct=0.20,
    movement_cap_pct=0.10,
)

engine = GuardrailEngine()
verdict = engine.evaluate_candidate(30.95, request)

print(verdict.feasible)             # True
print(verdict.binding_constraints)  # []
print(verdict.min_slack)            # 0.0672 — the movement cap is the tightest constraint

# Filter a whole ladder; falls back explicitly if nothing is feasible.
ladder = engine.evaluate_ladder([28.95, 30.95, 32.95, 35.95], request)
print(ladder.feasible_prices)       # [28.95, 30.95, 32.95]  — 35.95 breaches the 10% move cap
print(ladder.degradation_rung)      # DegradationRung.OK_OPTIMAL
```

Every rejection carries a reason code, the observed value, the limit it breached, and the size of the breach:

```python
verdict = engine.evaluate_candidate(14.00, request)
for result in verdict.results:
    if result.binding:
        print(result.reason_code.value, "|", result.detail, "|", result.slack)
# PP-G001 | Price 14.0000 is below floor 15.0000            | -1.0
# PP-G002 | Price 14.0000 is below floor 18.0000            | -4.0
# PP-G005 | Relative price change 0.5625 exceeds ceiling 0.1000 | -0.4625
# PP-G008 | Price 14.00 ends .00; allowed endings are .95, .99  | None
```

Boolean constraints (ladder compliance, fairness) report no `slack` — there is no meaningful distance to a threshold — which keeps `min_slack` describing the tightest *numeric* limit.

### Modelling layers (GPU required)

The modelling extras are opt-in, and **the ML stack runs on GPU only** — see §8.1.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

```bash
pip install -e ".[modelling,serving]"
```

---

## 1. The problem this solves

Pricing fails in two directions, and most systems only guard one of them.

| Failure | What it looks like | What it costs |
| --- | --- | --- |
| **Priced too low** | Discount-led volume, margin erosion, unprofitable "growth", anchoring customers to a price you cannot sustain | Contribution profit; pricing power is very hard to win back |
| **Priced too high** | Conversion drops, baskets abandoned, customers defect to competitors and do not return | Revenue today *and* the entire future value of a lost customer |

The second cost is the one that gets missed. A price that wins today's basket but costs two future baskets is a **loss**, and a system that scores single transactions cannot see it.

PrismPrice makes that trade-off explicit in the objective function.

---

## 2. The core idea

Most pricing engines maximise margin on the current transaction independently:

$$\max_{p} \quad \mathbb{E}[q(p)] \times (p - c)$$

PrismPrice scales this into a **Category Portfolio Optimization** problem. It maximises the transaction plus its effect on the customer relationship, accounts for network cannibalization across substitute SKUs, and penalises downside risk:

$$\max_{\mathbf{p} \in \mathcal{P}} \sum_{i=1}^{K} \left( \mathbb{E}[q_i(\mathbf{p})] \cdot (p_i - c_i - \nu_i) + \lambda_i \mathbb{E}[\Delta\text{CLV}_i(p_i)] \right) - \gamma \cdot \text{CVaR}_{\alpha}(\mathbf{p})$$

Where:

| Term | Meaning | Estimated by |
| --- | --- | --- |
| $\mathbf{p}$ | The price vector for all $K$ SKUs in a category, accounting for cross-elasticity | L3 Optimiser |
| $q_i(\mathbf{p})$ | Demand for SKU $i$ given category prices, as a distribution, not a point | Demand + Causal Elasticity model |
| $c_i$ | Fully-loaded unit cost (COGS + landed + fulfilment) | Cost service / assumption set |
| $\nu_i$ | Inventory Opportunity Cost (Shadow Price). Scales as stock nears zero to prevent selling out too cheaply | Inventory / Replenishment model |
| $\Delta\text{CLV}_i(p_i)$ | Change in discounted expected future margin from customers exposed to $p_i$ | Deep Survival / CLV model |
| $\lambda_i$ | Strategic weight on relationship vs. transaction. $\lambda=0$ is pure margin; higher $\lambda$ protects retention | Business policy, per category |
| $\text{CVaR}_{\alpha}$ | Conditional Value at Risk. The expected shortfall of the worst $\alpha\%$ profit outcomes, superior to heuristic bounds | Monte-Carlo over demand draws |

**Why $\lambda$ matters.** It is the single dial that encodes commercial strategy. Acquisition-phase categories run high $\lambda$; clearance runs $\lambda=0$. Making it explicit and configurable is what turns a model into a business tool.

### Three design decisions worth calling out

1. **Optimise expected profit, not profit at expected demand.** Profit is non-linear in demand once inventory caps and stockouts bind, so by Jensen's inequality these are different prices — and the gap systematically biases toward over-aggressive pricing. PrismPrice samples the demand distribution per candidate price and selects on the resulting profit distribution.
2. **Causal Elasticity over Correlation.** The system isolates true price response using Double Machine Learning (DML) to control for confounding variables (e.g., seasonality, promo calendars) that typically ruin observational elasticity estimates.
3. **Elasticity is estimated, never supplied.** The system produces its own elasticity with a confidence interval and a method tag. Callers may override it, but the default is that the system knows. Anything else moves the hard part of the problem onto the caller.

---

## 3. System architecture

```text
┌──────────────────────────────────────────────────────────────────────────┐
│  L7  OBSERVABILITY        KPIs · dashboards · drift · alerts · SLOs       │
├──────────────────────────────────────────────────────────────────────────┤
│  L6  SERVING              REST API · batch scoring · price feed export    │
├──────────────────────────────────────────────────────────────────────────┤
│  L5  LEARNING             experiments · safe bandits · off-policy eval    │
├──────────────────────────────────────────────────────────────────────────┤
│  L4  GOVERNANCE           guardrails · compliance · audit · reason codes  │
├──────────────────────────────────────────────────────────────────────────┤
│  L3  DECISION             ladder · portfolio simulation · optimiser       │
├──────────────────────────────────────────────────────────────────────────┤
│  L2  ESTIMATION           demand · causal elasticity · survival · game    │
├──────────────────────────────────────────────────────────────────────────┤
│  L1  FEATURES             feature store · embeddings · un-censoring       │
├──────────────────────────────────────────────────────────────────────────┤
│  L0  DATA FOUNDATION      ingestion · contracts · quality gates · lineage │
└──────────────────────────────────────────────────────────────────────────┘
```

Each layer depends only on the one below it and is independently testable. That is what makes the thing buildable step by step and debuggable when a number looks wrong.

### Request path for a single decision

```text
  price request (sku, date, context)
        │
        ▼
  [L1] assemble features ──────────► feature staleness check ──► degrade if stale
        │
        ▼
  [L2] demand distribution + causal elasticity CI + survival retention + competitor game model
        │
        ▼
  [L3] generate candidate ladder ──► Monte-Carlo portfolio simulation ──► score J(p) & CVaR
        │
        ▼
  [L4] apply guardrails ──► feasible set ──► if empty: fall back, emit reason code
        │
        ▼
  [L5] exploration check ──► safe bandit may substitute exploratory price (budgeted)
        │
        ▼
  [L4] write immutable decision record (inputs, model versions, scores, reasons)
        │
        ▼
  recommendation + explanation + confidence
```

---

## 4. Layer detail

### L0 — Data foundation
Ingest transactions, inventory, costs, competitor observations, web exposure and promotions. Every source has a **schema contract** and quality gates (nullability, ranges, referential integrity, freshness). Bad data fails loudly at the boundary rather than silently producing a wrong price.

### L1 — Features & semantic layer
Point-in-time-correct feature assembly: trailing demand, price history, reference price, promo depth, seasonality, inventory cover, competitor gap, customer cohort mix.
- **No leakage** — features are computed as of decision time, never using the future.
- **Demand Un-censoring** — applies Tobit/Kaplan-Meier adjustments to historical sales to estimate true demand during past stockout periods.
- **Cold Start Embeddings** — uses LLM text/visual embeddings (CLIP/BERT) to map new SKUs to historical priors of nearest-neighbor items.

### L2 — Estimation
- **Demand model** — gradient-boosted quantile regression producing p10/p50/p90, not a point.
- **Causal Elasticity model** — Double Machine Learning (DML) / Orthogonal R-Learner that disentangles price from promotional confounders, returning a causal CI and confidence tag.
- **Retention & CLV model** — Deep Survival model (e.g., Cox Proportional Hazards) with time-varying covariates. Attributes churn risk strictly to the price-vs-reference shock, supplying $\Delta\text{CLV}(p)$.
- **Competitor model** — estimates competitor reaction functions (Game-Theoretic Response) to prevent destructive algorithmic price wars.

### L3 — Decision
Generate a discrete candidate ladder (snapped to psychological price endings — real retail prices end .95/.99, and threshold effects are real). Monte-Carlo each candidate through the demand distribution. Handle **cross-price effects within category** so a discount on one SKU is not booked as a win when it merely cannibalised its neighbour.

### L4 — Governance  🟢 built

Guardrails, applied as a hard feasibility filter with a reason code on every rejection. All nine are implemented in [`governance/guardrails.py`](src/prismprice/governance/guardrails.py):

| Code | Guardrail | Rule | Notes |
| --- | --- | --- | --- |
| `PP-G001` | **Absolute floor** | $p \ge c$ | Never knowingly below cost |
| `PP-G002` | **Margin floor** | $p \ge c \times (1 + m_{\min})$ | Minimum margin per category |
| `PP-G003` | **Regulatory anchor** | $p_{\text{promo}} \le \min(p_{t-30 \to t})$ | EU Omnibus Directive. **Fails closed** — a promo with no 30-day history is rejected, not passed |
| `PP-G004` | **Competitive ceiling** | $p \le \kappa \times p_{\text{competitor}}$ | Cap relative to market competitor |
| `PP-G005` | **Movement cap** | $\|\Delta p\| / p_{\text{prev}} \le \delta$ | Two-sided, per step |
| `PP-G006` | **Change frequency** | $\le N$ changes | Scores the *prospective* change, not the historical count |
| `PP-G007` | **Inventory guard** | $p \ge c + \nu$ when cover $< $ threshold | Shadow price binds only below the days-of-cover floor |
| `PP-G008` | **Ladder compliance** | ending $\in \{.95, .99\}$ | Configurable; disabled by passing `None` |
| `PP-G009` | **Fairness** | Identical price for identical context | Structural check that the decision context carries no identity attribute or proxy — see below |

Three properties make this more than a list of `if` statements:

- **Hard, with no business slack.** The only numeric tolerance is an allowance for IEEE-754 representation error (`1e-9` relative), so `10.00 × 1.15` still admits a price of exactly `11.50` while a price a hundredth of a cent short is rejected.
- **`NOT_APPLICABLE` is not `PASSED`.** "Checked, fine" and "nothing to check" are separate statuses, because only the first is evidence of compliance. A missing-data pass that reads as a compliance pass is how a regulator-facing audit log becomes fiction.
- **Fairness is a real predicate.** `PP-G009` inspects the request contract itself: it fails if any field name matches a protected attribute or a known proxy (identity, geography, income, loyalty tier, cohort), and fails if the model stops forbidding unknown fields. It cannot be satisfied by a system that personalises, and it cannot silently pass — there is a test that adds `customer_id` and asserts rejection.

Plus an immutable decision log: inputs, model versions, candidate scores, binding constraints, seed, final price, approver.

### L5 — Learning
The honest answer to "you cannot get causal elasticity from observational data" is not to document the limitation — it is to build the mechanism that earns it.
- **Designed experiments** — switchback and store/geo splits by SKU-category.
- **Safe Contextual Bandits** — exploration with hard lower bounds, ensuring expected rewards never fall below $(1 - \alpha)$ of the baseline policy, explicitly budgeting exploration risk.
- **Off-policy evaluation** — inverse propensity scoring and doubly-robust estimates, so a candidate policy is scored against logged decisions before it ships.

### L6 — Serving
FastAPI decision endpoint, batch scoring for full catalogue refresh, and a price feed export. Idempotent by `(sku, date, policy_version)`.

### L7 — Observability
KPIs, dashboards, feature/prediction drift, and alerting. See §6.

---

## 5. Reliability

A pricing system that is wrong is worse than one that is down — a bad price is published, transacts, and cannot be recalled. The design reflects that.

### Degradation Matrix

The system always returns a price, and always documents which rung it used:

| Level | Condition / Trigger | Action Taken | Governance Reason Code | Built |
| --- | --- | --- | --- | --- |
| **1** | Full health | Optimal portfolio decision | `OK_OPTIMAL` | 🟢 |
| **2** | Stale competitor feed | Widen movement bound $\delta$ to limit exposure | `WARN_STALE_COMPETITOR` | 🟢 detection |
| **3** | Elasticity CI width $> \tau_{\max}$ | Fall back to category pooled mean elasticity | `FALLBACK_POOLED_ELASTICITY` | ⚪ needs L2 |
| **4** | Solver / MC non-convergent, or empty feasible set | Rule-based anchor (Cost-plus + competitor band) | `FALLBACK_RULE_ENGINE` | 🟢 empty-set path |
| **5** | Total model failure / 5xx | Return last known good price or $p_{\text{prev}}$ | `CRITICAL_MAINTAIN_PREV` | ⚪ needs L6 |

The rung is derived from the reason code rather than stored alongside it, so the number and the code cannot disagree in an audit record.

### Other guarantees

- **Deterministic** — same inputs + same model versions + same seed $\Rightarrow$ same output, byte for byte. Non-negotiable for auditability.
- **Shadow $\to$ canary $\to$ rollout** — every policy change runs in shadow against live traffic, then on a small SKU slice, before broad release.
- **Immutable audit** — every decision reconstructible months later, including the exact model artefacts.
- **Circuit breakers** — abnormal recommendation distributions (mass movement in one direction) halt publication and page a human.
- **Blast-radius limits** — caps on how many SKUs may change per run and on aggregate basket-level price movement.

---

## 6. KPIs and dashboards

Four dashboards, four audiences. Every metric has an owner and a threshold.

- **Executive** — contribution profit vs. plan, gross margin %, revenue, realised vs. recommended price adherence, retention cohort curves, price perception index.
- **Commercial / category manager** — per-SKU recommendation and reason, margin bridge (volume vs. price vs. mix), guardrail bind rate, competitor position, inventory cover and markdown risk.
- **Data science** — demand WAPE/MAPE by horizon, quantile calibration (are p10/p90 honest?), causal elasticity CI width, drift (PSI), off-policy uplift estimates with confidence bands.
- **Engineering** — decision latency p50/p95/p99, error rate, feature freshness, degradation-rung distribution, fallback frequency, throughput.

**The headline metric** is contribution profit per customer over a rolling 90 days — because it is the one number that moves only when both halves of the trade-off are handled well.

Full definitions, thresholds and alerting rules: [docs/metrics.md](docs/metrics.md). What to do when one of them fires: [docs/playbooks.md](docs/playbooks.md).

---

## 7. Repository layout

**Target** layout. `🟢` exists today; the rest arrive with their phase in §9.

```text
prismprice/
├── src/prismprice/
│   ├── config.py       🟢 tolerances, governance defaults, policy dials
│   ├── compute.py      🟢 GPU-only device policy (§8.1)
│   ├── governance/     🟢 guardrails, reason codes, decision-record contracts
│   ├── data/           🟢 schema contracts, quality gates, synthetic truth, UCI augmentation
│   ├── features/       🟢 point-in-time assembly, Tobit un-censoring, cold-start priors
│   ├── estimation/     ⚪ demand, causal elasticity (DML), survival CLV, competitor game
│   ├── decision/       ⚪ ladder, portfolio simulation, objective, optimiser
│   ├── learning/       ⚪ experiments, safe bandits, off-policy evaluation
│   ├── api/            ⚪ FastAPI service
│   └── cli/            ⚪ build / train / backtest / score
├── tests/              🟢 unit, property, contract  (golden & backtest to come)
├── docs/               🟢 architecture, data & modelling, metrics, playbooks
├── .github/workflows/  🟢 CI: lint, types, tests on 3.10-3.12
├── dashboards/         ⚪ Streamlit apps (exec, commercial, DS, eng)
├── notebooks/          ⚪ exploratory analysis, kept out of the import path
├── infra/              ⚪ Docker Compose, migrations
└── artifacts/          ⚪ trained models, metrics, decision logs (gitignored)
```

---

## 8. Tech stack

| Concern | Choice | Why |
| --- | --- | --- |
| **Language** | Python 3.10+ | Ecosystem for stats + serving in one runtime |
| **Core runtime** | Pydantic v2 *only* | The governance contracts need nothing else; the heavy stack is opt-in |
| **Compute** | CUDA GPU, enforced | See §8.1 — no silent CPU fallback |
| **Modelling** | scikit-learn, LightGBM, EconML | Quantile regression, Double Machine Learning for causality |
| **Survival** | scikit-survival / PyTorch | Deep proportional hazard modeling for dynamic $\Delta\text{CLV}$ |
| **Serving** | FastAPI + Pydantic v2 | Typed contracts, generated OpenAPI docs |
| **Storage** | DuckDB (local) $\to$ Postgres (deployed) | Same SQL both ways; no rewrite to productionise |
| **Ops/Audit** | MLflow, Prefect, Docker | Model versions, batch refresh, one-command start |
| **Quality** | pytest, Hypothesis, ruff, mypy --strict | Property tests are the right tool for guardrails |

Dependencies are split into extras — `modelling`, `serving`, `dashboards`, `ops`, `dev` — so installing the governance layer does not pull three gigabytes of CUDA wheels.

### 8.1 GPU-only compute policy

**All model training and batch inference runs on GPU.** Silent CPU fallback is prohibited and enforced in code, not by convention:

```python
from prismprice.compute import require_gpu

device = require_gpu("estimation.elasticity")   # raises GPUUnavailableError on CPU
```

Two reasons this is a hard failure rather than a warning:

1. **Reproducibility is a stated guarantee.** Determinism (README §5) means same inputs + same model versions + same seed ⇒ same output. Kernels and reduction orders differ between CPU and GPU, so a run that quietly switches device can no longer be reconstructed against the artefacts named in its own decision log.
2. **Silent degradation hides itself.** LightGBM warns and trains on CPU when its GPU build is missing; a nightly catalogue refresh then takes 40× longer and blows its window with nothing in the logs but a warning nobody reads. `lightgbm_device_params()` calls `require_gpu()` first, converting that into the same hard failure everything else gets.

`gpu_report()` distinguishes the three failure modes that otherwise look identical — torch missing, torch built without CUDA, CUDA present but no visible device — because each needs a different fix.

The single escape hatch is `PRISMPRICE_ALLOW_CPU=1`, for CI runners and the pure-Python governance tests. It emits a `RuntimeWarning` naming the component and stating that results are not reproducible against GPU-trained artefacts. It is set in CI ([ci.yml](.github/workflows/ci.yml)) and should never be set on a machine that produces prices.

> **Install note.** The default PyPI `torch` wheel is CPU-only and will fail `require_gpu()`. Install from the CUDA index:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu124
> ```

---

## 9. Build roadmap

Each phase ends with something that runs, is tested, and is pushed.

Governance (phase 6) was built first, out of order, for the reason given in §0.

| Phase | Status | Delivers | Done when |
| --- | --- | --- | --- |
| **0 Foundation** | 🟢 done | Repo, CI, config, compute policy, packaging | `pytest` green in CI on push |
| **6 Governance** | 🟢 done | All 9 guardrails, reason codes, audit contracts | Property tests prove guardrails never violated |
| **1 Data & Features** | 🟢 done | Contracts, quality gates, synthetic truth, un-censoring, cold-start priors, leakage tests | Quality report generated; leakage test passes; un-censoring beats the naive series on known truth |
| **2 Demand** | ⚪ next | Quantile demand model + calibration | p10/p90 coverage within tolerance |
| **3 Causal Elasticity** | ⚪ | DML implementation controlling for confounders | Recovers known true elasticity on synthetic data |
| **4 Retention** | ⚪ | Time-varying repurchase hazard + $\Delta\text{CLV}$ | Cohort curves reproduce holdout |
| **5 Decision** | ⚪ | Category solver, CVaR penalty, Monte-Carlo | Beats cost-plus and competitor-match on backtest |
| **7 Serving** | ⚪ | API + batch + degradation matrix | Contract tests + chaos test on model outage |
| **8 Learning** | ⚪ | Safe bandits, OPE | OPE recovers known policy value on synthetic data |

Validating each model against **synthetic data with known ground truth** before trusting it on real data is what separates this from a project that merely produces plausible numbers.

---

## 10. Honest limitations

Stated plainly, because a buyer or reviewer will find them anyway and the credibility is worth more than the claim.

- **Most of this is specification, not code.** See §0. The objective function, the estimators and the serving layer are designed and documented; they are not written. Treat every present-tense description of L0–L3 and L5–L7 as intent.
- **Unobserved Confounders.** While Double Machine Learning handles observed confounders (seasonality, promo flags) better than raw regression, unrecorded variables (e.g., a competitor running an un-tracked radio ad) will still bias elasticity estimates.
- **The public UCI dataset has no cost, inventory, or competitor columns.** Margin requires an assumed cost model, and those assumptions are declared in [docs/data-and-modelling.md](docs/data-and-modelling.md), not buried.
- **$\Delta\text{CLV}$ is a modelled quantity**, sensitive to the discount rate and horizon. Both are explicit configuration, and the dashboards show sensitivity bands rather than a single number.
- **Fairness here means non-discrimination in price setting.** `PP-G009` proves the decision context is identity-free; it does not prove outcomes are equitable. It is not a full algorithmic-fairness audit, and its proxy denylist is a judgement call, not a legal standard.
- **The inventory guard is only as good as $\nu$.** With no shadow price supplied it degrades to the absolute floor, and says so in its own detail string rather than pretending to bind.

---

## 11. Licence

**[Business Source License 1.1](LICENSE)** — source-available, not OSI open source.

| | |
| --- | --- |
| **Permitted now** | Evaluation, research, teaching, benchmarking, development, and shadow or offline analysis. Copy, modify and redistribute freely under these terms. |
| **Requires a commercial licence** | Any *Production Pricing Purpose* — using PrismPrice to generate, publish or otherwise set prices offered to customers in a live commercial setting. |
| **Change Date** | 2030-08-17, on which this version converts to **Apache 2.0** automatically. |

The earlier "MIT for non-production use" wording was self-contradictory: MIT is an unrestricted grant that already permits commercial production use, so it cannot carry that restriction. BUSL 1.1 is the licence that actually encodes the intended terms.

Dataset terms remain with their providers.
