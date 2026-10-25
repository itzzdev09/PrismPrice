# PrismPrice

[![CI](https://github.com/itzzdev09/PrismPrice/actions/workflows/ci.yml/badge.svg)](https://github.com/itzzdev09/PrismPrice/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: BUSL-1.1](https://img.shields.io/badge/license-BUSL--1.1-lightgrey.svg)](LICENSE)

**Dynamic pricing that optimises for the customer you keep, not just the sale you close.**

PrismPrice is a decision-support system for retail and e-commerce pricing. For every SKU it is designed to forecast demand, estimate price sensitivity, predict the effect of a price on repeat-purchase behaviour, and recommend the price that maximises **long-run contribution profit** — subject to hard guardrails on margin, competitiveness, inventory, fairness and price stability.

> Recommendations are decision support, not auto-published prices. Every recommendation ships with its assumptions, its uncertainty, the constraints that bound it, and a reason code explaining why it landed where it did.

---

## Overview

PrismPrice is a decision-support system for retail and e-commerce pricing. It combines demand estimates, price sensitivity, customer lifetime value, price optimization, and business guardrails. Recommendations include the inputs, constraints, and reason codes behind each decision; the system does not publish prices automatically.

The repository includes model evaluation against synthetic data, price optimization with business guardrails, explainable decision records, a FastAPI pricing service, and a browser workbench. Validate models and policy settings against representative business data before using recommendations operationally.

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

### Modelling layers

The modelling extras are opt-in; see section 7.1 for per-component device requirements.

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
3. **Elasticity is model output, not a request field.** Price requests do not accept a caller-supplied elasticity coefficient.

---

## 3. System architecture

```text
┌──────────────────────────────────────────────────────────────────────────┐
│  L7  OBSERVABILITY        KPIs · dashboards · drift · alerts · SLOs       │
├──────────────────────────────────────────────────────────────────────────┤
│  L6  SERVING              REST API · batch scoring · decision records    │
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
- **Demand model**: gradient-boosted quantile regression produces p10/p50/p90 demand estimates. Conformal calibration measures interval coverage on held-out data; isotonic projection preserves price monotonicity.
- **Causal Elasticity model** — Double Machine Learning (DML) / Orthogonal R-Learner that disentangles price from promotional confounders, returning a causal CI and confidence tag.
- **Retention & CLV model** — Deep Survival model (e.g., Cox Proportional Hazards) with time-varying covariates. Attributes churn risk strictly to the price-vs-reference shock, supplying $\Delta\text{CLV}(p)$.
- **Competitor model** — estimates competitor reaction functions (Game-Theoretic Response) to prevent destructive algorithmic price wars.

### L3 — Decision
Generate a discrete candidate ladder (snapped to psychological price endings — real retail prices end .95/.99, and threshold effects are real). Monte-Carlo each candidate through the demand distribution. Handle **cross-price effects within category** so a discount on one SKU is not booked as a win when it merely cannibalised its neighbour.

**Sequential markdown.** [`decision/robust_markdown.py`](src/prismprice/decision/robust_markdown.py) supports markdown planning under elasticity uncertainty using value- and regret-based CVaR objectives. See [docs/robust-markdown-evaluation.md](docs/robust-markdown-evaluation.md) for the method and evaluation results.

### L4 Governance

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
Double Machine Learning controls for recorded confounders, but unobserved confounders can still bias elasticity estimates.
- **Designed experiments** — switchback and store/geo splits by SKU-category.
- **Safe Contextual Bandits** — exploration with hard lower bounds, ensuring expected rewards never fall below $(1 - \alpha)$ of the baseline policy, explicitly budgeting exploration risk.
- **Off-policy evaluation** — inverse propensity scoring and doubly-robust estimates, so a candidate policy is scored against logged decisions before it ships.

### L6 — Serving
FastAPI endpoints serve single and bounded batch decisions; the browser workbench uses the same API. External integrations handle any price publication.

### L7 — Observability
KPIs, dashboards, feature/prediction drift, and alerting. See §6.

---

## 5. Reliability

A pricing system that is wrong is worse than one that is down — a bad price is published, transacts, and cannot be recalled. The design reflects that.

### Degradation Matrix

The API reports these decision states in each audit record:

| Level | Condition / Trigger | Action Taken | Governance Reason Code |
| --- | --- | --- | --- |
| **1** | Inputs and models available; no stale competitor feed | Return the highest-scoring feasible candidate | `OK_OPTIMAL` |
| **2** | Competitor observation is stale | Continue with an explicit stale-data warning | `WARN_STALE_COMPETITOR` |
| **4** | No feasible candidate | Hold the current price and record the fallback | `FALLBACK_RULE_ENGINE` |
| **5** | Model store is unavailable | Hold the current price and record the outage | `CRITICAL_MAINTAIN_PREV` |

`FALLBACK_POOLED_ELASTICITY` is reserved in the reason-code contract; the serving path currently emits levels 1, 2, 4, and 5.

The rung is derived from the reason code rather than stored alongside it, so the number and the code cannot disagree in an audit record.

### Other guarantees

- **Deterministic** — same inputs + same model versions + same seed $\Rightarrow$ same output, byte for byte. Non-negotiable for auditability.
- **Shadow $\to$ canary $\to$ rollout** — every policy change runs in shadow against live traffic, then on a small SKU slice, before broad release.
- **Immutable audit** — every decision reconstructible months later, including the exact model artefacts.
- **Circuit breakers**: abnormal recommendation distributions can mark a batch as not cleared for publication.
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

### PrismPrice web app

The browser workbench is served by the same FastAPI application as the pricing
API. From the repository root in PowerShell, create a project-local environment
and install the current compatible serving dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[serving]"
.\.venv\Scripts\python.exe -m prismprice.webapp
```

Open <http://127.0.0.1:8000>. Keep one local server running on port 8000; if it is
already running, use that instance instead of starting another. Stop it with
`Ctrl+C` in its terminal. Enter product cost, current price, and optional market
or stock context to request a recommendation. The workbench shows the candidate
ladder, the selected decision rung, each guardrail result, and a downloadable
JSON audit record. Interactive requests use `POST /decide`; service status comes
from `GET /health`.

The local command uses a historical **Online Retail II demo**. It includes a
dataset explorer with observed sales and model-review insights, and lets a
visitor select a real product to run through the workbench. The demand curve is
anchored to its recent product history and saved elasticity estimate; where a
SKU estimate did not meet the confidence checks, the demo uses the pooled
estimate. Unit costs use a labelled 45% margin assumption. The dataset ends in
2011, so these results are a worked example, not current pricing guidance.
Dataset source, cleaning, assumptions, and the saved run's findings are
documented in [docs/demo-dataset.md](docs/demo-dataset.md). The source is
Online Retail II from UCI, also available as a [Kaggle
mirror](https://www.kaggle.com/datasets/lakshmi25npathi/online-retail-dataset);
the original is CC BY 4.0 and its attribution is shown on the page.

It listens on localhost, allows anonymous decisions, and exposes API docs; do
not expose this demo server to the public internet.

For a deployment, set a strong `PRISMPRICE_API_KEY` of at least 32 characters
before creating the app. Also provide the permitted hostnames:

```python
from prismprice.serving.app import create_app

app = create_app(service, trusted_hosts=["pricing.example.com"])
```

`POST /decide` and `POST /decide/batch` require the `X-API-Key` header.
`GET /health` returns a limited readiness response; detailed diagnostics at
`GET /internal/health` require the same key. API docs are disabled by default.
POST request bodies are limited to 256 KiB, batches to 100 items, and SKU
identifiers to 128 characters. Terminate TLS and configure rate limits at the deployment
gateway. The browser workbench accepts an API key in its API access section and
keeps it in the page only until refresh.

`prismprice-web` is also installed as a command by `pip install -e
".[serving]"` and starts the same localhost demo.

### Run review dashboard

The generated run-review dashboard is a self-contained HTML artifact. Create it
from a saved pipeline run with:

```python
from prismprice.observability.dashboard import render_dashboard

render_dashboard("data/runs/real_run.json", "data/runs/dashboard.html")
```

Open `data/runs/dashboard.html` in a browser to inspect the publish verdict,
breaker reasons, degradation rungs, elasticity health, and per-SKU decisions.
The decision table supports:

- cross-filtering by degradation rung, elasticity confidence, direction, and SKU;
- live KPIs for the filtered view, including expected profit;
- CSV and JSON export of exactly the decisions currently in view;
- bookmarkable filters via the URL hash and a **Copy view link** control;
- light/dark themes, sortable columns, `/` to focus search, and `Esc` to reset.

The artifact has no JavaScript or stylesheet dependencies, so it can be shared
as a file without a running application server.

---

## 7. Tech stack

| Concern | Choice | Why |
| --- | --- | --- |
| **Language** | Python 3.10+ | Ecosystem for stats + serving in one runtime |
| **Core runtime** | Pydantic v2 *only* | The governance contracts need nothing else; the heavy stack is opt-in |
| **Compute** | CUDA for Torch-backed components; backend-aware gradient boosting | See §7.1 |
| **Modelling** | scikit-learn, LightGBM, EconML | Quantile regression, Double Machine Learning for causality |
| **Survival** | scikit-survival / PyTorch | Deep proportional hazard modeling for dynamic $\Delta\text{CLV}$ |
| **Serving** | FastAPI + Pydantic v2 | Typed contracts, generated OpenAPI docs |
| **Storage** | DuckDB for local analytical storage | Hosted database integration is not included |
| **Ops/Audit** | MLflow, Prefect, Docker | Model versions, batch refresh, one-command start |
| **Quality** | pytest, Hypothesis, ruff, mypy --strict | Property tests are the right tool for guardrails |

Dependencies are split into extras — `modelling`, `serving`, `dashboards`, `ops`, `dev` — so installing the governance layer does not pull three gigabytes of CUDA wheels.

### 7.1 GPU compute policy

**Model training and batch inference run on GPU wherever the library supports it, and the device actually used is recorded either way.** The policy is enforced per library, because "does this machine have a GPU" and "was this library built to use it" are different questions with different fixes.

```python
from prismprice.compute import require_gpu, lightgbm_device_params

device = require_gpu("estimation.retention")            # torch: raises on CPU
params = lightgbm_device_params("estimation.demand")    # boosting: best available, loudly
```

**Torch-backed components require CUDA.** Survival CLV, the embedding encoder, Monte-Carlo simulation and policy learning raise `GPUUnavailableError` rather than degrading. These are the workloads where a GPU is worth multiples of wall-clock time, and torch ships a CUDA build, so the requirement is one an operator can actually satisfy.

**Gradient-boosting components use CUDA only when their installed build provides it**; otherwise they train on CPU with a `RuntimeWarning`. This is independent of PyTorch CUDA availability. The actual backend and device are recorded with each artefact.

`compute_report()` reports machine-level CUDA availability and support for each compute backend separately:

```jsonc
{
  "device":   { "available": true,  "device_name": "NVIDIA GeForce RTX 3060 Laptop GPU", "cuda_version": "12.8" },
  "backends": {
    "lightgbm": { "available": false, "reason": "LightGBM 4.6.0 was not built with CUDA support ..." },
    "xgboost":  { "available": true,  "reason": "XGBoost CUDA available" }
  }
}
```

`gpu_report()` reports whether PyTorch is installed, whether its build includes CUDA, and whether a CUDA device is visible to the process.

The `PRISMPRICE_ALLOW_CPU=1` override is intended for CI and CPU-only development. Leave it unset in production when using components that require CUDA.

> **Install note.** The default PyPI `torch` wheel is CPU-only and will fail `require_gpu()`. Install from the CUDA index:
> ```bash
> pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch
> ```
> LightGBM has no CUDA wheel on PyPI at all; GPU support there requires a source build with `-DUSE_CUDA=1`, and at panel sizes in the low millions of rows CPU histogram training is usually faster anyway.

---

### 7.2 Parameter provenance

Every constant that can change a recommended price carries a record saying where it came from, declared in [`config.py`](src/prismprice/config.py) through [`provenance.register`](src/prismprice/provenance.py). CI fails if any decision-path constant is unsourced.

Each parameter is categorized by its source so operators can distinguish research evidence, business policy, technical constraints, placeholders, and measured values:

| Kind | Meaning | Record must include |
| --- | --- | --- |
| `LITERATURE` | From published research | A citation specific enough to check |
| `POLICY` | A business preference | Owner, the **trade** it encodes, and a sensitivity bracket |
| `TECHNICAL` | Forced by a machine or statistical constraint | The constraint |
| `PLACEHOLDER` | Unsourced | Permitted outside the decision path; a CI failure inside it |
| `MEASURED` | An estimate | **Rejected in configuration** — estimates belong in model artefacts |

**Policy weights.** The transaction-versus-customer-value weight is derived from an explicit business trade-off. For example:

```python
DEFAULT_CLV_WEIGHT_LAMBDA = lambda_from_tradeoff(margin_sacrificed=0.30, clv_gained=1.00)
```

Policy values include sensitivity brackets so their effect on recommendations can be evaluated. Synthetic-data priors are documented separately from production decision parameters.

**Calibration requirement.** The repurchase-hazard sensitivity currently uses a placeholder value of `-1.2`. Calibrate it against representative repurchase data before using CLV-based recommendations operationally.


---

## 8. Operational limitations

- **Real-world validation is still needed.** Evaluation metrics in this repository use synthetic data. Validate demand, elasticity, and CLV models against representative transaction and customer data before relying on their recommendations.
- **Input data has limits.** The UCI Online Retail dataset does not include unit costs, inventory, or competitor prices. Those inputs must come from the retailer's own systems or documented assumptions.
- **Data ingestion uses supplied datasets.** The project includes validation and preprocessing, but does not provide automated downloads from source systems.
- **Storage is local.** The implemented store uses DuckDB. A hosted database adapter and production deployment are not included.
- **Policy values need local review.** Margin floors, movement caps, competitor ceilings, and objective weights are documented defaults, not business-specific approvals.
- **CLV is modelled.** Its estimate depends on the discount rate, horizon, and repurchase-hazard calibration; inspect the sensitivity results before using it in decisions.
- **Fairness checks are limited.** The protected-data guardrail checks decision inputs; it does not establish equitable outcomes or replace a full fairness review.
- **Inventory protection depends on its shadow price.** Without a supplied inventory opportunity cost, the guardrail can enforce only the configured absolute floor.
- **Deployment credentials are shared API keys.** This is suitable for trusted integrations. A browser user can inspect credentials they enter, so use an identity-aware access layer for untrusted or multi-user public access. Terminate TLS and apply rate limits at the gateway.

---

## 9. Licence

**[Business Source License 1.1](LICENSE)** — source-available, not OSI open source.

| | |
| --- | --- |
| **Permitted now** | Evaluation, research, teaching, benchmarking, development, and shadow or offline analysis. Copy, modify and redistribute freely under these terms. |
| **Requires a commercial licence** | Any *Production Pricing Purpose* — using PrismPrice to generate, publish or otherwise set prices offered to customers in a live commercial setting. |
| **Change Date** | 2030-08-17, on which this version converts to **Apache 2.0** automatically. |

The earlier "MIT for non-production use" wording was self-contradictory: MIT is an unrestricted grant that already permits commercial production use, so it cannot carry that restriction. BUSL 1.1 is the licence that actually encodes the intended terms.

Dataset terms remain with their providers.
