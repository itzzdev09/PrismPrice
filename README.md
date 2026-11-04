# PrismPrice

[![CI](https://github.com/itzzdev09/PrismPrice/actions/workflows/ci.yml/badge.svg)](https://github.com/itzzdev09/PrismPrice/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: BUSL-1.1](https://img.shields.io/badge/license-BUSL--1.1-lightgrey.svg)](LICENSE)

**Retail pricing research, with a connected historical decision workbench.**

PrismPrice contains research modules for demand, causal price response, retention, portfolio effects, and pricing policy. Its connected local workbench is narrower: it uses a time-validated model of historical active-day sales to compare one SKU's candidate prices under explicit guardrails. The historical prices are observational, so the workbench does not establish the causal effect of changing a price.

> Recommendations are decision support, not auto-published prices. Every recommendation ships with its assumptions, its uncertainty, the constraints that bound it, and a reason code explaining why it landed where it did.

---

## Overview

The repository combines model research, candidate scoring, guardrails, and decision records. In the local workbench, a verified historical demand bundle feeds point-in-time features and a single-SKU candidate decision; the resulting evidence is stored in DuckDB. Causal elasticity, customer lifetime value, category substitution, retailer feeds, and publishing are not connected to that request path.

The repository includes model evaluation against synthetic data, price optimization with business guardrails, explainable decision records, a FastAPI pricing service, and a browser workbench. Validate models and policy settings against representative business data before using recommendations operationally.

### Architecture status

The local historical request path is connected end to end, but PrismPrice is not a production-connected pricing platform. Retailer source feeds, causal price-response evidence, approval operations, hosted storage, and a production model registry are not included. `InMemoryModelStore` remains a test adapter; the local webapp loads its gated bundle and persists decision records through DuckDB. See the [architecture review and modernization plan](docs/architecture-modernization.md) for current status, target design, and remaining integration work.

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

Pricing can fail in two directions, and most systems only guard one of them.

| Failure | What it looks like | What it costs |
| --- | --- | --- |
| **Priced too low** | Discount-led volume, margin erosion, unprofitable "growth", anchoring customers to a price you cannot sustain | Contribution profit; pricing power is very hard to win back |
| **Priced too high** | Conversion drops, baskets abandoned, customers defect to competitors and do not return | Revenue today *and* the entire future value of a lost customer |

The second cost is the one that gets missed. A price that wins today's basket but costs two future baskets is a **loss**, and a system that scores single transactions cannot see it.

The research objective makes that trade-off explicit. The current local workbench evaluates one SKU using expected contribution and downside risk; it does not model retained customer value or portfolio substitution.

---

## 2. The core idea

Most pricing engines maximise margin on the current transaction independently:

$$\max_{p} \quad \mathbb{E}[q(p)] \times (p - c)$$

The broader research objective scales this into **Category Portfolio Optimization**: transaction contribution plus customer relationship value and substitute-SKU effects, with a downside penalty:

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

### Target request path (not every component is connected)

The diagram below is the target component map. The connected local workbench currently runs only the historical, single-SKU path described after the model sections; modules shown for causal response, experimentation, portfolio effects, and publishing are not part of that request.

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

The current local workbench connects this narrower path:

```text
POST /decide
  -> point-in-time features from the historical SKU panel
  -> verified quantile bundle (P10/P50/P90 at each candidate price)
  -> single-SKU contribution and downside scoring
  -> hard guardrails
  -> durable DuckDB record with model/data/feature evidence
  -> reviewable result; no automatic price publishing
```

---

## 4. Layer detail

### L0 — Data foundation
The local DuckDB store ingests the historical Online Retail II transactions and builds a daily SKU panel. Live retailer systems, effective-dated costs, inventory, competitor prices, and promotions are not connected.

### L1 — Features & semantic layer
The connected workbench assembles SKU history strictly before `as_of`: observed demand, price history, reference price, seasonality, and provenance. Inventory, competitor, customer, embedding, and uncensoring features are not connected to this serving path.
- **No leakage** — features are computed as of decision time, never using the future.
- Missing sales days remain unknown because the source does not say whether a product was available.

### L2 — Estimation
- **Demand model**: the local bundle contains a fitted quantile model, temporal calibration metrics, data snapshot, artifact digest, and supported SKU list.
- **Challenger comparison**: the bundle builder compares LightGBM and scikit-learn quantile models on identical expanding time folds and refuses to publish when neither passes interval and tail calibration gates.
- **Causal elasticity, retention/CLV, and competitor response**: research modules exist, but they are not consumed by the connected workbench.
- The demand model learns historical associations; observed prices were not randomized, so its curve does not identify the causal effect of changing a price.

#### Demand model second opinion

The local bundle builder runs both quantile candidates on the **same expanding, time-ordered folds**. It records each candidate's WAPE, pinball loss, interval coverage, lower-tail coverage, and fold scores, then selects among candidates that pass the calibration gates. This supports the historical workbench; it is not retailer-specific model approval.

```python
from prismprice.estimation import compare_demand_models

# `training_frame` comes from FeatureBuilder.training_frame() and carries an
# (sku, as_of) index, so every comparison fold respects the time boundary.
comparison = compare_demand_models(training_frame, n_folds=4)
print(comparison.as_dict())
```

The local bundle builder stores this comparison in the model manifest and makes the selected version visible in the workbench and each decision record.

### L3 — Decision
The connected service scores one SKU's candidate ladder using forecast demand, an operator-supplied unit-cost assumption, expected contribution, and downside risk. It does not model category substitution or portfolio economics; those research modules are offline.

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

The connected local service writes a durable DuckDB record with request inputs, model and data versions, feature snapshot, candidate forecasts and scores, guardrail results, seed, and selected price. The ledger is local and does not imply that an operator approved or applied the price.

### L5 — Learning
Experiment design, contextual bandits, and off-policy evaluation are research modules. They are not connected to the local decision path and do not select prices.

### L6 — Serving
The local FastAPI app loads the verified historical bundle, assembles point-in-time SKU features, scores a request, applies guardrails, persists its record to DuckDB, and returns it for human review. It has no retailer price publisher.

### L7 — Observability
The app reports model readiness and temporal validation evidence. Production drift monitoring, outcome joins, alert routing, and operational SLOs are not connected. See §6 for target KPIs.

---

## 5. Reliability

The local workbench does not publish prices. Its outage path holds the supplied current price and records the failure; the demo has no authoritative feed for the last applied retailer price.

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

- **Repeatable scoring** — candidate forecasts and scores use a pinned model version and random seed; record timestamps naturally differ.
- **Durable local audit** — each returned scenario is appended to the DuckDB decision ledger with its model and feature evidence.
- Shadow rollout, publication circuit breakers, and catalogue blast-radius controls are future deployment work, not guarantees of the local app.

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
.\.venv\Scripts\python.exe -m pip install -e ".[demo]"
.\.venv\Scripts\python.exe scripts/ingest_uci.py path\to\online_retail_II.xlsx
.\.venv\Scripts\python.exe scripts/build_demand_bundle.py
.\.venv\Scripts\python.exe -m prismprice.webapp
```

Open <http://127.0.0.1:8000>. Keep one local server running on port 8000; if it is
already running, use that instance instead of starting another. Stop it with
`Ctrl+C` in its terminal. Enter product cost, current price, and optional market
or stock context to request a recommendation. The workbench shows the candidate
ladder, the selected decision rung, each guardrail result, and a downloadable
JSON audit record. Interactive requests use `POST /decide`; service status comes
from `GET /health`.

The local command uses a historical **Online Retail II workbench**. It compares
two quantile model candidates on rolling time folds, writes a digest-checked
bundle only after calibration gates pass, and connects that bundle to
point-in-time features, the decision engine, hard guardrails, and a durable
DuckDB decision ledger. The page shows candidate demand intervals, expected
contribution, downside, model validation, and saved evidence. Unit costs are
operator-supplied assumptions. Unsupported SKUs or unavailable bundles hold
the supplied current price and record why; there is no illustrative demand
curve.

The dataset ends in 2011 and contains positive sales on observed active SKU-days
only. Missing sales days are not zero-filled, stockout availability is unknown,
and prices were not randomized. Treat the forecast as a historical research
scenario, not current demand or a causal estimate of a price change. Causal
elasticity, CLV, cross-SKU substitution, live retailer data, and price publishing
are outside this connected path. Dataset source, cleaning, assumptions, and
model details are documented in [docs/demo-dataset.md](docs/demo-dataset.md).
The source is Online Retail II from UCI, also available as a [Kaggle
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

`prismprice-web` is installed as a command by `pip install -e
".[demo]"` and starts the same localhost workbench. Build the model bundle first
with `scripts/build_demand_bundle.py`.

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

- **Retailer validation is still needed.** The workbench bundle is evaluated on historical UCI Online Retail II only. Its temporal metrics do not establish current-market accuracy, causal price response, or performance on a retailer's data.
- **Input data has limits.** The UCI Online Retail dataset does not include unit costs, inventory, or competitor prices. Those inputs must come from the retailer's own systems or documented assumptions.
- **Data ingestion uses supplied datasets.** The project includes validation and preprocessing, but does not provide automated downloads from source systems.
- **Storage is local.** The implemented store uses DuckDB. A hosted database adapter and production deployment are not included.
- **Policy values need local review.** Margin floors, movement caps, competitor ceilings, and objective weights are documented defaults, not business-specific approvals.
- **CLV is not connected to the workbench.** Retention research code exists but does not affect its candidate scores.
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
