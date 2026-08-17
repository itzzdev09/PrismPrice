# PrismPrice — Architecture & Component Specification

Engineering detail behind the [README](../README.md). This is the document to read before writing code in any layer.

> **Status.** §3 (data contracts), §4.1 (features) and §4.7 (governance) describe code that exists. Every other component section is a specification for code that does not yet exist — see [README §0](../README.md#0-status). Sections describing unbuilt components are marked ⚪.

---

## 1. Design principles

1. **The objective is the product.** Models are replaceable; the definition of a good price is not. It lives in one module (`decision/objective.py`) and everything else feeds it.
2. **Uncertainty is carried, never collapsed.** Estimators return distributions (quantiles, confidence intervals, survival probability curves). Collapsing to a point happens once, at the final decision, and the discarded spread is reported.
3. **Every number is reconstructible.** Any recommendation can be rebuilt months later from the decision log plus pinned model artefacts and exact random seeds.
4. **Guardrails are not model outputs.** They are hard constraints applied after scoring, so a model bug or wild inference cannot produce an illegal price.
5. **Fail visible, never silent.** Degrade to a simpler decision level and explicitly log the reason code; never emit a confident number from broken inputs.
6. **Validate on known ground truth first.** Every estimator is proven on synthetic data where the true causal parameter is known before it touches real data.
7. **The device is part of the contract.** All model training and inference runs on GPU, enforced by `prismprice.compute.require_gpu()`. Silent CPU fallback would break principle 3: kernels and reduction orders differ across devices, so a CPU-trained artefact cannot reproduce the number its decision record claims. See [README §8.1](../README.md#81-gpu-only-compute-policy).

---

## 2. Domain model

| Entity | Key | Notes |
| --- | --- | --- |
| `Product` | `sku` | Category, brand, lifecycle stage, cost basis, embedding vector |
| `PricePoint` | `(sku, valid_from)` | Effective price with provenance (recommended / manual / promo) |
| `DemandObservation` | `(sku, date)` | Units sold, revenue, web exposure, conversion, stockout flag |
| `InventorySnapshot` | `(sku, date)` | On hand, on order, days of cover, shadow price $\nu_i$ |
| `CompetitorObservation` | `(sku, competitor, observed_at)` | Price, availability, staleness timestamp |
| `Customer` | `customer_id` | Cohort, RFM, tenure, price sensitivity segment |
| `PurchaseEvent` | `(customer_id, order_id, sku)` | Basis for retention and CLV survival estimation |
| `Decision` | `decision_id` | Immutable record of one recommendation |
| `Experiment` | `experiment_id` | Assignment, propensity score, treatment price ladder |

**Invariant:** a `Decision` is never mutated. Corrections are new decisions that supersede prior ones, linked by `supersedes_id`.

---

## 3. Data contracts — 🟢 built

Every source declares a contract enforced at the boundary. A violation quarantines the batch and raises an alert; bad data never flows downstream.

```yaml
source: transactions
primary_key: [invoice_id, sku, line_number]
freshness_sla_hours: 24
columns:
  invoice_id:    {type: string, nullable: false}
  sku:           {type: string, nullable: false, foreign_key: products.sku}
  quantity:      {type: int,    nullable: false, min: -10000, max: 10000}
  unit_price:    {type: decimal, nullable: false, min: 0}
  invoice_ts:    {type: timestamp, nullable: false, max: now()}
  customer_id:   {type: string, nullable: true}
checks:
  - returns_are_negative_quantity
  - price_is_positive
```

**Checks are names, not expressions.** An earlier draft of this document wrote them as evaluable strings (`expr: "quantity < 0 implies ..."`). Resolving those at runtime means either `eval` on configuration — an injection surface on the one code path whose entire job is to distrust its input — or writing an expression parser. Instead, checks are Python callables in `CHECK_REGISTRY` (`data/contracts.py`), referenced by name; a name that is not registered fails at load time rather than being silently skipped.

**Why `nullable: true` on `customer_id` matters:** guest checkouts represent a significant portion of real transaction logs. Retention modelling must exclude them explicitly rather than silently treating them as one synthetic mega-customer.

**Twelve gate codes.** `DQ-001` missing column, `DQ-002` wrong dtype, `DQ-003` unexpected null, `DQ-004` out of range, `DQ-005` disallowed value, `DQ-006` duplicate primary key, `DQ-007` null primary key, `DQ-008` stale source, `DQ-009` future timestamp, `DQ-010` row check failed, `DQ-011` orphan foreign key, `DQ-012` empty batch. All checks run in one pass, so a report diagnoses the whole batch rather than the first fault found. `DQ-009` is an error rather than a curiosity: a row stamped after `as_of` is a leakage vector.

---

## 4. Component specifications

### 4.1 Feature builder (`features/`) — 🟢 built

**Contract:** given `(sku, as_of_date)`, return features computed using **only** data with `timestamp < as_of_date`.

| Feature group | Examples & Special Methods |
| --- | --- |
| Demand history | trailing units 7/28/91d, trend, volatility, zero-sales streak |
| Price history | current, reference (trailing 60d median), discount depth, EU Omnibus 30-day min price |
| Seasonality | week of year, holiday proximity, category seasonal index |
| Inventory & Cover | on hand, days of cover, markdown risk, opportunity shadow price $\nu_i$ |
| Competition | gap to competitor, rank, observation staleness, competitor reaction score |
| Customer mix | new vs. repeat share, cohort value mix, price shock exposure |
| **Demand Un-censoring** | **Right-censored Tobit** on log demand, correcting sales recorded during stockouts |
| **Cold-Start Embeddings** | Text/visual embeddings mapping new SKUs to nearest-neighbour elasticity priors |

**Leakage test (mandatory):** build features at `T`, then append data from `T+1..T+30` and rebuild. Values must be identical. Implemented three ways in `tests/test_features.py`: appending the future, removing the decision day, and corrupting future rows outright and demanding no reaction.

**Un-censoring is right-censored, not censored at zero.** The textbook Type I Tobit assumes a stockout means zero recorded sales. Real replenishment produces *partial* fulfilment — stock runs out having served some units — so what is known is `demand >= units_served`. Censoring at zero would be the wrong likelihood. The model is fitted on log demand, because demand is multiplicative and the normality assumption the likelihood rests on is false in levels. Measured on synthetic data with known latent demand, it removes ~78% of the censoring error.

**Cold start returns nothing rather than something.** `knn_prior` drops neighbours below a similarity floor instead of down-weighting them. A weighted average over the whole catalogue always returns a number, and that number is the catalogue mean wearing a similarity score — the caller must be able to tell "no analogue exists" from "here is a weak analogue", because only the first should trigger degradation.

---

### 4.2 Demand model (`estimation/demand.py`) — ⚪ spec only

- **Output:** quantiles `{p10, p50, p90}` of units at candidate price $\mathbf{p}$ and context.
- **Method:** LightGBM quantile regression with monotonic constraints (decreasing in price).
- **Validation:** rolling-origin backtest; empirical coverage of p10/p90 within $\pm 3\text{pp}$ of nominal.

---

### 4.3 Causal Elasticity model (`estimation/elasticity.py`) — ⚪ spec only

- **Output:** `{point, ci_low, ci_high, method, confidence}` per SKU.
- **Method:** Double Machine Learning (DML) / Orthogonal R-Learner (via `EconML`).
  - Stage 1: Partial out confounders (seasonality, marketing spend, promo depth) from treatment (price) and outcome (log demand).
  - Stage 2: Fit non-parametric treatment effect on residuals to isolate true causal price elasticity.
- **Confidence tag:** `high` when genuine experimental or uncorrelated price variation exists; `low` when purely observational or CI width exceeds threshold $\tau_{\max}$.

---

### 4.4 Retention & Deep CLV model (`estimation/retention.py`) — ⚪ spec only

- **Output:** $\Delta\text{CLV}_i(p_i)$ — change in discounted expected future margin over horizon $H$.
- **Method:** Deep Survival Model (Cox Proportional Hazards / PyTorch) with time-varying covariates.
  - Attributes churn risk strictly to the price-vs-reference shock: $(p / p_{\text{ref}} - 1)$.
- **Mathematical Form:**
  $$\Delta\text{CLV}(p) = \mathbb{E}[\text{buyers}(p)] \times \sum_{k=1}^{H} \left( \rho_k(p) - \rho_k(p_{\text{ref}}) \right) \cdot \text{margin}_k \cdot (1 + r)^{-k}$$

---

### 4.5 Competitor Game model (`estimation/competitor.py`) — ⚪ spec only

- **Output:** Predicted competitor price response function $\hat{p}_{\text{comp}}(p)$.
- **Method:** Game-theoretic best-response estimator preventing automated downward spirals (price wars) by simulating multi-agent equilibrium bounds.

---

### 4.6 Decision engine & Objective (`decision/`) — ⚪ spec only

The objective function optimizes expected category portfolio contribution plus customer relationship impact minus risk penalties:

$$\max_{\mathbf{p} \in \mathcal{P}} \sum_{i=1}^{K} \left( \mathbb{E}[q_i(\mathbf{p})] \cdot (p_i - c_i - \nu_i) + \lambda_i \mathbb{E}[\Delta\text{CLV}_i(p_i)] \right) - \gamma \cdot \text{CVaR}_{\alpha}(\mathbf{p})$$

- **Candidate ladder:** Snapped to psychological price endings (`.95`, `.99`) bounded by movement caps.
- **Monte-Carlo simulation:** $N=2,000$ demand draws per candidate price vector.
- **Category portfolio pass:** Re-scores price vectors jointly to penalize intra-category cannibalization.

---

### 4.7 Governance & Guardrails (`governance/guardrails.py`) — 🟢 built

Each guardrail is a pure predicate returning a structured `GuardrailResult`:

```python
def default_guardrails() -> list[BaseGuardrail]:
    return [
        AbsoluteFloorGuardrail(),       # PP-G001  p >= unit_cost
        MarginFloorGuardrail(),         # PP-G002  p >= cost * (1 + m_min)
        EUOmnibusAnchorGuardrail(),     # PP-G003  p_promo <= min(p_{t-30 -> t}), fails closed
        CompetitiveCeilingGuardrail(),  # PP-G004  p <= kappa * competitor_price
        MovementCapGuardrail(),         # PP-G005  |dp| / p_prev <= delta
        ChangeFrequencyGuardrail(),     # PP-G006  <= N changes per rolling window
        InventoryGuardGuardrail(),      # PP-G007  p >= cost + nu below the cover threshold
        LadderComplianceGuardrail(),    # PP-G008  allowed price endings only (.95, .99)
        FairnessGuardrail(),            # PP-G009  no identity attribute or proxy in context
    ]
```

**Result shape.** Verdicts are machine-readable, not prose:

```python
GuardrailResult(
    reason_code=GuardrailCode.MARGIN_FLOOR,
    constraint_name="Margin Floor",
    status=GuardrailStatus.FAILED,     # PASSED | FAILED | NOT_APPLICABLE
    observed=14.00,                    # the quantity under test
    limit=18.00,                       # the threshold it breached
    slack=-4.00,                       # signed: >= 0 iff passed, for floors and ceilings alike
    detail="Price 14.0000 is below floor 18.0000",
)
```

`slack` carries one sign convention across every predicate, which is what lets a dashboard rank "how close did we run to the edge?" without parsing strings, and lets a single property test catch a sign error in any guardrail.

**Three invariants, each enforced by a test:**

| Invariant | Why |
| --- | --- |
| Tolerance is representation-error only (`rel_tol=1e-9`) | An absolute slack admits marginally-illegal prices and scales inconsistently with price magnitude |
| Regulatory constraints fail closed | A promo with no 30-day anchor is a data failure; passing it is how a non-compliant price ships |
| `NOT_APPLICABLE` ≠ `PASSED` | Only "checked, fine" is evidence of compliance. A missing-data pass that reads as a compliance pass makes the audit log fiction |

**Ladder interface.** `GuardrailEngine.evaluate_ladder()` is the L3 → L4 boundary: it filters candidates to the feasible set, counts binding constraints for observability, and on an empty feasible set returns the previous price with `FALLBACK_RULE_ENGINE` rather than raising or relaxing a constraint.

**Fairness (PP-G009) is a real predicate.** It inspects the request contract: it fails if any field name matches a protected attribute or known proxy, and fails if the model stops forbidding unknown fields (`extra="forbid"`). `PriceRequest` therefore cannot carry `customer_id` — construction raises. The check is structural, so it cannot be satisfied by a system that personalises.

**Property-based testing.** Verified with Hypothesis: 1,000 generated cases per property locally, 2,000 in CI, across 14 properties plus example-based tests — 100 tests in total. The properties assert system-level guarantees ("no feasible price is below cost", "slack sign agrees with status", "every reason code has an implementation") rather than restating each predicate's own arithmetic, which would pass for any self-consistent implementation including a self-consistently wrong one.

---

### 4.8 Learning layer (`learning/`) — ⚪ spec only

- **Designed experiments:** Switchback and geo/store splits by category.
- **Safe Contextual Bandits:** Exploration with lower bounds ($1 - \alpha$ of baseline policy) and monetary risk budgets.
- **Off-policy evaluation (OPE):** Doubly-robust and inverse propensity scoring estimates.

---

## 5. Decision record schema

The audit artefact. One row per recommendation, immutable.

This is the exact serialisation of `DecisionRecord` in [`governance/schemas.py`](../src/prismprice/governance/schemas.py):

```json
{
  "decision_id": "uuid-v4",
  "sku": "SKU-001",
  "as_of": "2026-08-17T02:00:00Z",
  "created_at": "2026-08-17T02:00:03.117Z",
  "policy_version": "1.4.2",
  "model_versions": {
    "demand": "d-2026.08.1",
    "causal_elasticity": "dml-2026.08.1",
    "retention_survival": "deepclv-2026.07.4"
  },
  "inputs": {
    "current_price": 32.00,
    "unit_cost": 15.00,
    "is_promo": false,
    "competitor_price": 31.50,
    "inventory_cover_days": 14.0,
    "inventory_shadow_price": 0.0,
    "min_price_last_30d": 29.99
  },
  "estimates": {
    "demand_quantiles": {"p10": 9.6, "p50": 14.2, "p90": 20.1},
    "causal_elasticity": {
      "point": -1.38, "ci_low": -1.81, "ci_high": -0.94,
      "method": "dml-orthogonal-r-learner", "confidence": "high"
    },
    "delta_clv": -0.42
  },
  "candidates": [
    {
      "price": 30.95, "j_score": 241.1, "cvar": 180.2, "cvar_alpha": 0.05,
      "feasible": true, "binding_constraints": []
    }
  ],
  "recommended_price": 30.95,
  "binding_constraints": ["PP-G005"],
  "guardrail_results": [
    {
      "reason_code": "PP-G005", "constraint_name": "Movement Cap",
      "status": "FAILED", "observed": 0.0328, "limit": 0.03, "slack": -0.0028,
      "detail": "Relative price change 0.0328 exceeds ceiling 0.0300"
    }
  ],
  "degradation_reason_code": "OK_OPTIMAL",
  "degradation_rung": 1,
  "exploration": {"is_exploratory": false, "propensity": 0.82},
  "seed": 20260817,
  "approver": null,
  "supersedes_id": null
}
```

Four properties of this record are enforced by the model, not by convention:

- **`degradation_rung` is derived from `degradation_reason_code`.** Supplying it is permitted so records round-trip through JSON, but a value contradicting the code is rejected. The number and the code cannot drift apart.
- **`policy_version` and `seed` are required.** A record that defaults its own version is not reconstructible, which defeats the point of having it.
- **`is_promo` is captured.** Without it, whether `PP-G003` applied cannot be determined months later.
- **The record is frozen and forbids extras.** Corrections are new records linked by `supersedes_id`; nothing is ever mutated.

`cvar_alpha` is explicit rather than baked into a field name (`cvar5`), because the objective parameterises the tail probability and a record that assumes 5% cannot describe a run that used 1%.

---

## 6. Reliability & Degradation Matrix

| Level | Trigger / Condition | Action Taken | Governance Reason Code |
| --- | --- | --- | --- |
| **1** | Full system health | Optimal portfolio decision | `OK_OPTIMAL` |
| **2** | Competitor feed stale | Widen movement bound $\delta$ to limit exposure | `WARN_STALE_COMPETITOR` |
| **3** | Elasticity CI width $> \tau_{\max}$ | Fall back to category pooled mean elasticity | `FALLBACK_POOLED_ELASTICITY` |
| **4** | MC optimization non-convergent | Rule-based anchor (Cost-plus + competitor band) | `FALLBACK_RULE_ENGINE` |
| **5** | Model store 5xx / outage | Return last known good price or $p_{\text{prev}}$ | `CRITICAL_MAINTAIN_PREV` |

---

## 7. What "done" means per phase

| Phase | Status | Executable Gate |
| --- | --- | --- |
| 0 Foundation | 🟢 met | `ruff`, `mypy --strict` and `pytest` green in CI on Python 3.10–3.12 |
| 1 Data & Features | 🟢 met | Quality report generated with 12 gate codes; leakage test passes; Tobit un-censoring removes ~78% of censoring error against known latent demand |
| 6 Governance | 🟢 met | Property tests prove 0 guardrail violations; every `GuardrailCode` has an implementation; regulatory constraints fail closed |
| 2 Demand | ⚪ | Quantile LightGBM p10/p90 empirical coverage within $\pm 3\text{pp}$ |
| 3 Causal Elasticity | ⚪ | DML recovers synthetic ground-truth $\beta$ within CI on $\ge 90\%$ of SKUs |
| 4 Retention | ⚪ | Deep Survival cohort curve MAE below threshold on held-out test window |
| 5 Decision | ⚪ | Category portfolio backtest beats cost-plus and competitor-match baselines |
| 7 Serving | ⚪ | API contracts pass; chaos test (killed model store) degrades cleanly to Level 5 (`CRITICAL_MAINTAIN_PREV`) |
| 8 Learning | ⚪ | OPE recovers known policy value on synthetic logs within tolerance |
