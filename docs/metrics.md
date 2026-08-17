# PrismPrice — Metrics & Dashboard Specification

Expands README §6. Every metric here has a definition precise enough to implement, an owner, a threshold, and a stated action when the threshold is breached. A metric with no action attached is decoration and does not belong on a dashboard.

**Status:** specification. The KPI definitions are fixed; no dashboard is built yet (see README §0).

---

## 1. The headline metric

$$\text{CPPC}_{90} = \frac{\sum_{t \in [T-90,\,T]} \sum_{i} q_{it}(p_{it} - c_{it})}{\left|\{\text{customers transacting in } [T-90, T]\}\right|}$$

**Contribution profit per customer over a rolling 90 days.**

It is the headline because it is the only single number that cannot be gamed by winning one half of the trade-off. Discount-led volume lifts the denominator and depresses the numerator. Margin extraction lifts per-transaction profit and shrinks the customer count as defectors leave. Both failures show up here; neither shows up in gross margin percentage.

Ninety days because the repurchase cycle for the target categories sits inside it. Shorter and price-shock churn has not yet expressed itself; longer and the signal lags the decision that caused it.

---

## 2. Executive dashboard

| Metric | Definition | Threshold | On breach |
| --- | --- | --- | --- |
| Contribution profit vs. plan | Actual − planned contribution, cumulative MTD | < −5% | Category review; consider raising $\lambda$ |
| Gross margin % | $(\sum q(p-c)) / \sum qp$ | < floor − 1pp | Check guardrail bind rate before blaming the model |
| Revenue vs. plan | Cumulative MTD | < −8% | Joint review with commercial |
| **CPPC 90d** | §1 | Declining 3 periods running | Escalate; this is the metric that matters |
| Price adherence | Share of recommendations published unmodified | < 70% | The system is not trusted — find out why before tuning it |
| Retention cohort curve | Repurchase rate by weeks-since-first-order, by cohort | Cohort below prior-year band | Investigate price-shock exposure of that cohort |
| Price perception index | Basket of KVIs vs. competitor mean | > 1.05 | Competitiveness risk on the items customers actually price-check |

**Price adherence is the honest health metric for a decision-support system.** A high recommendation rate with low adherence means the system is producing numbers nobody uses. That is a worse outcome than a system that is down, and it is invisible in every profit metric.

---

## 3. Commercial / category manager dashboard

| Metric | Definition | Threshold |
| --- | --- | --- |
| Per-SKU recommendation + reason | Price, delta, binding constraints, degradation rung | — |
| Margin bridge | Decomposition of margin change into volume / price / mix | — |
| Guardrail bind rate | Share of candidates rejected, by reason code | Any single code > 60% |
| Competitor position | Gap to competitor, rank, observation staleness | Staleness > 24h |
| Inventory cover | Days of cover, markdown risk, shadow price $\nu$ | Cover < 14d |
| Recommendation churn | Share of SKUs whose recommendation reversed direction within the window | > 10% |

**Guardrail bind rate is a policy diagnostic, not an error rate.** A code binding on most candidates means the constraint, not the model, is setting the price. That may be correct — a margin floor *should* bind in a low-margin category — but it must be a decision someone made, not a fact nobody noticed. `PP-G008` (ladder compliance) binding at 95%+ is expected and is excluded from the alert.

**Recommendation churn** catches oscillation: a system that moves a price up on Monday and down on Wednesday is destroying trust even if each individual decision scored well.

---

## 4. Data science dashboard

| Metric | Definition | Threshold |
| --- | --- | --- |
| Demand WAPE | $\sum\|q - \hat{q}\| / \sum q$, by horizon | > prior model + 2pp |
| Quantile calibration | Empirical coverage of p10/p90 | Outside nominal ± 3pp |
| Elasticity CI width | $\text{ci}_{\text{high}} - \text{ci}_{\text{low}}$, distribution across SKUs | Median > $\tau_{\max}$ |
| Elasticity sign violations | Share of SKUs with $\hat{\beta} \ge 0$ | > 2% |
| Feature drift | PSI vs. training window, per feature | PSI > 0.25 |
| Prediction drift | PSI on predicted demand distribution | PSI > 0.25 |
| OPE uplift | Doubly-robust estimate vs. logged policy, with CI | CI includes 0 → do not ship |

**Quantile calibration is the one to watch.** The objective samples the demand distribution, so if p10/p90 are dishonest the CVaR term is meaningless and the risk penalty is decorative — the system will look risk-aware while taking unmeasured risk. Coverage is checked on a rolling-origin backtest, never in-sample.

A positive estimated elasticity is almost always confounding rather than a Giffen good. Sign violations are a DML health check, not a finding.

---

## 5. Engineering dashboard

| Metric | Definition | Threshold |
| --- | --- | --- |
| Decision latency | p50 / p95 / p99, single-SKU endpoint | p99 > 500ms |
| Error rate | 5xx / total | > 0.1% |
| Feature freshness | Age of newest row per source vs. SLA | Beyond source SLA |
| **Degradation rung distribution** | Share of decisions at rungs 1–5 | Rung ≥ 3 > 5% of decisions |
| Fallback frequency | Empty-feasible-set fallbacks per run | > 1% of SKUs |
| Batch throughput | SKUs scored per minute, full-catalogue refresh | Refresh exceeds window |
| GPU utilisation | Device utilisation and memory during training | Job on CPU ⇒ page immediately |

**Rung distribution is the single most informative operational chart.** A system that always answers hides its own degradation inside a healthy-looking success rate; the rung histogram is what makes "we returned a price" and "we returned a *good* price" different numbers.

A training job running on CPU is a page, not a warning — see README §8.1.

---

## 6. Alerting

Circuit breakers, distinct from dashboards. These halt publication rather than notifying someone.

| Trigger | Action |
| --- | --- |
| > 30% of SKUs move in the same direction in one run | Halt publication, page a human |
| Aggregate basket-level price movement > 3% | Halt publication |
| Any decision violating a guardrail post-hoc | Halt, quarantine, incident (this should be impossible; the check exists because "impossible" is a claim, not a guarantee) |
| Feature source beyond freshness SLA | Degrade to rung 2+, alert |
| Model store unreachable | Degrade to rung 5, alert |

---

## 7. What is deliberately not measured

- **Per-customer profitability as a pricing input.** It would be the natural optimisation target and it is exactly what `PP-G009` forbids. Measured for reporting; never fed back into a price.
- **Competitor undercut rate as a goal.** Optimising it invites the price war the competitor game model exists to avoid.
- **Model accuracy as a headline.** A better-calibrated demand model that does not move contribution profit has not earned attention. Accuracy is a diagnostic on the DS dashboard, not a business metric.
