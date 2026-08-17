# PrismPrice — Data & Modelling Specification

This document details the data generation, dataset augmentation, demand un-censoring, and machine learning methods used across PrismPrice.

> **Status.** Specification only — none of the components below are implemented yet ([README §0](../README.md#0-status)). The declared assumptions here are binding on the implementations when they arrive: the cost model in §2.1 in particular is an *assumption*, not a measurement, and any margin figure derived from it inherits that.
>
> **Compute.** Every model described here trains on GPU. `prismprice.compute.require_gpu()` raises rather than falling back to CPU — see [README §8.1](../README.md#81-gpu-only-compute-policy).

---

## 1. Synthetic Panel Data Generator

To validate causal estimators (DML, Deep Survival) against **known ground truth**, PrismPrice includes a synthetic data generator (`src/prismprice/data/synthetic.py`).

### 1.1 Structural Demand Equation

Demand for SKU $i$ at time $t$ given price vector $\mathbf{p}_t$ is generated as:

$$\log q_{it} = \alpha_i + \beta_i \log p_{it} + \sum_{j \neq i} \eta_{ij} \log p_{jt} + \boldsymbol{\gamma}_i' \mathbf{X}_{t} + \epsilon_{it}$$

Where:
- $\alpha_i \sim \mathcal{N}(3.5, 0.5)$: Baseline log-demand for SKU $i$.
- $\beta_i \sim \mathcal{N}(-1.8, 0.4)$: True causal price elasticity ($\beta_i < 0$).
- $\eta_{ij} > 0$: Cross-price elasticity between substitute SKUs $i$ and $j$.
- $\mathbf{X}_t$: Confounders (seasonality, marketing campaigns, day-of-week).
- $\epsilon_{it} \sim \mathcal{N}(0, \sigma^2)$: Unobserved demand shock.

### 1.2 Customer Repurchase Hazard & Survival Model

Customer repurchase event hazard $h(t | x)$ at tenure $t$ given paid price ratio $r = p / p_{\text{ref}}$ is modeled as:

$$h(t | r) = h_0(t) \cdot \exp\left( \theta \cdot (r - 1) + \boldsymbol{\beta}_{\text{cust}}' \mathbf{Z} \right)$$

Where:
- $h_0(t)$: Baseline Weibull hazard function $h_0(t) = p k t^{k-1}$.
- $\theta \approx -1.2$: Sensitivity parameter. Paying above reference price ($r > 1$) increases hazard of churn (reduces repurchase probability).

---

## 2. UCI Online Retail Dataset Augmentation

The public UCI Online Retail dataset provides real transaction logs but lacks cost (COGS), inventory levels, and competitor pricing. PrismPrice augments the dataset deterministically using declared assumptions:

### 2.1 COGS Cost Model Assumptions

For SKU $i$, unit cost $c_i$ is constructed via category gross margin target $m_i$:

$$c_i = \bar{p}_{i, \text{hist}} \times (1 - m_{\text{cat}(i)})$$

Where $m_{\text{cat}(i)} \sim U(0.40, 0.55)$ is fixed per product category.

### 2.2 Inventory Cover & Shadow Price ($\nu_i$)

Inventory $I_{it}$ is generated using periodic replenishment cycles. When stock falls below $I_{\text{min}}$, Inventory Opportunity Cost (Shadow Price $\nu_{it}$) scales exponentially:

$$\nu_{it} = \begin{cases} 
0 & \text{if } I_{it} \ge I_{\text{threshold}} \\
c_i \times \left( \exp\left( \kappa \cdot \frac{I_{\text{threshold}} - I_{it}}{I_{\text{threshold}}} \right) - 1 \right) & \text{if } I_{it} < I_{\text{threshold}}
\end{cases}$$

### 2.3 Competitor Price Series & EU Omnibus Compliance

- **Competitor Observation**: $p_{\text{comp}, it} = p_{it} \times \mathcal{N}(1.02, 0.04)$ with random lag intervals simulating web scraping staleness.
- **EU Omnibus Reference Price**: For promo validation ($PP-G003$), $p_{\text{ref}, 30d} = \min_{s \in [t-30, t]} p_{is}$. Promoted price must not exceed $p_{\text{ref}, 30d}$.

---

## 3. Demand Un-Censoring (Tobit / Kaplan-Meier)

During stockout periods ($I_{it} = 0$), observed sales equal zero, but true unconstrained demand $q^*_{it} > 0$. Naive models underestimate demand.

### 3.1 Tobit Model Formulation

Unconstrained demand $q^*_{it}$ is treated as a latent variable observed only when inventory is positive:

$$q_{it} = \begin{cases} 
q^*_{it} & \text{if } I_{it} > 0 \text{ (Uncensored)} \\
\text{Censored at } 0 & \text{if } I_{it} = 0 \text{ (Censored)}
\end{cases}$$

The log-likelihood for Tobit estimation adjusts for right-truncated/censored sales points:

$$\ln L = \sum_{q_{it} > 0} \left[ -\ln \sigma + \phi\left( \frac{q_{it} - \mathbf{x}'_{it}\boldsymbol{\beta}}{\sigma} \right) \right] + \sum_{q_{it} = 0} \ln \Phi\left( \frac{0 - \mathbf{x}'_{it}\boldsymbol{\beta}}{\sigma} \right)$$

---

## 4. Cold-Start Embeddings (CLIP / BERT)

New SKUs lack historical demand logs. PrismPrice projects new SKUs into embedding space using text and visual descriptions:

$$\mathbf{e}_i = \text{Normalize}\left( W_t \cdot \text{BERT}(\text{title}_i, \text{category}_i) + W_v \cdot \text{CLIP}(\text{image}_i) \right)$$

The historical prior elasticity $\beta_{\text{new}}$ is derived from $K$-nearest neighbor historical SKUs:

$$\beta_{\text{new}} = \sum_{k \in \text{KNN}(i)} w_k \cdot \beta_k, \quad w_k = \frac{\cos(\mathbf{e}_i, \mathbf{e}_k)}{\sum_j \cos(\mathbf{e}_i, \mathbf{e}_j)}$$
