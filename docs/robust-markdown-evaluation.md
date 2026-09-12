# Sequential markdown under elasticity ambiguity

**Status:** implemented in [`decision/robust_markdown.py`](../src/prismprice/decision/robust_markdown.py), tested in [`tests/test_robust_markdown.py`](../tests/test_robust_markdown.py) (32 tests), benchmarked by [`scripts/evaluate_robust_markdown.py`](../scripts/evaluate_robust_markdown.py). Every number below is reproduced by that script into [`data/runs/robust_markdown_benchmark.json`](../data/runs/robust_markdown_benchmark.json); none is quoted from a paper or estimated by hand.

---

## 1. The gap

[`estimation/elasticity.py`](../src/prismprice/estimation/elasticity.py) produces a cross-fitted DML estimate that is **an interval, not a number**: `point`, `ci_low`, `ci_high`, and a `confidence` tag. [`decision/markdown.py`](../src/prismprice/decision/markdown.py) then solves the markdown MDP exactly by backward induction — and takes `elasticity` as a scalar. The interval is discarded at the module boundary, and a whole season is planned as though the point estimate were exact.

On the real UCI panel that discarded interval has a median width of **0.88** among high-confidence SKUs. That is not a rounding error; it spans elasticities that imply materially different markdown ladders.

Three literatures touch this and none closes it:

- **Robust and distributionally-robust MDPs** (Xu & Mannor; Wiesemann et al.; and 2025 work such as linear-mixture DRMDPs, [arXiv:2505.18044](https://arxiv.org/abs/2505.18044), and Bayesian Risk MDPs, [arXiv:2106.02558](https://arxiv.org/pdf/2106.02558)) take the ambiguity set as **given**. None derives it from a specific estimator's finite-sample confidence region.
- **Conformal prediction in sequential decisions** (conformal off-policy prediction, PMLR v206; *Calibrating Decision Robustness via Inverse Conformal Risk Control*, [arXiv:2510.07750](https://arxiv.org/pdf/2510.07750)) addresses policy *evaluation* and single-shot predict-then-optimize. Fetching the last of these confirms it sweeps robustness levels to trace a miscoverage–regret frontier — it does not propagate a set through a multi-period Bellman recursion.
- **Markdown pricing under demand uncertainty** (*Markdown Pricing Under an Unknown Parametric Demand Model*, [arXiv:2312.15286](https://arxiv.org/abs/2312.15286); *Offline Dynamic Inventory and Pricing*, [arXiv:2504.09831](https://arxiv.org/abs/2504.09831)) handles uncertainty through parametric families, censoring-aware Bellman equations, or offline RL. The second, on inspection, expands the state to track consecutive censoring events — it is not estimator-interval ambiguity.

**The gap:** nobody propagates a causal estimator's own confidence interval through an *exact* markdown backward induction, and nobody attaches a per-decision robustness statement to the result. That last part is newly load-bearing: NY's Algorithmic Pricing Disclosure Act (Nov 2025) and the EU AI Act's high-risk obligations (Aug 2026) both push toward pricing decisions that can be audited for how they handle what the model does not know.

---

## 2. Two operators

Demand is Poisson with mean `base_demand · (p / base_price)^e`, where `e` is known only to lie in `[ci_low, ci_high]`. Write `Q_e(t,i,p)` for the action value under a particular `e`.

### 2.1 Value-robust (`solve_robust_markdown`)

$$V(t,i) = \max_p\Big[(1-\rho)\,Q_{\hat e}(t,i,p) + \rho\,\mathrm{CVaR}_\alpha\big(Q_e(t,i,p)\big)\Big]$$

The inner `Q` is computed against `V` itself, so ambiguity compounds across periods instead of being applied once at the end. Two properties make this a strict generalisation rather than a different model:

- **ρ = 0 reproduces `solve_markdown` exactly** — verified to 2·10⁻¹¹, policy array identical, asserted as a test. Any measured difference between robust and classical policies is therefore the operator, not an implementation discrepancy.
- **ρ = 1, α → 0 is classical minimax** over the interval.

A zero-width interval makes the dial provably inert — also a test.

### 2.2 Regret-robust (`solve_regret_robust_markdown`)

The value operator was built first, benchmarked, and **found wanting** (§4). It protects the season's *value* across the set, which on a markdown problem drags the policy toward the inelastic end — where profit is low for reasons no policy can repair. Defending that sacrifices real profit in the elastic states where a decision genuinely was available.

The second operator minimises the CVaR of **regret** instead:

$$p^*(t,i) = \arg\min_p \mathrm{CVaR}_\alpha\big(V^*_g(t,i) - Q_g(t,i,p)\big)$$

carrying two families of value function per grid elasticity `g`: `V*[g]`, the optimum under `g`, and `W[g]`, the value of *this* policy under `g`, which supplies the continuation so the policy is scored against its own future actions. Both advance in one backward sweep.

Both internal families are validated against independently computed ground truth to **10⁻¹¹** — `V*[g]` against `solve_markdown` at each grid point, `W[g]` against evaluating the finished policy from scratch.

### 2.3 Why not just solve at the pessimistic endpoint

Because there isn't one. A more elastic customer punishes a price *rise* and rewards a *cut*, so the damaging end of the interval flips sign along the ladder and flips again as inventory pressure moves the optimiser along it. An endpoint policy is pessimistic in some states and *optimistic* in others. It is carried as a baseline throughout, and it loses everywhere.

---

## 3. Method notes

**Exact policy evaluation, not simulation.** `exact_policy_value` runs the same backward induction with the maximisation removed, giving a policy's true expected value under any assumed truth with no Monte-Carlo error. The policies here differ by a few percent and several hundred simulated seasons carry a standard error of the same order — a benchmark whose noise matches its effect decides nothing. Validated three ways: it reproduces the DP optimum exactly when handed the optimal policy, agrees with simulation inside its CI, and never scores any policy above the optimum.

**The grid measure is an assumption.** CVaR needs a measure and a confidence interval is a *set*. Grid points are weighted uniformly — the flat-prior reading. A sampling distribution would concentrate near the point estimate and make the same α less conservative.

**Tail size rounds up.** At α = 0.05 over 21 grid points the tail is 1.05 atoms; rounding down leaves one, at which the "CVaR" is a worst-case and α stops doing anything below 1/21. `DEFAULT_ROBUST_GRID_SIZE` carries this as its stated constraint.

**§1 and §2 measure different things, and both are reported.** §1 scores regret at the single *true* elasticity — which rewards a good point estimate. §2 scores worst-case regret *across the interval* — which is what a robust operator is built to control. Reporting only the first understates the method; only the second dodges what it costs when the estimate was fine. §1 now reports both.

---

## 4. Results

Full run: 128 synthetic scenarios, 226 real high-confidence SKUs, 2298 s. Four policies plus two ablations, all scored by exact policy evaluation.

### 4.1 Synthetic, known ground truth (n = 128)

DML was fitted to generated panels at four lengths (120–540 days) across four seeds and eight SKUs, then every policy was scored against the elasticity the generator actually used. The fitted intervals covered the truth **73.4%** of the time — below the nominal 95%, which is itself worth knowing about the estimator and is why the intervals are wide enough to matter.

**Regret at the single true elasticity** — the metric that rewards a good point estimate:

| Policy | Mean | Median | Worst |
| --- | ---: | ---: | ---: |
| Certainty-equivalent DP (baseline) | 0.69% | 0.21% | 6.51% |
| Pessimistic endpoint DP | 2.15% | 1.09% | 13.16% |
| Value-robust, ρ = 1.0 | 0.73% | 0.13% | 14.65% |
| **Regret-robust, CVaR** | **0.65%** | 0.26% | **5.98%** |

Essentially a tie on the mean. That is the honest reading and it is not a disappointment: when the point estimate is usually decent, a policy built to survive the interval should cost almost nothing at the centre of it. The value operator's **14.65% worst case** against a 6.51% baseline is the first sign of the problem in §4.3.

**Worst-case regret across the interval** — the metric a robust operator is actually built to control:

| Policy | Overall | Narrow (w̄ = 0.41) | Medium (w̄ = 0.57) | Wide (w̄ = 0.94) |
| --- | ---: | ---: | ---: | ---: |
| Certainty-equivalent DP | 2.12% | 0.83% | 1.50% | 4.01% |
| Pessimistic endpoint DP | 4.95% | 2.39% | 3.78% | 8.64% |
| Value-robust, ρ = 1.0 | 4.92% | 2.31% | 3.61% | 8.82% |
| **Regret-robust, CVaR** | **1.65%** | **0.72%** | **1.19%** | **3.03%** |
| — win rate vs. baseline | 71.9% | 62.8% | 73.8% | **79.1%** |
| — mean improvement | 0.47pp | 0.11pp | 0.31pp | **0.98pp** |

**This is the headline.** The improvement grows monotonically with interval width — 0.11pp → 0.31pp → 0.98pp — which is the behaviour the method is supposed to have and the one that is hardest to get by accident. The operator is nearly inert where the estimate is sharp and does real work where it is vague, because the ambiguity set it prices against is the estimator's own.

### 4.2 Real UCI Online Retail II panel

No true elasticity exists here, so regret is measured across each SKU's own DML interval — the set the data cannot distinguish between.

**The panel is mostly degenerate, and this is reported rather than averaged in.** Of 226 high-confidence SKUs, **109 have a wholly inelastic interval** (median point elasticity −0.37), where holding full price is optimal at every elasticity in the set and all policies emit an identical ladder. A further 97 are elastic but their optimal ladder does not vary across the interval. **20 are decision-relevant** (median point −1.37, median interval width 0.88). Pooling the other 206 would divide the effect by ten and report it as "no difference".

On those 20:

| Policy | Mean regret over interval | Worst over interval | Max worst |
| --- | ---: | ---: | ---: |
| Certainty-equivalent DP (baseline) | 0.48% | 2.06% | 5.84% |
| Pessimistic endpoint DP | 3.97% | 9.29% | 16.05% |
| Value-robust, ρ = 1.0 | 0.78% | 2.99% | 15.79% |
| **Regret-robust, CVaR** | 0.67% | **1.63%** | **4.34%** |

The regret operator wins worst-case on **80% of SKUs**, improving it by 0.43pp on average and cutting the worst SKU's exposure from 5.84% to 4.34%. It pays 0.19pp of mean regret for that. **That is the robustness trade-off working correctly** — and it is the same direction and rough magnitude as the synthetic result, obtained on real transactions.

### 4.3 The negative result: value-robustness protects the wrong quantity

The value operator was built first, and it is what the robust-MDP literature prescribes. On both datasets it is **beaten by the certainty-equivalent baseline** it was supposed to improve on — 4.92% vs 2.12% interval-worst regret on synthetic, 2.99% vs 2.06% on real.

The cause is structural, not a tuning failure. A max-min-*value* policy is pulled toward whichever elasticity in the set makes the season poorest, and on a markdown problem that is the inelastic end: there is simply less volume available there at any price. That shortfall is not the policy's to prevent and no policy can repair it. Steering the whole season toward defending it forfeits real profit in the elastic states, where a decision genuinely was available.

The floor-calibration table shows the degeneracy directly. As the interval widens past the elastic/inelastic boundary, the value-robust policy collapses onto holding full price and its robustness cost goes to exactly zero:

| Interval half-width | Certified floor | CE profit | Robustness cost | Certificate error | Season-level violations |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.3 | 5926.1 | 6356.6 | 430.5 | −0.0015% | 4.9% |
| 0.9 | 5480.0 | 5480.0 | 0.0 | −0.0000% | 49.1% |
| 1.5 | 5480.0 | 5480.0 | 0.0 | −0.0000% | 49.1% |

Subtracting the unreachable part — what the best possible policy would have earned under the same elasticity — is what the regret operator does, and it is the whole difference between the two.

### 4.4 The certificate is tight, not a bound — a retracted claim

`certified_profit_floor` was first documented as a conservative lower bound, reasoning that a nested risk measure (tail applied afresh each period) is dominated by the static one taken over whole trajectories. **That reasoning does not transfer here.** The elasticity that is worst at a given state need not be the one that is worst over a whole season, so the per-state tail and the end-to-end tail select different members of the set and neither dominates.

Measured across 40 instances spanning point estimates −1.6 to −3.4 and half-widths 0.2 to 1.2: the floor is **optimistic in 88% of cases**, tight in the rest, worst overpromise **0.042%**. It is now documented as empirically tight to within 0.05%, which is negligible against the 5–10% effects it is used to decide but is not the guarantee originally claimed.

Separately, the floor covers **parameter error only, not demand noise**. The season-level violation column above makes that concrete: on a tightly-estimated SKU the floor sits near the mean, so about half of individual seasons finish below it. That is expected and is reported rather than left to be discovered.

### 4.5 Multi-SKU, where exactness is lost

The joint state is not enumerable, so there is no recursion to take a tail inside. Domain randomisation over the interval is what remains: each REINFORCE training episode draws its elasticity from the CI. Three SKUs under a shared markdown budget, evaluated at three truths:

| True elasticity | Fixed training | Robust training | Independent per-SKU DP |
| ---: | ---: | ---: | ---: |
| −3.0 | 2401.7 | **3125.0** | 1829.0 |
| −2.0 (the point estimate) | 2459.5 | **2904.3** | 1507.9 |
| −1.2 | 2748.3 | 2748.3 | **2842.4** |

Robust training wins at the elastic truths by 18–30%, and does it while spending *less* of the markdown budget (0.71 vs 1.00 utilisation) — it learned to hold the allowance for the SKUs that need it.

**But it also wins at the nominal point estimate, where fixed training should have had the advantage, and that is a warning rather than a bonus.** Part of the gain is almost certainly regularisation rather than robustness: REINFORCE with an entropy bonus can settle early, and varying the environment across episodes supplies exploration pressure that a fixed environment does not. The honest claim is that domain randomisation helps this policy; the claim that it helps *because it confers robustness* is not separable at this training budget. Note also that the independent per-SKU DP still beats both learned policies at the inelastic truth.

### 4.6 Assumption sensitivity

**Gross margin** is the benchmark's most consequential assumption. At 35% and 45% margin the ladder floor sits at or below unit cost and the season has no discountable range at all — those rows are refused and recorded, not silently dropped. Where markdown is rational:

| Gross margin | Certainty-equivalent | Pessimistic endpoint | Regret-robust, CVaR |
| ---: | ---: | ---: | ---: |
| 55% | 3.75% | 18.30% | 6.10% |
| 65% (shipped) | 10.51% | 10.95% | **5.72%** |
| 75% | 2.46% | 1.72% | **1.57%** |

The regret operator is not uniformly best — at 55% margin the certainty-equivalent DP wins on this single instance. The single-instance table is noisier than the 128-scenario sweep in §4.1 and should be read as a scope check, not a result.

**Scale.** Demand and inventory are rescaled together to cap the state space. Across an 4× range of season size (100–400 opening units) regret moves by **1.03pp**, drifting mildly with size as the Poisson coefficient of variation changes — small against the effects measured, and reported rather than assumed away. An earlier version of this check swept sizes that all hit the same clamp and reported a spread of exactly zero; it was measuring the clamp.

---

## 5. Scope and limitations

- **The markdown MDP only has an interior solution at high margin.** Discounting multiplies units by `(p/p_full)^e` and unit margin by `(p-c)/(p_full-c)`. At a 45% gross margin — what `real_run.json` assumes for UCI — a 34% price cut at elasticity −2.2 raises volume 2.5× while cutting unit margin to 0.15×, destroying 62% of contribution. The DP correctly refuses to mark down, every policy holds full price, and every comparison ties at exactly zero. The benchmark therefore assumes a 65% gross margin, the fashion/seasonal range where markdown is actually practised, and measures the sensitivity to that choice rather than hiding it.
- **On the real UCI panel most SKUs are inelastic**, so the ambiguity is not decision-relevant for them. This is reported as a split, not averaged away — pooling degenerate SKUs with live ones would divide the real effect by an order of magnitude and report it as "no difference".
- **The certificate is empirically tight, not a proven bound.** See §4.
- **The multi-SKU extension is weaker by construction.** The joint state cannot be enumerated, so there is no recursion to take a tail inside; domain randomisation over the interval optimises the *mean*, not the tail, and carries no certificate.
- **Season length, opening buy, salvage and gross margin are ours, not the data's.** No public retail panel carries them. They are declared as module constants with the reasoning attached, in the same spirit as `AssumptionSet`.
