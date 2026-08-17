# PrismPrice — Operational Playbooks

What to do when something is wrong. Written for the person on call at 02:00 during the nightly catalogue refresh, who did not build this.

**Status:** specification. These procedures describe the system as designed; only the governance layer exists today (README §0). Steps referencing unbuilt components are marked ⚪.

**The standing rule:** a bad price is worse than no price. When in doubt, hold the previous price and escalate. Nobody has ever been fired for not changing a price for one day.

---

## 1. Triage: which playbook do I need?

| Symptom | Playbook |
| --- | --- |
| Recommendations look wrong but the system is up | §2 Suspicious recommendation |
| Alert: mass movement / circuit breaker fired | §3 Circuit breaker |
| Alert: degradation rung ≥ 3 for many SKUs | §4 Degraded operation |
| A guardrail is rejecting everything | §5 Empty feasible set |
| A price was published that should not have been | §6 Bad price published |
| Training job failed or is running on CPU | §7 GPU / training failure |
| Elasticity estimates changed sharply after a retrain | §8 Model regression |

---

## 2. Suspicious recommendation

A category manager reports a price that looks wrong.

1. **Pull the decision record** by `decision_id` or `(sku, as_of)`. It contains every input, estimate, model version, candidate score and binding constraint. You should not need to re-run anything to explain the number.
2. **Check the degradation rung first.** Rung ≥ 3 means the recommendation came from a fallback, not the optimiser. That usually *is* the explanation.
3. **Check `binding_constraints`.** If a guardrail bound, the constraint set the price, not the model. Confirm the constraint is configured correctly for the category before questioning the estimate.
4. **Check `min_slack`.** A near-zero slack means the price sat against a constraint edge; small input changes will move it visibly. That is expected behaviour, not instability.
5. **Check the inputs, not the model.** In order of likelihood: wrong `unit_cost`, stale `competitor_price`, `inventory_cover_days` from the wrong snapshot. Cost errors are the single most common cause of a wrong-looking price.
6. ⚪ **Only then** examine the elasticity CI and demand quantiles.

**Do not** patch the recommendation by editing the decision record. Issue a new decision with `supersedes_id` set to the original. Records are immutable; that is what makes the audit trail worth having.

---

## 3. Circuit breaker fired

Publication has already halted. Do not re-enable it to "see if it happens again."

1. **Confirm which breaker.** Mass directional movement, aggregate basket movement, or a post-hoc guardrail violation (see [metrics.md §6](metrics.md#6-alerting)).
2. **Mass movement in one direction is almost always an input, not a model.** A cost file that loaded in the wrong currency, or a competitor feed that came back after an outage with every price at zero, will move the whole catalogue coherently. Check the freshness and value ranges of every input source before looking at the model.
3. **A post-hoc guardrail violation is a P1.** It means a price passed the feasibility filter and was still illegal, which the property tests assert cannot happen. Quarantine the batch, capture the request that produced it as a regression test, and do not resume publication until that test is red-then-green.
4. **Resume deliberately.** Re-run in shadow mode, compare the distribution against the last known-good run, and resume only if the distribution is explicable.

---

## 4. Degraded operation (rung ≥ 3)

The system is returning prices; they are just not optimal ones. This is working as designed — but it is not a state to sit in for days.

| Rung | Meaning | First check |
| --- | --- | --- |
| 2 `WARN_STALE_COMPETITOR` | Competitor feed older than 24h | Scraper health; is the source blocking us? |
| 3 `FALLBACK_POOLED_ELASTICITY` | Per-SKU elasticity CI too wide | Is this a new SKU (expected) or did a retrain widen CIs everywhere (not expected)? |
| 4 `FALLBACK_RULE_ENGINE` | Optimiser did not converge, or no candidate was feasible | See §5 |
| 5 `CRITICAL_MAINTAIN_PREV` | Model store unreachable | Infrastructure; the pricing system is fine |

**Rung 3 across many SKUs after a retrain is a model regression, not a data problem** — go to §8.

Holding at rung 5 is safe indefinitely: it republishes the previous price. Holding at rung 3 quietly erodes value, because pooled elasticity ignores everything specific about the SKU. Set a clock on rung 3.

---

## 5. Empty feasible set

Every candidate on the ladder was rejected. `evaluate_ladder()` returns the current price with `FALLBACK_RULE_ENGINE`; nothing is broken, but nothing is optimising either.

1. **Read `binding_constraint_counts`** on the `LadderEvaluation`. It tells you which constraint killed the ladder and how many candidates it took out.
2. **Diagnose by code:**

   | Code | Typical cause |
   | --- | --- |
   | `PP-G001` / `PP-G002` | Cost rose above the achievable price. A commercial problem, correctly surfaced — this SKU cannot be sold profitably at market price. |
   | `PP-G003` | Promo requested with no 30-day anchor, or an anchor below every candidate. Check whether this item should be on promotion at all. |
   | `PP-G004` ∧ `PP-G002` | Margin floor above the competitive ceiling. The category policy is internally inconsistent; no price can satisfy both. Escalate to the policy owner — do not relax either constraint locally. |
   | `PP-G005` | Ladder generated outside the movement cap. A ladder-generation bug ⚪, not a governance issue. |
   | `PP-G006` | Change budget exhausted for the window. Correct behaviour; wait. |
   | `PP-G008` | Ladder not snapped to allowed endings. A ladder-generation bug ⚪. |

3. **Never widen a guardrail to clear a fallback.** The fallback is the guardrails working. If a constraint is genuinely wrong, change it as a versioned policy change with an owner, in shadow first — not as an incident action at 02:00.

---

## 6. Bad price published

The one failure with no clean rollback: it has already transacted.

1. **Stop the bleeding.** Halt publication for the affected SKUs and republish the last known-good price.
2. **Quantify exposure.** Units sold at the bad price × the delta. You will be asked; have the number before you are.
3. **Preserve evidence.** Snapshot the decision records *before* any repair run. They are immutable, but the feature store behind them is not; capture the inputs too.
4. **Honour the price where it was advertised.** This is a commercial and legal call, not an engineering one, but it is the default expectation under most consumer-protection regimes — including the one `PP-G003` exists to satisfy.
5. **Write the regression test before the fix.** Any incident that produced an illegal price becomes a permanent test case in `tests/test_guardrails.py`.
6. **Blameless postmortem.** The question is which guarantee was missing, not who ran the job.

---

## 7. GPU / training failure

PrismPrice refuses to train on CPU (README §8.1). A hard failure here is the system working.

1. **Read the error.** `GPUUnavailableError` names the component and the specific cause.
2. **Run the diagnostic:**

   ```python
   from prismprice.compute import gpu_report
   print(gpu_report().as_dict())
   ```

3. **Fix by cause:**

   | `reason` contains | Fix |
   | --- | --- |
   | `PyTorch is not installed` | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
   | `CPU-only build` | The default PyPI wheel was installed. Uninstall and reinstall from the CUDA index. |
   | `no CUDA device is visible` | Driver or `CUDA_VISIBLE_DEVICES`. Confirm with `nvidia-smi`. |

4. **Do not set `PRISMPRICE_ALLOW_CPU=1` to get the job through.** It is for CI. A model trained on CPU cannot be reproduced against the artefacts its decision records will name, which quietly voids the audit guarantee for every price it produces. If you genuinely must, record it in the incident and retrain on GPU before those artefacts serve any decision.
5. **Out-of-memory is different from unavailable.** Reduce batch size or Monte-Carlo draws; do not switch device.

---

## 8. Model regression after retrain

Estimates moved sharply and the new model has not been promoted yet — which is the point of shadow mode.

1. **Do not promote.** Shadow → canary → rollout exists for exactly this.
2. **Compare against the ground-truth suite first.** ⚪ Every estimator is validated on synthetic data with a known true parameter. If the DML no longer recovers the synthetic $\beta$, the model is wrong; stop looking at real data.
3. **Then check for drift.** PSI on features and predictions. Wide CIs everywhere usually mean the training window lost price variation — often because the *previous* policy was working and prices stopped moving. This is the well-known feedback trap: a good policy starves its own successor of identifying variation. The answer is designed experiments (L5), not a modelling change.
4. **Check calibration before accuracy.** A model with better WAPE and broken p10/p90 coverage is worse for this system, because the objective consumes the distribution, not the point.
5. **Roll forward, not back, once diagnosed.** Pin the previous model version in config; do not delete the new artefacts.

---

## 9. Routine: promoting a policy change

Not an incident, but the procedure most likely to cause one.

1. Change the policy in config with a version bump. Never edit a threshold in place.
2. Run in **shadow** against live traffic for at least one full demand cycle (7 days for weekly-seasonal categories).
3. Compare: rung distribution, guardrail bind rates, recommendation deltas, and the projected CPPC impact.
4. **Canary** on a SKU slice that is representative rather than convenient — not just the low-risk long tail, since the tail will not surface the failure modes that matter.
5. Roll out with the circuit breakers armed.
6. Record the change: what, why, who approved, and the shadow evidence. The decision records will reference the policy version; the policy version needs to reference this.
