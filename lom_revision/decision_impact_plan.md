# Decision-Level Impact Plan
**Date:** 2026-05-04  
**Status:** Narrative A implemented and integrated into paper  
**Linked to:** `revision_plan.md` → T1.b (decision impact), T1.c (CI coverage)  
**Addresses:** AE concern "demonstrate earlier stopping, fewer underpowered launches"

---

## Context

The LLM-as-judge pass (2026-04-30) identified two blocking gaps for AOAS acceptance:
1. No decision-level demonstration — the paper shows accurate point predictions but never shows a concrete decision scenario
2. CI coverage ≤2% on REES46/ASOS — acknowledged but not remediated

This document plans two complementary narratives to address gap 1. Gap 2 is addressed separately (see `revision_plan.md` T1.c).

---

## Narrative A: Duration Planning via Participation Targets

### One-sentence pitch
Given pilot data, our method predicts when an experiment will have enough users to detect a meaningful effect — more accurately than baselines.

### Setup
- Dataset: UCI (13 experiments) or REES46 k=21 (186 experiments). After D₀=7 days, fit BNP model.
- Fix participation targets M = η × N_{D₀} for η ∈ {1.5, 2, 3}.
- Compute predicted hitting time: $\hat{D}_\eta$ = smallest ℓ such that $N_{D_0} + \hat{U}_{D_0}^{(\ell)} \geq M$.
- Compare to true $D_\eta$ from actual data.

### Baselines
- **Linear extrapolation:** assume new users arrive at pilot rate → $\hat{D}_\eta^{\text{lin}} = (M - N_{D_0}) / (N_{D_0}/D_0)$
- **Jackknife (J3):** already fitted on UCI/REES46
- **Constant (no growth):** $D_\eta = \infty$ — shows value of any forecasting

### Output metric
Mean absolute error in days: $|\hat{D}_\eta - D_\eta|$, averaged across experiments and η values.

### Expected result
NB-SSP has lower MAE than linear extrapolation, especially for larger η (longer horizons where sub-linear growth matters most). Linear extrapolation systematically underestimates $D_\eta$ because it ignores the power-law slowdown.

### Paper framing
> "Table X shows that NB-SSP regression predicts the time to reach 2× pilot participation within Y days on average, compared to Z days for linear extrapolation. This translates directly to more accurate experiment duration planning: an experimenter using our method would plan for X days, avoiding the Y-day underestimate that leads to underpowered experiments."

### Connection to power
The participation target M corresponds to a specific power level for a fixed effect size δ and variance σ². Specifically, power ≈ Φ(√M · δ/σ − z_{α/2}). So "time to M users" = "time to target power" under a fixed δ/σ assumption. This connection can be stated in a remark without requiring a specific δ/σ value.

### Implementation
- **Code:** Extend `SubmissionAOAS/fitting/compute_dm.py` to compute hitting-time MAE
- **Data:** REES46 k=21 (186 experiments, well-behaved parameters) preferred over UCI (13 experiments, degenerate D_M on some)
- **Effort:** ~1 day
- **Risk:** Low — D_M code already written, just needs hitting-time inversion and MAE computation

---

## Narrative B: Re-triggering Interventions

### One-sentence pitch
When the treatment is designed to increase user re-engagement, the BNP model directly measures the intervention's effect on the re-triggering distribution — something non-parametric competitors cannot do.

### Setup
Consider an A/B test where treatment T increases re-triggering (e.g., push notification, loyalty reward, re-engagement email). Fit BNP model separately on control arm (C) and treatment arm (T). The fitted score distribution parameter $\hat{\theta}_n$ (or $\hat{\alpha}$, $\hat{\sigma}$ in NB-SSP) directly captures the re-triggering rate.

### What this gives
1. A *mechanistic* interpretation of the treatment effect — not just "more users" but "users return more often"
2. A prediction of the long-run effect: if treatment increases θ, how many more users trigger by day 90?
3. A natural test: does the posterior on $\theta^T - \theta^C$ exclude zero?

### Data options (in order of preference)
1. **Proprietary data** (1,774 experiments) — identify re-engagement experiments by experiment type/description. Strongest if available.
2. **Semi-synthetic** — take REES46 control arm, simulate treatment by inflating score distribution (multiply θ_n by 1.2), fit both arms, show recovery of the inflation factor.
3. **ASOS** — unlikely (first-trigger only, no re-triggering structure).

### Paper framing
> "When the treatment targets user re-engagement, the BNP model provides a natural decomposition: the treatment effect on θ_n captures the change in repeat-visit propensity, while the effect on α captures the change in new-user acquisition rate. This decomposition is not available to non-parametric competitors such as Jackknife or Good-Toulmin, which treat the trigger process as a black box."

### Implementation
- **If proprietary data:** identify re-engagement experiments, fit per-arm, compare θ distributions
- **If semi-synthetic:** ~1 day to construct and fit
- **Effort:** 2–3 days
- **Risk:** Medium — semi-synthetic may feel contrived; proprietary data requires experiment identification

---

## Decision Framework

| | Narrative A | Narrative B |
|---|---|---|
| Closes AE's blocking gap | ✅ Directly | ⚠️ Partially |
| Requires new data | ❌ No | ✅ Yes (or semi-synthetic) |
| Effort | ~1 day | 2–3 days |
| Novelty vs. Camerlenghi | Low | High |
| Risk | Low | Medium |
| Paper section | Extends §6.3 (already scaffolded) | New subsection or remark |

## Recommended approach
1. **Implement A first** — closes the blocking gap, low risk, 1 day
2. **Add B as a remark** — 1 paragraph in §6 or §7: "In settings where the treatment targets re-engagement, the BNP model additionally provides..." without full empirical demonstration
3. **If time permits:** prototype B on proprietary data or semi-synthetic; promote to a subsection if results are clean

---

## Open questions before implementing A
1. UCI vs. REES46: REES46 k=21 (186 experiments) preferred — better-behaved parameters, more statistical power for the MAE comparison. Confirm?
2. η values: {1.5, 2, 3} — does this cover the practically relevant range for the datasets?
3. Should the hitting-time MAE table replace or supplement the existing §6.3 D_η framework?
