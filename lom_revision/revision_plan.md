# Revision Plan — BNP-ABTests (AOAS Submission)
*"Online activity prediction via generalized Indian buffet process models"*

**Decision:** Major Revision (AOAS)
**Editor:** Po-Ling Loh
**Deadline:** 6 months from decision (4 months recommended)
**Reviewers:** Associate Editor + Reviewer 1 (R1) + Reviewer 2 (R2)

---

## Editor & AE Summary

The editor's letter and AE report converge on three points:
1. The paper is a nice methodological contribution but not yet suitable for AOAS given its applied focus.
2. The practical impact is unclear — simulations don't clearly show superiority, data analysis is thin.
3. Novelty vs. Camerlenghi et al. (2022) needs sharper articulation.

The AE specifically recommends: "focus more on the data analysis and the downstream impact."

The editor additionally requests:
- A dedicated Section 2 introducing the motivating application and dataset
- Stay within ~20 pages (AOAS template)
- Minimize color in figures (use dashed/dotted lines)
- Point-by-point response to Editor, AE, and both referees in a single file

---

## Reviewer Summary

**R1** is positive ("I thoroughly enjoyed reading this paper") but has technical clarifications, mostly around Section 3.4 (D_M estimation) and notation. 11 specific points, all addressable.

**R2** is the critical reviewer. Major concerns: (a) novelty vs. Camerlenghi, (b) paper reads as methodology not applied, data analysis thin, (c) unclear primary goal, (d) theoretical results lack practical interpretation. Plus 12 minor points and several typos.

---

## Macro Themes

| ID | Theme | Summary |
|----|-------|---------|
| **T1** | **Framing & Practical Impact** | AE + R2 (major b, c): sharpen the paper's goal, demonstrate practical impact on real decisions. Editor requests dedicated Section 2 for motivating application. |
| **T2** | **Real Data Analysis** | AE + R2 (major b, minor a, d, j): enrich data analysis with visualizations, dataset descriptions, decision-making evidence. |
| **T3** | **Theoretical Exposition** | R2 (major a, d, minor f–h, l): clarify novelty, add practical interpretation. Mostly done in current revision. |
| **T4** | **Section 3.4 & Notation (R1)** | R1 (points 1–11): clarify D_M section, fix notation issues, resolve cross-references. |
| **T5** | **Simulations & Presentation** | R2 (minor i, k): rebalance simulations vs. real data. Editor: 20-page limit, color minimization. |
| **T6** | **Typos & Mechanical** | R1 + R2: numerous typos, unresolved cross-references, formatting. |

---

## Status Legend

> **Status icons:** 🔲 Not started · 🔄 In progress · ✅ Done

> **Priority:** 🔴 High · 🟡 Medium · 🟢 Low

---

## Data Strategy

See `data_pipeline.md` for full details. Summary:

- **Primary new dataset:** REES46 eCommerce behavior data (Kaggle). Per-user, per-event timestamps over 5 months. Can be aggregated to (user_id, day, count) and windowed into hundreds of "experiments."
- **Secondary:** Semi-synthetic enrichment of ASOS data with synthetic re-triggers. Demonstrates NB-SSP on data where first-triggers are real.
- **Keep:** Existing ASOS analysis (TG-SSP / Be-SSP comparison).

This combination addresses every open data-dependent item (T1.b, T1.c, T2.a–e).

---

## Revision Tables

### T1 — Framing & Practical Impact

| Item | Source | Priority | Summary | Status |
|------|--------|----------|---------|--------|
| T1.a | AE, R2-c, Editor | 🔴 High | Clarify paper's primary goal and estimand. | ✅ Done |
| T1.b | AE, R2-b | 🔴 High | Demonstrate practical impact (stopping decisions). | ✅ Done — Narrative A (hitting-time MAE) implemented. Script: `fitting/compute_hitting_time_mae.py`. Results on REES46 (k=21,50,100), UCI, ASOS. NB-SSP reduces hitting-time MAE by 22–42% on ASOS vs competitors. Paper §6.3 updated (green) with combined ASOS+REES46 table, decision figure (trajectory case study + planning error boxplots), and power connection remark. Full analysis in `lom_revision/latex/content/7_hitting_time_mae.tex`. |
| T1.c | R2-b | 🔴 High | Add calibration diagnostics (CI coverage). | ✅ Done — Honest framing adopted. Paper §6.3 includes "Note on credible intervals" paragraph explaining plug-in EB coverage shortfall. Discussion §7 already has limitation paragraph with remedies (fully Bayesian, conformalized-Bayes). Reply letter updated to remove "calibrated uncertainty" overclaim. Coverage results documented in `lom_revision/latex/content/6_fitting_evaluation.tex` §6.6. |
| T1.d | Editor | 🔴 High | Add dedicated Section 2 for motivating application & dataset. | ✅ Done |

- **T1.a** — Rewrite intro to be precise: the goal is *prediction of user engagement to inform experiment duration decisions*. The estimand is U_{D₀}^{(D₁)} and T_{D₀}^{(D₁)}. The decision: given pilot data, should the experiment continue or stop? Frame around D_M estimation as the decision-support tool. Address R2-c directly: this is a design/planning contribution, not causal estimation.
- **T1.b** — Add concrete case studies showing: "with 7 days of pilot, our method predicted X users at day 28; the experiment could have been stopped Y days early." Compare to competing methods.
- **T1.c** — CI coverage computed and figures generated (A3 done). However, empirical coverage is 46% on UCI and ≤2% on REES46/ASOS. The reply letter and manuscript currently frame this as "calibrated uncertainty" — this is misleading and a reviewer will notice. **Must either**: (a) investigate and fix the coverage (model misspecification? interval construction bug?), or (b) reframe honestly: acknowledge poor coverage on REES46/ASOS, explain why (e.g., heavy extrapolation, model assumptions), and reposition the contribution accordingly. This is the highest-risk open item for acceptance.
- **T1.d** — Editor explicitly requests: "a dedicated section, typically Section 2, to introduce the motivating application and dataset." Restructure: current Section 2 (data + existing methods) becomes the new Section 2 with a richer dataset introduction up front.

---

### T2 — Real Data Analysis

| Item | Source | Priority | Summary | Status |
|------|--------|----------|---------|--------|
| T2.a | R2-j, Editor | 🔴 High | Fuller dataset description (types, goals, suitability). | ✅ Done |
| T2.b | R2-minor-a | 🟡 Medium | Visualize data characteristics (sparsity, heavy tail). | ✅ Done |
| T2.c | R2-minor-d | 🟡 Medium | Reference or visualization for power-law claim. | ✅ Done |
| T2.d | R2-minor-c | 🟡 Medium | Polish response on "isn't M always known?" | ✅ Done |
| T2.e | R2-j, AE | 🔴 High | Show decision-making impact on real data. | ✅ Done |

- **T2.a** — Describe REES46 dataset: cosmetics e-commerce, event types (view/cart/purchase), 5 months, 1.6M users. Explain how we window it into experiments. For ASOS: describe what's available. This directly addresses the editor's request for a motivating dataset section.
- **T2.b** — Figure: (i) histogram of per-user trigger counts on log-log scale, (ii) cumulative new users over time showing power-law growth.
- **T2.c** — Cite Clauset et al. (2009) "Power-law distributions in empirical data." Pair with T2.b visualization.
- **T2.d** — Current reply draft is solid. Polish and ensure Section 2 text reflects the argument.
- **T2.e** — Pick 3–5 REES46 experiment windows. Show prediction trajectories, D_M intervals, comparison to competitors.

---

### T3 — Theoretical Exposition

| Item | Source | Priority | Summary | Status |
|------|--------|----------|---------|--------|
| T3.a | R2-a | ✅ Done | Clarify contribution vs. Camerlenghi et al. (2022). | ✅ Done |
| T3.b | R2-d | ✅ Done | Add practical interpretation after Corollary 3.5 and Prop 3.7. | ✅ Done |
| T3.c | R2-f | ✅ Done | Reorganize Section 3.1 (trait process motivation). | ✅ Done |
| T3.d | R2-g | ✅ Done | Clarify NB parametrization. | ✅ Done |
| T3.e | R2-h | ✅ Done | Clarify importance of prior choice. | ✅ Done |
| T3.f | R2-e | ✅ Done | Clarify first-trigger-time modeling advantage. | ✅ Done |
| T3.g | R2-l | ✅ Done | Add model selection guidance in Discussion. | ✅ Done |
| T3.h | R2-minor-b | ✅ Done | Clarify difference from algorithmic approaches in Section 2.2. | ✅ Done |
| T3.i | R2-i | ✅ Done | Fix SSP vs Be-SSP naming confusion. | ✅ Done |
| T3.j | R2-l | 🟡 Medium | Discuss how to assess model fit / select score distribution. | ✅ Done |

- **T3.j** — R2 asks for model fit assessment beyond just choosing Be vs TG vs NB. Consider adding: marginal likelihood comparison across models, or posterior predictive checks. The Discussion paragraph on model choice (already added) partially addresses this, but a concrete diagnostic procedure would strengthen it.

---

### T4 — Section 3.4 & Notation (R1)

| Item | Source | Priority | Summary | Status |
|------|--------|----------|---------|--------|
| T4.1 | R1-1 | 🔴 High | Clarify Section 3.4: D_M definition, "new" vs "unobserved" users, relationship of F' to Z^FT. | ✅ Done |
| T4.2 | R1-2 | 🟢 Low | Eq.(1): second sum should run from D₀+1 to D₀+D₁. | ✅ Done |
| T4.3 | R1-3 | 🟡 Medium | p.4: "this assumption clearly..." — too strong without distributional assumptions. Soften or add conditions. | ✅ Done |
| T4.4 | R1-4 | 🟢 Low | Definition 3.1: change iid to ind (depends on n). | ✅ Done |
| T4.5 | R1-5 | 🟢 Low | Eq.(5): s vs θ typo; rephrase "puts to zero" more formally. | ✅ Done |
| T4.6 | R1-6 | 🟢 Low | Mention P₀ is non-atomic in Definition 3.2 or preceding paragraph. | ✅ Done |
| T4.7 | R1-7 | 🟢 Low | End of Definition 3.2: μ̃ vs μ. | ✅ Done |
| T4.8 | R1-8 | 🟢 Low | Theorem 3.4: define π_G before use. | ✅ Done |
| T4.9 | R1-9 | 🟢 Low | Capitalize "Theorem" and "Algorithm" consistently. | ✅ Done |
| T4.10 | R1-10 | 🟢 Low | Proposition 3.6: sum for U should start from D₀+1. | ✅ Done |
| T4.11 | R1-11 | 🟢 Low | p.9: "usres" → "users". | ✅ Done — "hitherto" typo fixed at line 145 of `3_bnp_method.tex`. |

- **T4.1** — This is the most substantive R1 comment. Section 3.4 needs a rewrite: (i) define D_M clearly as the number of additional days beyond D₀ needed to accumulate M new users, (ii) clarify that "new users" = users not observed in pilot, (iii) make the connection between F' and Z^FT explicit with a sentence like "F' is the restriction of Z^FT to users whose first trigger falls after D₀."
- **T4.2–T4.11** — All straightforward fixes. Batch these in a single editing pass.

---

### T5 — Simulations & Presentation

| Item | Source | Priority | Summary | Status |
|------|--------|----------|---------|--------|
| T5.a | R2-k | 🟡 Medium | Consider moving some simulations to supplement; expand real data. | ✅ Done |
| T5.b | Editor | 🟡 Medium | Stay within ~20 pages (AOAS template). | 🔄 In progress — not verifiable without compile. |
| T5.c | Editor | 🟢 Low | Minimize color in figures; use dashed/dotted lines. | 🔄 In progress — not verifiable from .tex source alone. |

- **T5.a** — Move "data from true model" simulations (Section 5.1) to supplement. Keep Zipf (5.2) and interval comparison (5.3). Use freed space for richer Section 6.
- **T5.b** — Current paper is likely over 20 pages. Moving simulations to supplement + tightening prose should help. Check page count after restructuring.
- **T5.c** — Audit all figures. Replace color-only distinctions with line styles (dashed, dotted, markers).

---

### T6 — Typos & Mechanical

| Item | Source | Priority | Summary | Status |
|------|--------|----------|---------|--------|
| T6.1 | R2 | 🟢 Low | Unresolved cross-references (??) throughout. | 🔄 In progress — not verified in source. |
| T6.2 | R2 | 🟢 Low | p.3: F_n redundant in tuple (can be derived from A). | 🔄 In progress — listed in C3 as verified but not confirmed by audit. |
| T6.3 | R2 | 🟢 Low | p.5 Section 3.2: define E(1) (exponential distribution). | 🔲 Not started |
| T6.4 | R2 | 🟢 Low | Corollary 3.5: define Δ̃_{1,h}^{-α} in the corollary statement. | 🔲 Not started |
| T6.5 | R2 | 🟢 Low | p.11: "does require" → "does not require". | 🔲 Not started |
| T6.6 | R2 | 🟢 Low | p.16: "$\tau$" not formatted as math. | 🔲 Not started |
| T6.7 | main.tex | 🟢 Low | Abstract \input'd twice. | ✅ Done |
| T6.8 | main.tex | 🟢 Low | Keyword: "user prdiction" → "user prediction". | ✅ Done |
| T6.9 | 1_intro.tex | 🟢 Low | Double brace in \Cref{{sec:numerical_implementation}}. | ✅ Done |
| T6.10 | 2_data... | 🟢 Low | Typos: "uesers sexhibit", "propsensity", "Morever". | 🔲 Not started |
| T6.11 | 3_bnp... | 🟢 Low | Typos: "racall", "ininitely", "uner", "whcih", "hiterto", "predicitve", "migh", "uesers", "consits". | 🔄 In progress — "hitherto" fixed; remaining typos in this file need a pass. |
| T6.12 | 4_num... | 🟢 Low | "close form" → "closed-form". | 🔲 Not started |
| T6.13 | 5_num... | 🟢 Low | "accuarcy", "esimates". | 🔲 Not started |
| T6.14 | 6_real... | 🟢 Low | "dasetest" → "datasets". | 🔲 Not started |
| T6.15 | 3_bnp... | 🟢 Low | Verify file not truncated (line 345). | 🔲 Not started |
| T6.16 | Various | 🟢 Low | Remove all \textcolor{red} markers before submission. | 🔲 Not started — C4 was reverted (git restore on 2026-04-29). |

---

## Priority Ordering (Suggested Workflow)

1. **Data pipeline** — Download REES46, preprocess, create experiment windows
2. **T2.b, T2.c** — Power-law visualizations (quick win, uses new data)
3. **T1.c** — Calibration study across experiment windows
4. **T1.b, T2.e** — Decision-impact case study (3–5 experiments)
5. **T2.a, T1.d** — Write dataset description + restructure Section 2
6. **T1.a** — Reframe intro (do after analyses)
7. **T4.1** — Rewrite Section 3.4 (R1's main concern)
8. **T4.2–T4.11** — R1 notation fixes (batch)
9. **T3.j** — Model fit assessment discussion
10. **T5.a–c** — Restructure simulations, page count, figure colors
11. **T6.1–T6.16** — Proofreading pass (do last)

---

## Audit Findings — 2026-04-30

Reviewer-pushback agent (`full` audit @ 2026-04-29T22:00:51Z) checked the reply letter against the manuscript source. Results: 22 ✅ SUFFICIENT · 5 ⚠️ PARTIAL · 0 ❌ INSUFFICIENT.

**Submission risk: MEDIUM**

### Open gaps (require action before submission)

| Gap | Maps to | Severity | Finding | Action required |
|-----|---------|----------|---------|-----------------|
| CI coverage misrepresentation | T1.c / AE.3 / R2.b | 🔴 HIGH | Coverage is 46% UCI, ≤2% REES46/ASOS. Reply claims "calibrated uncertainty." | Fix coverage OR reframe claim honestly. Owner decision required. |
| `\iid` fix in wrong definition | T4.4 / R1.4 | ✅ DONE | Fixed `\iid` → `\ind` in `def:models` line 107 of `3_bnp_method.tex`. | — |
| "hiterto" typo + proofreading | T6.11 / AE.6 / R1.11 | 🟡 MEDIUM | "hitherto" fixed at line 145. Remaining T6.3–T6.6, T6.10, T6.12–T6.15 still need a pass. | Full proofreading pass on remaining T6 items. |

### Items confirmed done by audit (previously Not started in plan)

T1.a, T1.b, T1.d, T2.a–e, T3.j, T4.1–T4.3, T4.5–T4.10, T5.a, T6.7–T6.9 — all verified SUFFICIENT in manuscript source.

---

## To-Do: CI Coverage Correction (T1.c follow-up)

**Goal:** Improve empirical coverage of credible intervals, currently 46% on UCI and ≤2% on REES46/ASOS (95% nominal). Root cause: empirical Bayes plug-in hyperparameters understate total uncertainty.

**Approach log** (add entries as attempts are made):

| # | Date | Method | Result | Notes |
|---|------|--------|--------|-------|
| — | — | Baseline (plug-in EB) | UCI 46%, REES46 1–2%, ASOS 0% | Current state |

**Candidate approaches to explore (in order of effort):**

1. **CI inflation factor** — multiply CI half-widths by a dataset-specific constant calibrated on held-out windows. Quick, unprincipled, but may be sufficient for the reply.
2. **Parametric bootstrap over hyperparameters** — resample (α̂, σ̂, Δ̂) from their asymptotic distribution and propagate uncertainty into the predictive CIs. Moderate effort.
3. **Profile likelihood CIs** — replace point estimates with likelihood-ratio-based confidence regions for hyperparameters, then integrate. More principled.
4. **Full MCMC** — place a prior on hyperparameters and sample jointly. Most principled, highest cost; likely out of scope for this revision.

**Decision rule:** If approach 1 or 2 brings coverage to ≥80% on UCI and ≥50% on REES46, it is worth including. Otherwise, honest framing (already applied) is sufficient.
