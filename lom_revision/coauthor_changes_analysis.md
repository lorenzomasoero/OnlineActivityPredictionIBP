# Co-Author Changes Analysis (commit 1e2edf3)

Date: 2025-04-12
Compared: ae5d7e8 → 1e2edf3

## Summary

The co-author made two types of changes:
1. **Substantive framing edits** (blue text) — directly addressing R2-c (unclear goal) and R2-d (no practical interpretation)
2. **Typo/grammar fixes** — addressing many items from our T6 list and some from T4

---

## Substantive Changes (blue text additions)

### Abstract (`0_abstract.tex`)
- **Old:** "Motivated by the problem of activity prediction for A/B tests at Amazon"
- **New:** "Motivated by the need to make early, design-facing decisions in A/B tests (e.g., how long to run an experiment or when it is safe to stop)"
- **Added:** "The primary estimands are predictive quantities that support duration and resource-allocation decisions; causal-effect estimation is downstream and not the focus here."
- **Impact on revision plan:** This directly addresses **T1.a** (clarify primary goal). Aligns perfectly with our Q1 decision (duration planning framing).

### Introduction (`1_intro.tex`)
- **Added** a clarifying sentence after the first paragraph: "we do not focus on causal-effect estimation itself; rather, we address the upstream design problem of forecasting participation and activity to plan experiment duration."
- **Added** a full paragraph stating the three estimands (U, T, D_M) and their decision purpose — matches our Q1 use cases exactly.
- **Added** a "Decision target" paragraph explaining D_M with credible intervals for stop/extend decisions.
- **Added** citations to `\citep{Gualavisi2025,Masoero2025}` (the power calculations and early termination papers).
- **Reworded** contribution #1 to emphasize "forecasting user engagement to support experiment design."
- **Reworded** contribution #3 to mention "duration and stopping decisions."
- **Impact on revision plan:** **T1.a is now substantially addressed.** The intro rewrite we planned is largely done.

### Section 2 (`2_data_and_existing_methods.tex`)
- **Added** a sentence linking to the design/duration framing and citing Gualavisi2025, Masoero2025.
- **Added** at end of Section 2: "Beyond the likelihood extension, our inferential targets are explicitly decision-oriented (multi-horizon forecasts, total-trigger prediction, and time-to-threshold D_M), which are not addressed in Camerlenghi et al. and are central to A/B test duration and early-stopping decisions."
- **Fixed** typos: "uesers sexhibit" → "users exhibits", "propsensity" → "propensity", "Morever" → "Moreover"
- **Impact on revision plan:** Partially addresses **T3.a** (novelty vs Camerlenghi) with the decision-oriented framing. Fixes **T6.10** (typos in Section 2).

### Section 3 (`3_bnp_method.tex`)
- **Added** inline definition of "score distribution" in Definition 3.1: "(the score distribution, i.e., the conditional law of A_{d,n}|θ_n)"
- **Added** definition of E(1) as exponential distribution with rate 1.
- **Added** definition of Δ̃_{1,h} in Theorem 3.4 statement.
- **Added** practical interpretation after Corollary 3.5: explains that the choice between Be-SSP and TG-SSP affects stopping/extension decisions via hyperparameter fitting.
- **Added** practical interpretation after Proposition 3.7: the mixture decomposition supports extend/stop decisions.
- **Fixed** sum index in Proposition 3.6: D₀+1 to D₀+D₁ (was D₀+1 to D₁).
- **Fixed** typos: "racall" → "recall", "casted" → "cast", "ininitely" → "infinitely", "uner" → "under", "whcih" → "which", "hiterto" → "hitherto", "predicitve" → "predictive", "consits" → "consists", "migh" → "might", "uesers" → "users"
- **Impact on revision plan:**
  - **T3.b** (practical interpretation) — now done
  - **T3.c** (score distribution definition) — now done via R2-f
  - **T4.4** (iid→ind) — NOT addressed (still says \ind)
  - **T4.5** (s vs θ in Eq 5) — NOT checked
  - **T4.6** (P₀ non-atomic) — NOT addressed
  - **T4.8** (define π_G) — addressed via blue text in Theorem 3.4
  - **T4.10** (sum index) — DONE
  - **T6.3** (define E(1)) — DONE
  - **T6.4** (define Δ̃_{1,h}) — DONE
  - **T6.5** ("does require" → "does not require") — DONE (in Section 4)
  - **T6.11** (typos in Section 3) — mostly DONE

### Section 4 (`4_numerical_implementation.tex`)
- **Fixed** "close form" → "closed-form"
- **Fixed** "does require" → "does not require"
- **Impact:** **T6.5, T6.12** — DONE

### Section 5 (`5_numerical_illustrations.tex`)
- **Fixed** "accuarcy" → "accuracy", "esimates" → "estimates", "dataset" → "datasets", "prediction" → "predictors", "addition to providing esimates for" → "in addition to providing estimates for", "retrigger" → "re-trigger"
- **Fixed** "misscalibrated" → "miscalibrated", "\\tau" → "$\tau$"
- **Impact:** **T6.6, T6.13** — DONE

### Section 6 (`6_real_data.tex`)
- **Fixed** "dasetest" → "datasets", "maximum marginal" → "maximum marginal likelihood", "retriggered" → "re-triggered"
- **Impact:** **T6.14** — DONE

### Section 7 (`7_discussion.tex`)
- **Fixed** "estimates the parameters" → "estimated parameters"
- Minor grammar fix.

### Reply letter (`reply/reply_letter.tex`)
- **Added** blue-text response to R2-b: explains the revised abstract/intro framing.
- **Added** blue-text response to R2-c: explains the primary estimand and decision target.
- **Added** blue-text response to R2-d: explains the practical interpretation additions.
- **Fixed** various typos in existing responses.
- **Impact:** Reply letter now has substantive responses for R2-b, R2-c, R2-d (previously empty or stub).

---

## Revision Plan Status Update

### Items now DONE (changed from open to done):

| Item | Description | How addressed |
|------|-------------|---------------|
| T1.a | Clarify paper's primary goal | Intro rewrite with duration planning framing, three estimands, decision target paragraph |
| T3.b | Practical interpretation after Cor 3.5 and Prop 3.7 | Blue text additions in Section 3 |
| T4.8 | Define π_G in Theorem 3.4 | Blue text: "Let Δ̃_{1,h} denote the latent jump-scale..." |
| T4.10 | Sum index in Prop 3.6 | Fixed: D₀+1 to D₀+D₁ |
| T6.3 | Define E(1) | Blue text in Section 3.2 |
| T6.4 | Define Δ̃_{1,h} in Corollary 3.5 | Blue text in Theorem 3.4 |
| T6.5 | "does require" → "does not require" | Fixed in Section 4 |
| T6.6 | Format τ as math | Fixed in Section 5 |
| T6.10 | Typos in Section 2 | Fixed |
| T6.11 | Typos in Section 3 | Mostly fixed |
| T6.12 | "close form" → "closed-form" | Fixed |
| T6.13 | "accuarcy", "esimates" | Fixed |
| T6.14 | "dasetest" → "datasets" | Fixed |

### Items still OPEN:

| Item | Description | Status |
|------|-------------|--------|
| T1.b | Demonstrate practical impact (case studies) | Still needs data analysis |
| T1.c | Calibration diagnostics (CI coverage) | Still needs data analysis |
| T1.d | Dedicated Section 2 for motivating application | NOT done — editor requested this, co-author didn't restructure |
| T2.a | Fuller dataset description | NOT done — still thin |
| T2.b | Visualize data characteristics | Figures exist in lom_revision/plots/ but not in paper |
| T2.c | Power-law reference | NOT in paper yet |
| T2.d | Polish "isn't M always known" response | Reply letter response exists, minor polish |
| T2.e | Decision-making impact on real data | Still needs data analysis |
| T3.j | Model fit assessment procedure | NOT done |
| T4.1 | Rewrite Section 3.4 (D_M clarity) | NOT done — R1's main concern |
| T4.2 | Eq(1) sum index | NOT checked |
| T4.3 | "this assumption clearly..." too strong | NOT done |
| T4.4 | iid → ind in Definition 3.1 | NOT done |
| T4.5 | s vs θ in Eq(5) | NOT checked |
| T4.6 | P₀ non-atomic mention | NOT done |
| T4.7 | μ̃ vs μ in Definition 3.2 | NOT done |
| T4.9 | Capitalize Theorem/Algorithm consistently | NOT done |
| T4.11 | "usres" → "users" p.9 | NOT checked (may be fixed by other typo fixes) |
| T5.a | Move simulations to supplement | NOT done |
| T5.b | Page limit ≤20 pages | NOT checked |
| T5.c | Figure colors → line styles | NOT done |
| T6.7 | Abstract \input'd twice | NOT done |
| T6.8 | Keyword "user prdiction" | NOT done |
| T6.9 | Double brace \Cref{{...}} | NOT done |
| T6.15 | Verify 3_bnp_method.tex not truncated | NOT done |
| T6.16 | Remove \textcolor{red} markers | NOT done (and new blue markers added) |

### New items from co-author's changes:

| Item | Description |
|------|-------------|
| NEW.1 | Two new citations added: Gualavisi2025, Masoero2025 — need to add to references.bib |
| NEW.2 | Blue text markers ({\color{blue}...}) need to be reviewed and eventually removed before submission |
| NEW.3 | R1 comments still completely unaddressed in the reply letter |

---

## Key Takeaway

The co-author addressed the **framing problem** (T1.a) thoroughly — the intro now clearly states the duration-planning goal with three estimands and a decision target paragraph. This was the single most important reviewer concern. Many typos are also fixed.

The **data analysis gap** (T1.b, T1.c, T2.a-e) remains the main open work. The paper still has the same thin Section 6 with proprietary + ASOS data. Our UCI/REES46 pipeline results haven't been integrated yet. This is where the remaining effort needs to go.

R1's comments (T4.1–T4.11) are partially addressed (sum index, some definitions) but the main concern (Section 3.4 rewrite) is untouched.
