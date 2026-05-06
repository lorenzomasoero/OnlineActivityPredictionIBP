# Strategy Questions — Data-Driven Response to Reviewers

These are the open strategic decisions we need to make before writing any analysis code.
Each maps to specific reviewer comments and determines what analyses we run.

---

## Q1: What is the primary decision the method supports?

**Reviewer source:** R2-c, AE

**The problem:** The reviewer says the intro oscillates between (i) improving causal effect estimation and (ii) improving experimental design. We need to commit to one primary framing.

**Options:**
- **(a) Stopping rule:** "Given D₀ days of pilot, should we extend the experiment or stop?" — This frames the method as a decision-support tool. The D_M estimation (Section 3.4) is the natural deliverable: "you need X more days to reach your target sample size."
- **(b) Duration planning:** "How many more days do we need to reach N users?" — Same as (a) but framed as planning rather than real-time decision.
- **(c) Pure forecasting:** "How many users/triggers will we see at day D₀+D₁?" — This is what the accuracy metric v_{D₀}^{(D₁)} measures. Simpler framing but the reviewer explicitly wants to see decision impact, not just forecast accuracy.

**Implication for data analysis:**
- If (a) or (b): we need D_M interval computations on UCI data, plus a narrative showing "experiment could have stopped early"
- If (c): accuracy boxplots + calibration suffice, but may not satisfy the reviewer

**Decision:** Option (a) — stopping/duration planning. Primary framing: "given pilot data, how long should the experiment run?" D_M is the decision output; U and T are intermediate quantities that feed into it.

**Concrete use cases the method supports (to be spelled out in the intro):**
1. **Power/sample-size planning:** "After 7 pilot days, will we have enough users by day 28 to detect a given effect size?" — U predictions feed into power calculations.
2. **Exposure targets:** "How many days until at least k% of a known customer base is exposed to the treatment?" — D_M estimation directly answers this.
3. **Differential re-trigger forecasting:** "How will total trigger volume differ between arms over the follow-up?" — T predictions under the NB-SSP model address this.
4. **Connection to early stopping:** The predictions produced here can inform Bayesian early termination frameworks (Masoero et al., 2024, amazon.science) — not as a prerequisite, but as a complementary planning tool that operates at the experiment-design stage rather than the analysis stage.

**Experiments required to demonstrate Q1 (detailed in `lom_revision/scripts/` once designed):**

| ID | Use case | What we compute | Data needed | Output |
|----|----------|----------------|-------------|--------|
| E1 | Power planning | Û_{D₀}^{(D₁)} + 95% CI for all 13 UCI windows | Pilot matrices (D₀=7) | Accuracy table (v metric) + CI coverage rate |
| E2 | Exposure targets | D_M intervals for 3 selected experiments (small/medium/large) | Fitted parameters + Algorithm 1 | D_M point estimate + CI vs. truth; narrative per experiment |
| E3 | Re-trigger forecasting | T̂_{D₀}^{(D₁)} via NB-SSP for all 13 UCI windows | Full count matrices | Accuracy of total trigger prediction |
| E4 | Competitor comparison | Same as E1/E3 but for IBP, BB, BG, Jackknife, GT | Same data | Relative accuracy across methods |

**Implementation notes:**
- E1 and E4 share the same fitting loop — different models, same data, same outputs.
- E2 requires the D_M sampling algorithm (Algorithm 1 in the paper, `NegBintSBSP` class doesn't currently have it — may need to implement based on Theorem 3.8).
- E3 uses `total_retriggers_new_users` + `total_retriggers_old_users` from `utils_NBP.py`.
- All experiments use the 13 preprocessed UCI windows in `lom_revision/data/`.

---

## Q2: What shape should the "practical impact" demonstration take?

**Reviewer source:** R2-b, AE ("little evidence the method has practical impact and changes decisions")

**Options:**
- **(a) Case-study narrative:** Pick 3–5 UCI experiments. For each, show: prediction trajectory over time, D_M intervals, comparison to competitors. Write a paragraph per experiment: "Experiment X had 278 pilot users. Our method predicted 682 new users (truth: 682). BB predicted 450. Our 95% CI covered the truth; BB's did not. Using our D_M estimate, the experimenter could have committed to a 28-day run on day 7 with calibrated confidence."
- **(b) Aggregate summary:** Across all 13 experiments, report: median accuracy, CI coverage, median D_M error. Less narrative, more statistical.
- **(c) Both:** Case studies for depth + aggregate for breadth.

**Implication for data analysis:**
- (a) requires fitting + D_M + careful selection of representative experiments
- (b) requires fitting + CI computation across all 13
- (c) requires both

**Decision:** Option (c) — both aggregate + case studies.

**Layer 1: Aggregate evidence (all 13 experiments)**

| Analysis | Methods | Output figure/table | Reviewer point |
|----------|---------|-------------------|----------------|
| Accuracy of Û | Be-SSP, TG-SSP, NB-SSP, IBP, BB, BG | Boxplot of v_{D₀}^{(D₁)} across 13 experiments | "practical advantage over competing methods" |
| CI coverage of Û | Be-SSP, TG-SSP, NB-SSP | Table: nominal 95% vs. empirical coverage | "calibrated risk," "sparse diagnostics" |
| Accuracy of T̂ | NB-SSP (only method that can do this) | Boxplot or table of accuracy across 13 experiments | NB-SSP contribution on public data |

**Layer 2: Case studies (3 experiments)**

| Experiment | Why chosen | What we show |
|-----------|-----------|-------------|
| Exp 1 (days 28–56, N_pilot=34) | Sparse pilot — stress test | Prediction trajectory + CI; D_M interval; narrative: "even with 34 users, our method produces a useful forecast" |
| Exp 6 (days 168–196, N_pilot=348) | Medium traffic — typical case | Same; narrative: "representative experiment, method predicts accurately, CI covers truth" |
| Exp 12 (days 336–364, N_pilot=474) | High traffic — holiday season | Same; narrative: "high-volume period, method scales, D_M interval is tight" |

For each case study, produce a 2-panel figure:
- Panel A: cumulative new users over 28 days. Black = truth, colored line = our prediction from day 7 + shaded CI, dashed = best competitor.
- Panel B: D_M bar chart — "days to reach 2× pilot users." Our estimate + CI vs. truth vs. competitor estimate.

One paragraph per experiment connecting prediction → decision → practical consequence.

**Draft response paragraph (for reply letter):**
"We now demonstrate the practical impact of our method on 13 publicly available experiment windows constructed from the UCI Online Retail dataset. Across all 13 experiments, our 95% credible intervals for U_{D₀}^{(D₁)} achieve X% empirical coverage, confirming calibrated uncertainty quantification. In terms of point prediction accuracy, [our method] achieves a median accuracy of Y%, compared to Z% for the best competitor. We illustrate the decision-support value through three case studies spanning sparse, medium, and high-traffic experiments, showing how the D_M intervals produced on day 7 of the pilot provide actionable duration estimates."

---

## Q3: Is 13 experiments enough for calibration?

**Reviewer source:** R2-b ("sparse diagnostics, calibrated risk")

**The concern:** 13 experiments is a small sample for reporting CI coverage. With 13 data points, a single miss takes coverage from 100% to 92%.

**Options:**
- **(a) Use 13 as-is.** Report coverage honestly. 13 is what we have; the proprietary analysis had 1,774 but we can't share it.
- **(b) Create more windows.** Use overlapping windows (e.g., sliding by 7 days instead of 28) to get ~50 experiments. Introduces dependence between windows but increases the sample.
- **(c) Vary D₀.** For each of the 13 windows, also try D₀ = 3, 5, 7, 10, 14. This gives 13 × 5 = 65 calibration points and shows how coverage changes with pilot length.
- **(d) Combine (b) and (c).**

**Implication for data analysis:**
- (a): minimal extra work
- (b): re-run preprocessing with overlapping windows
- (c): re-run fitting with multiple D₀ values per experiment
- (d): most work but strongest result

**Decision:** Use multiple data sources for calibration breadth:
- **ASOS:** 76 treatment arms (already available, first-trigger counts → Be-SSP, TG-SSP)
- **UCI Online Retail:** 13 non-overlapping windows + optionally more via overlap or varying D₀ (full counts → all three models)
- **REES46 (if downloaded):** 1 month (~3GB) would yield hundreds of windows with count data

Additionally, vary D₀ ∈ {3, 5, 7, 10, 14} within each dataset to show how coverage changes with pilot length. This is both a calibration strategy and useful content ("how much pilot data do you need?").

Total calibration points: 76 (ASOS) + 13–65 (UCI) + potentially hundreds (REES46) = well over 100.

**Implementation plan:**
1. Start with UCI (pipeline already built) + ASOS (data already in repo)
2. Add REES46 later if needed for additional scale
3. Report calibration separately per dataset and aggregated

**Reviewer source:** R2-minor-a ("visualize data characteristics"), R2-minor-d ("power-law reference"), Editor ("dedicated Section 2")

**Options:**
- **(a) New Section 2** (as editor requests): introduce the UCI dataset, show the power-law / heavy-tail plots, motivate why BNP is needed. Then Section 3 = existing methods, Section 4 = methodology, etc.
- **(b) Section 6** (real data analysis): keep current structure, add visualizations alongside results.
- **(c) Both:** Teaser visualization in Section 2 (one panel), full analysis in Section 6.

**Implication:** This is a structural decision about the paper, not an analysis decision. But it determines what figures we produce and where they go.

**Decision:** Option (a) — New Section 2 "Motivating Application and Data."
- Introduce the experiment duration planning problem
- Show 1–2 figures: (i) cumulative new users over time (power-law growth), (ii) histogram of per-user trigger counts (heavy tail)
- Cite Clauset et al. (2009)
- Explain why these features motivate BNP
- Current Section 2 (data + existing methods) becomes Section 3
- Section 6 (real data) contains full analysis results; reader already knows the data from Section 2

**Figures needed (from raw data, no model fitting):**
- Fig 1a: cumulative new users over time for a representative UCI experiment (e.g., Exp 6)
- Fig 1b: histogram of per-user total trigger counts across the full UCI dataset (log-log scale)

---

## Q5: What role does UCI data play relative to proprietary + ASOS?

**Reviewer source:** R2-j, AE

**Current paper has:**
- Proprietary data (1,774 experiments, first-trigger counts only, no re-triggers except for 50-experiment subset) — can't be shared or described in detail
- ASOS data (76 treatment arms, first-trigger counts only) — public but coarse

**Options:**
- **(a) UCI as primary, drop proprietary.** Fully reproducible. But loses the scale argument (1,774 experiments).
- **(b) UCI as primary public analysis, keep proprietary as secondary.** "We demonstrate on public data (UCI) and confirm at scale on proprietary data." Best of both worlds but the reviewer may still complain about the proprietary part being thin.
- **(c) UCI + ASOS + proprietary.** Three datasets. Risk: paper gets long. But shows breadth.
- **(d) UCI as primary, ASOS as secondary, drop proprietary.** Fully reproducible, two public datasets, different granularities (UCI has counts, ASOS has first-triggers only).

**Implication:** Determines how much of the current Section 6 we keep vs. rewrite.

**Decision:** Deferred until we see results from the UCI and ASOS experiments. Will decide then whether to keep proprietary, drop it, or use all three.

---

## Summary: What needs deciding before we code

| Question | Key choice | Blocks |
|----------|-----------|--------|
| Q1 | Primary framing (stopping rule vs. forecasting) | What analyses to run |
| Q2 | Case studies vs. aggregate vs. both | How to present results |
| Q3 | How many calibration points | Preprocessing decisions |
| Q4 | Where visualizations go | Paper structure |
| Q5 | Role of UCI vs. proprietary vs. ASOS | What to keep/drop in Section 6 |
