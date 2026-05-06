# Data Pipeline for Revised Real-Data Analysis

## Goal

Produce public, reproducible analyses that address the reviewer's core complaint:
"the data analysis feels thin." We need data that supports:

1. **Visualizations** of power-law / heavy-tail behavior (T2.b, T2.c)
2. **Calibration study** — credible interval coverage across many "experiments" (T1.c)
3. **Decision-impact case study** — concrete stopping-time examples (T1.b, T2.e)
4. **NB-SSP showcase** — count-valued re-trigger data (not just binary), which the ASOS data cannot provide

## Two Strategies

### Strategy A: Find New Fine-Grained Public Data

**What we need:** A dataset with (user_id, day, event_count) structure — i.e., for each user, how many times they triggered on each day, over a multi-week window. Ideally from an e-commerce or web-app context to match the paper's motivation.

**Candidate datasets:**

| Dataset | Source | Granularity | Size | Pros | Cons |
|---------|--------|-------------|------|------|------|
| REES46 cosmetics store | [Kaggle](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store) | user_id × event_time × event_type (view/cart/purchase) | ~20M rows, 5 months (Oct 2019–Feb 2020), 1.6M users | Per-event timestamps → can aggregate to (user, day, count). Multiple event types. Large scale. | Not an A/B test per se — would need to frame as "simulating an experiment window" |
| OTTO RecSys dataset | [GitHub](https://github.com/otto-de/recsys-dataset) | session-level events (clicks, carts, orders) | 12M sessions, 4 weeks | Real e-commerce, timestamped events | Session-based, not user-based longitudinal |
| UCI Online Retail | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/online+retail) | InvoiceNo × CustomerID × InvoiceDate × Quantity | ~500K rows, 1 year, UK retailer | Classic dataset, well-known, has customer IDs and dates | Smaller scale, invoice-level not event-level |
| Instacart Market Basket | [Kaggle](https://www.kaggle.com/c/instacart-market-basket-analysis) | user_id × order_id × order_dow × order_hour | 3.4M orders, 200K users | Large, per-user longitudinal orders | No exact dates (only day-of-week and days-since-prior-order) |
| Yoochoose RecSys 2015 | [RecSys Challenge](https://recsys.acm.org/recsys15/challenge/) | session_id × timestamp × item_id × category | 33M clicks, 6 months | Large, timestamped | Session-based, no persistent user IDs |

**Recommendation:** The **REES46 / eCommerce behavior dataset** is the strongest candidate. It has:
- Per-user, per-event timestamps over 5 months
- Can be aggregated to (user_id, day, count_of_events) trivially
- Large enough to simulate many "experiments" by windowing
- Multiple event types (view, cart, purchase) → can define "trigger" flexibly
- Publicly available on Kaggle

The **UCI Online Retail** dataset is a good secondary option — smaller but well-established in the literature.

### Strategy B: Semi-Synthetic Enrichment

**Idea:** Take the ASOS data (which only has first-trigger counts per day) and enrich it with synthetic re-trigger behavior generated from a plausible model. This lets us:
- Keep the real first-trigger structure (which is the hardest part to get right)
- Add count-valued re-triggers drawn from a model (e.g., NB with user-specific rates)
- Show that NB-SSP recovers the synthetic re-trigger structure while also predicting real first-triggers accurately

**How it works:**
1. For each ASOS experiment, we observe N_{d} (new users per day) for d = 1, ..., D.
2. For each user who first triggers on day F_n, generate re-trigger counts:
   - Draw θ_n ~ Beta(a, b) (user-specific engagement rate)
   - For d > F_n: A_{d,n} ~ NegBin(r, 1 - θ_n)
   - Choose (a, b, r) to produce realistic heavy-tailed behavior
3. Now we have a full (user, day, count) dataset where first-triggers are real and re-triggers are synthetic.
4. Fit all three models (Be-SSP, TG-SSP, NB-SSP) and show:
   - NB-SSP can predict both U and T
   - Calibration of credible intervals
   - Decision-impact examples

**Pros:** Fully reproducible, builds on existing data, clearly demonstrates the NB-SSP advantage.
**Cons:** Reviewer might push back on "you generated the re-triggers yourself." Need to be transparent about this and frame it as a controlled evaluation.

### Recommended Approach: Combine Both

1. **REES46 as the primary new dataset** — real data, fine-grained, publicly available. This is the "new real data analysis" that directly addresses the reviewer.
2. **Semi-synthetic enrichment of ASOS as a secondary analysis** — demonstrates NB-SSP on a structure where first-triggers are real. Frame it as: "to evaluate the NB-SSP model on data with known ground truth for re-triggers, we augment the ASOS first-trigger data with synthetic re-trigger counts."
3. **Keep the existing ASOS analysis** — it's already in the paper and works fine for TG-SSP / Be-SSP comparison.

## Pipeline Steps

### Phase 1: Data Acquisition & Preprocessing

```
Step 1.1: Download REES46 dataset from Kaggle
Step 1.2: Parse event timestamps → (user_id, date, event_type)
Step 1.3: Aggregate to (user_id, day_index, trigger_count) per "experiment"
Step 1.4: Create experiment windows:
          - Slice the 5-month period into overlapping or non-overlapping
            windows of D = 28–42 days
          - Each window = one "experiment"
          - Within each window, define pilot (D₀ = 7 days) and
            follow-up (D₁ = D - D₀ days)
Step 1.5: Compute ground truth for each experiment:
          - U_{D₀}^{(D₁)}: new users in follow-up
          - T_{D₀}^{(D₁)}: total triggers in follow-up
          - N_{D₀}: users in pilot
```

### Phase 2: Model Fitting & Prediction

```
Step 2.1: For each experiment window, fit:
          - Be-SSP (binary activity)
          - TG-SSP (first-trigger times only)
          - NB-SSP (count-valued activity)
          - Competing methods: BB, HBG, Jackknife, Good-Toulmin, LP
Step 2.2: Compute predictions:
          - Û_{D₀}^{(D₁)} for all methods
          - T̂_{D₀}^{(D₁)} for NB-SSP
          - 95% credible intervals for U and T
Step 2.3: Compute D_M intervals (days to reach participation threshold)
```

### Phase 3: Analysis & Figures

```
Step 3.1: Power-law visualization (T2.b, T2.c)
          - Histogram of per-user trigger counts (log-log scale)
          - Cumulative new users over time for representative experiments
          - Fit power-law exponent, cite Clauset et al. (2009)

Step 3.2: Accuracy comparison (extends current Figure 7)
          - Boxplots of v_{D₀}^{(D₁)} across all experiment windows
          - Compare all methods
          - Stratify by experiment size (small/medium/large)

Step 3.3: Calibration study (T1.c) — NEW
          - For each experiment: does truth fall in 95% CI?
          - Report empirical coverage across all experiments
          - Plot: nominal vs. empirical coverage at multiple levels
          - Stratify by D₁ (extrapolation horizon)

Step 3.4: Decision-impact case study (T1.b, T2.e) — NEW
          - Pick 3–5 representative experiments (small, medium, large)
          - For each: show prediction trajectory over time
          - Show D_M intervals: "our method says you need X more days"
          - Compare to competing methods
          - Narrative: "experiment could have been stopped Y days early"

Step 3.5: Total triggers prediction (NB-SSP showcase) — NEW for public data
          - Accuracy of T̂_{D₀}^{(D₁)} on REES46 data
          - This is currently only shown on proprietary data
```

### Phase 4: Semi-Synthetic ASOS Enrichment (if needed)

```
Step 4.1: For each ASOS experiment, generate synthetic re-triggers
Step 4.2: Fit NB-SSP and compute T̂
Step 4.3: Show calibration and accuracy
Step 4.4: Frame transparently as controlled evaluation
```

## Code Requirements

We need Python code that:
1. Loads and preprocesses the REES46 dataset
2. Creates experiment windows
3. Fits Be-SSP, TG-SSP, NB-SSP (need the existing codebase — is there a repo?)
4. Fits competing methods (BB, HBG, Jackknife, GT, LP)
5. Computes accuracy metrics, credible intervals, calibration
6. Generates all figures

**Key question:** Where is the existing code for fitting the BNP models? Is there a Python package or scripts associated with this paper?

## Output

The pipeline produces:
- 3–5 new figures for the paper
- Updated Section 6 with REES46 analysis
- Calibration table/figure
- Decision-impact narrative for 3–5 experiments
- All code and data publicly available (reproducibility)
