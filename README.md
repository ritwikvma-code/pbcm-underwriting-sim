# PBCM underwriting simulation — reproducibility package (v1.1.0)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22712490.svg)](https://doi.org/10.5281/zenodo.22712490)

Synthetic, policy-consistent evaluation suite accompanying *"Underwriting embedded
merchant cash advances from marketplace settlement data: a design blueprint with a
policy-consistent synthetic evaluation"* (manuscript version 3, September 2026).

This package regenerates **every quantitative value in the paper's tables and figures**
and **every value in the worked example** from a stated data-generating process and a
fixed seed. Version 1.1.0 keeps the data-generating process of v1.0.0 unchanged and
rebuilds the evaluation so that every reported outcome is measured under the terms
actually offered to the seller. See `CHANGELOG.md` for the itemised changes.

## Data statement

Fully synthetic. No platform, lender, customer, or company data is used, referenced, or
required. Every seller, transaction history, bureau score and repayment path is generated
at runtime from a seeded random number generator. There are no input data files.

## What is validated — and what is not

The simulation validates the **internal behaviour of the underwriting, sizing/pricing and
monitoring mechanisms** under a data-generating process (DGP) that is stated in full below.
Because the DGP is synthetic, results are evidence that the mechanisms behave as described
*when behavioural features carry information about repayment*; they are not estimates of
real-world model performance, default rates, loss rates or margins. The robustness
experiments (E3) show how far the behavioural signal can be degraded before the
conclusions change. The API layer described in the paper is a **proposal**: no service is
implemented or benchmarked here.

## Two label families, kept apart

* **Prediction target (reference product P0).** The scorer is trained to predict full
  repayment of a fixed reference advance — A₀ = 0.15·GMV₉₀ at factor 1.25 with a 15%
  holdback — within 1.5× its estimated window (16.25 weeks → 25-week deadline). Because
  A₀ is proportional to GMV₉₀ the deadline is a constant of the product terms, so the
  target measures the post-origination trajectory rather than seller size. P0 is a fixed
  product, **not** a policy-neutral one.
* **Policy outcomes (the advance actually offered).** Every pricing, portfolio, stress and
  monitoring result is computed on the offer the seller would actually receive — its own
  amount α(E)·GMV₉₀, its own tier price, and its own deadline ceil(1.5 × FR × α × 13 / 0.15).
  These outcomes are never presented as model-performance metrics.

## The data-generating process (E1, unchanged from v1.0.0)

For each of three cohorts of 12,000 sellers (training, held-out test, validation):

* A latent business-health variable *h* ~ N(0, 1) drives everything downstream.
* Category (6, equal weights) with a base return rate, a platform category-risk index and a
  hazard multiplier (fashion 1.25 … industrial 0.85).
* Tenure ~ lognormal (median ≈ 20 months). A **13-week transaction history** is simulated
  from a base weekly GMV (lognormal; depends weakly on *h* and tenure), a linear trend
  (0.25 *h* + noise) and multiplicative weekly noise whose scale falls with *h*. GMV₉₀,
  momentum and revenue CV are **computed from that history**.
* Return rate (category base × exp(−0.35 *h*)), within-category percentile, dispute rate,
  fulfilment score and repeat-buyer rate depend on *h* with noise. Platform concentration
  has a true value (Beta(4, 2)) and a **seller-declared value that 30% of sellers
  over-report** by 0.10–0.35; the model sees the declared value.
* **Bureau-proxy score** = 680 + 70·(ρ *h* + √(1−ρ²) ε), ρ = 0.55 for thick-file and 0.15
  for thin-file sellers (tenure < 12 months, or 25% of the rest).
* **Post-origination path (52 weeks):** lognormal random walk (drift 0.003 *h* +
  0.02·trend − 0.002, σ = 0.06) with a permanent shock process — weekly hazard
  0.004·exp(−0.9 *h*)·category multiplier·(1 − 0.3·true concentration); on a shock GMV is
  multiplied by a severity ~ U(0.05, 0.60) for the rest of the horizon. Shocks are
  independent across sellers. The held-out test cohort uses hazard × 1.20 (a **simulated
  cohort shift**, not a historical out-of-time sample); the validation cohort uses × 1.0.
  The simulator's weekly GMV is the collectible settlement base; a 12% collection
  shortfall relative to the sizing base is run as a stress.
* Amounts unpaid at 52 weeks are treated as written off with no later recovery and no
  servicing cost — an accounting horizon, not an observed closure event.

## Experiments

| ID | What it does | Paper |
|---|---|---|
| E2 | Held-out comparison of four scorers on the prediction target: bureau proxy (logistic on score + log tenure), logistic on the 10 behavioural features (standardised; no post-hoc calibration), **PBCM** (Platt-calibrated `HistGradientBoostingClassifier`), PBCM + bureau. AUC, KS, Brier, ECE, calibration slope/intercept, reliability bins with Wilson intervals; paired bootstrap of the logistic–GBM AUC difference; shuffled-label negative control; 52-week outcome as ranking check and as a separately trained model; in-time 5-fold | Table 3, Figs 2, 3, 5 |
| E2b | **Frozen operating policy**: approval threshold chosen on the validation cohort for an 8% missed-window budget, frozen, evaluated on the test cohort; the retrospective ranking frontier is reported separately | Table 3 |
| E3 | Robustness: bureau strength ρ, thin-file share, **feature measurement noise (crossover with the bureau proxy)**, missingness in the quality features, manipulation of three gameable features at the operating threshold | Table 4, Fig 4 |
| E4 | Selection: training on the 50% the bureau incumbent would approve vs a random-50% control | Section 6 |
| E5 | Pricing: tier inputs (expected unpaid share, expected funded term) estimated **under the offered terms** on out-of-fold training scores by fixed-point iteration; compared with the reference-product shortcut and a linear rule; realised margin by tier (equal- and capital-weighted) against a pre-declared ±1.0-point tolerance; paired no-shift cohort; common-shock and collection-base stresses; market-cap feasibility on [0.50, 1] | Tables 5–6, Fig 6 |
| E7 | Monitoring: velocity rule (Algorithm 2, 1-indexed weeks, watch from week 1), cumulative-progress and projected-payoff baselines; thresholds selected on validation by F1 and frozen; event = miss of the offer's own deadline; flags count only with ≥ 2 weeks' lead; workload metrics | Table 7, Fig 7 |
| E8 | Per-feature ablations (declared concentration, category index, both); approval parity by category and tenure at the frozen operating threshold and at a fixed 60% rate, with group sizes; permutation importance | Table 8 |
| E9 | Seeds 42–51: mean ± sd of every headline metric (`source_data/multiseed_runs.csv`) | all "10 seeds" entries |
| WX | The representative seller scored by the trained model: score, tier, sizing, tier price, principal reasons, prototype timing of inference and sampled Shapley values, a consistent repayment schedule with its cash-flow APR, and a stressed path evaluated at the frozen alert threshold | Appendix / Supplement |
| S | Deployment scenarios (genuine truncated normal, impaired unsold inventory, no-financing counterfactual, adverse variant) — assumption-driven illustrations | Supplement only |

`pricing_sensitivity.py` re-evaluates the tier price at multiples of the estimated
loss and term inputs and reports where the assumed market cap binds. `arch_figure.py`
draws Figure 1 (illustration of a proposal).

## Reproducing

```bash
pip install -r requirements.txt
python pbcm_simulation.py        # ~15 minutes on a 2-core machine
python pricing_sensitivity.py    # < 1 s (reads results.json)
python arch_figure.py            # Figure 1
```

Outputs: `results.json`, `worked_example.json`, `pricing_sensitivity.json`,
`fig1_architecture.png` … `fig8_scenario.png`, and `source_data/*.csv` (editable source
data for every figure and the per-seed table).

`results.reference.json` and `worked_example.reference.json` are the committed outputs
this package was verified against (Python 3.11.15, numpy 2.4.4, scikit-learn 1.8.0,
matplotlib 3.10.9). Tree learners and Platt scaling are deterministic given
`random_state`; the sampling-based Shapley values use a seeded generator. Timing fields
(`explanation_timing`) are machine-dependent and are excluded from the comparison:

```bash
python -c "import json;a=json.load(open('results.json'));b=json.load(open('results.reference.json'));print('MATCH' if a==b else 'DIFFERS')"
python -c "import json;a=json.load(open('worked_example.json'));b=json.load(open('worked_example.reference.json'));a.pop('explanation_timing');b.pop('explanation_timing');print('MATCH' if a==b else 'DIFFERS')"
```

Floating-point differences at the 1e-10 level can appear across Python minor versions
and BLAS builds; the paper reports values to three or four significant figures.

## Mapping results.json to the paper

| Paper | Key |
|---|---|
| Table 3 (primary seed) | `primary.bureau_proxy`, `.logit_behavioral`, `.pbcm_gbm`, `.pbcm_plus_bureau`, `.pbcm_gbm_in_time_cv`, `.label52`, `.calibration`, `.auc_diff_logit_minus_gbm`, `.frozen_policy`, `.selection`, `.shuffled_label_control_auc` |
| Table 3 (10 seeds) | `multiseed.*` |
| Table 4 | `robustness.*` |
| Tables 5–6 | `primary.pricing.tier_inputs`, `.tiers[rule]`, `.flatness`, `.portfolio_operating_book`, `.no_shift_paired`, `.stress`, `.cap`; `pricing_sensitivity.json` |
| Table 7 | `primary.monitoring.algorithm2`, `.baseline_cumulative_progress`, `.baseline_projected_payoff`, `.frozen`, `.a2_sweep` |
| Table 8 | `primary.parity`, `primary.ablation_auc`, `primary.permutation_importance_auc_drop` |
| Table 1 | `table1_apr.fr125`; worked example `draw.cash_flow_apr` |
| Worked example | `worked_example.json` |

## Reading the results correctly

1. **The learner is not the point.** On this DGP a standardised logistic model on the
   same ten features slightly exceeds gradient boosting (paired bootstrap interval reported).
   The advantage over the bureau baseline comes from the features.
2. **Loss rates are the DGP's.** Unpaid shares are small because a fixed holdback keeps
   collecting from slow payers; real MCA portfolios lose more. `pricing_sensitivity.py`
   shows where the assumed market cap binds as losses rise.
3. **Exposure does most of the risk adjustment.** Because the advance rate falls with the
   score, the expected unpaid share per dollar advanced is nearly flat across tiers under
   the offered terms, so the tier prices are close together. The reference-product shortcut
   ℓ = (1 − E)·L₀ over-prices low tiers relative to their realised loss.
4. **The test cohort is a simulated shift.** Realised margins below target on the test
   cohort reflect a 20% hazard increase that raises the realised missed-window rate by
   about one percentage point; the paired no-shift cohort isolates the effect.
5. **Parity across synthetic categories is a descriptive diagnostic.** Categories are not
   protected classes and the four-fifths statistic is a descriptive ratio here, not a
   compliance standard.

## Limitations of the artifact

* Fraud rings, multi-account sellers, seasonal categories and macro shocks correlated
  across sellers are not modelled beyond the deterministic common-shock stress.
* The bureau proxy is a single linear signal; real bureau data is richer.
* Sampling-based Shapley values approximate the exact tree algorithm; the measured runtime
  is a single-process prototype figure, not a benchmark.
* Selection is simulated by truncating on the incumbent score; real censoring also depends
  on the incumbent's unobserved criteria.
* No intervention is simulated: monitoring lead time is lead to a contractual deadline,
  not evidence that an intervention changes the outcome.

## Licence and citation

MIT. See `LICENSE` and `CITATION.cff`.

R. Verma, "PBCM underwriting simulation for embedded merchant cash advances (v1.1.0)," Zenodo, 2026. Concept DOI 10.5281/zenodo.22712490 (resolves to the latest version); v1.0.0 snapshot 10.5281/zenodo.22712491.
