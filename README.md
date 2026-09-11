# PBCM underwriting simulation — reproducibility package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22712491.svg)](https://doi.org/10.5281/zenodo.22712491)

Synthetic evaluation suite accompanying *"AI-Driven Underwriting for Embedded
Merchant Cash Advances: A Financial Platform Engineering Framework for
E-Commerce Marketplaces"* (v2, September 2026).

This package regenerates **every quantitative value in Sections 6–8 of the
paper** (Tables 4–9, Figures 2–8) and **every value in the API worked example**
(Appendix A), from a stated data-generating process and a fixed seed.

## Data statement

Fully synthetic. No platform, lender, customer, or company data is used,
referenced, or required. Every seller, transaction history, bureau score and
repayment path is generated at runtime from a seeded random number generator.
There are no input data files.

## What is validated — and what is not

The simulation validates the **underwriting, pricing and monitoring
mechanisms** of the paper under a data-generating process (DGP) that is stated
in full below and in Section 6.1 of the paper. Because the DGP is synthetic,
results are evidence that the mechanisms behave as described *when behavioural
features carry information about repayment*; they are not estimates of
real-world model performance, default rates, or loss rates. The paper says so
in Section 9 (Limitations), and the sensitivity experiments (E3) are there so a
reader can see how the conclusions move as the DGP's assumptions move.

## The data-generating process (E1)

For each of two origination cohorts of 12,000 sellers:

* A latent business-health variable *h* ~ N(0, 1) drives everything downstream.
* Category (6, equal weights) with a base return rate, a platform category-risk
  index, and a hazard multiplier (fashion 1.25 … industrial 0.85).
* Tenure ~ lognormal (median ≈ 20 months). A **13-week transaction history** is
  simulated from a base weekly GMV (lognormal; depends weakly on *h* and
  tenure), a linear trend (0.25 *h* + noise) and multiplicative weekly noise
  whose scale falls with *h*. The features GMV₉₀, momentum and revenue CV are
  **computed from that history**, not drawn directly.
* Return rate (category base × exp(−0.35 *h*)), within-category percentile,
  dispute rate, fulfilment score, repeat-buyer rate all depend on *h* with
  noise. Platform concentration has a true value (Beta(4, 2)) and a
  **seller-declared value that 30% of sellers over-report** by 0.10–0.35; the
  model sees the declared value.
* **Bureau-proxy score** = 680 + 70·(ρ *h* + √(1−ρ²) ε), with ρ = 0.55 for
  thick-file sellers and ρ = 0.15 for thin-file sellers (tenure < 12 months, or
  25% of the rest). ρ is swept in E3.
* **Post-origination path (52 weeks):** weekly GMV follows a lognormal random
  walk (drift 0.003 *h* + 0.02·trend − 0.002, σ = 0.06) with a permanent shock
  process — weekly hazard 0.004·exp(−0.9 *h*)·category multiplier·(1 − 0.3·true
  concentration); on a shock the seller's GMV is multiplied by a severity drawn
  from U(0.05, 0.60) for the rest of the horizon. The out-of-time test cohort
  uses hazard × 1.20 (tighter conditions) so the temporal test is a real one.
* **Labels.** Advances used to *generate* labels are policy-neutral:
  A₀ = 0.15·GMV₉₀ at factor 1.25 with a 15% holdback. Label 1 (the paper's
  definition): full repayment within 1.5× the estimated window. Because A₀ is
  proportional to GMV₉₀, the estimated window is a constant of the product
  terms (1.25 × 0.15 × 13 / 0.15 = 16.25 weeks → horizon 25 weeks) and does not
  vary with GMV, which is why the label is not mechanically a function of the
  size feature. Label 2 (robustness): full repayment within 52 weeks. The
  unpaid share of the obligation at 52 weeks is the loss-given-default.

## Experiments

| ID | What it does | Backs |
|---|---|---|
| E2 | Out-of-time comparison of four scorers: bureau-proxy (logistic on score + log tenure), logistic on the 10 behavioural features, **PBCM** (Platt-calibrated gradient boosting on the 10 features), PBCM + bureau. AUC, KS, Brier, ECE, bad rate at 60% approval, approval rate at an 8% bad-rate budget, thin-/thick-file AUC; in-time 5-fold for comparison; label-2 robustness | Table 4, Figs 2, 3, 5 |
| E3 | Sweeps of bureau signal strength ρ (0.30–0.85) and thin-file share | Table 5, Fig 4 |
| E4 | Reject inference: PBCM trained only on the 50% of sellers the bureau incumbent would have approved | Table 4 (last row) |
| E5 | Expected-loss factor rate FR\*(E) vs the linear rule; realised lender margin by tier under both; eligibility floor implied by a market cap; calibration bins | Tables 6–7, Fig 6 |
| E6 | Capital-deployment scenario as a distribution over margin and sell-through; break-even share of growth deployments | Table 8, Fig 7 |
| E7 | Algorithm 2 (repayment-velocity flags): precision, recall, false-alarm rate, lead time, by threshold | Table 9, Fig 8 |
| E8 | Ablation dropping the seller-declared and category features; approval-rate parity (adverse-impact ratio) by category and tenure group; permutation importance | Section 6.8 (text) |
| E9 | Seeds 42–51: mean ± sd of every headline metric | Table 4 "10 seeds" column |
| WX | The Section 5 seller scored by the trained model: E(s), sizing, pricing, Shapley explanation, a consistent repayment schedule, and a stressed path through Algorithm 2 | Appendix A |

`pricing_sensitivity.py` evaluates the closed-form FR\*(E) at other LGD and
term values (Table 6, right-hand block). It runs no simulation.

## Reproducing

```bash
pip install -r requirements.txt
python pbcm_simulation.py        # ~11 minutes (18 when the machine is busy)
python pricing_sensitivity.py    # < 1 s
python arch_figure.py            # Figure 1 (illustration only)
```

Outputs: `results.json`, `worked_example.json`, `pricing_sensitivity.json`,
`fig1_architecture.png` … `fig8_monitoring.png`.

To confirm the committed values regenerate:

```bash
python -c "import json;a=json.load(open('results.json'));b=json.load(open('results.reference.json'));print('MATCH' if a==b else 'DIFFERS')"
```

`results.reference.json` and `worked_example.reference.json` are the committed
outputs this package was verified against (Python 3.11, numpy 2.4.4,
scikit-learn 1.8.0). Tree-based learners and Platt scaling are deterministic
given `random_state`; the sampling-based Shapley values in the worked example
use a seeded generator. Runtime is dominated by the ten-seed sweep.

## Mapping results.json to the paper

| Paper | Key |
|---|---|
| Table 4 (primary seed) | `primary.bureau_proxy`, `.logit_behavioral`, `.pbcm_gbm`, `.pbcm_plus_bureau`, `.pbcm_gbm_in_time_cv`, `.reject_inference`, `.label52_auc` |
| Table 4 (10 seeds) | `multiseed.*` |
| Table 5 | `sensitivity.rho_thick_sweep`, `.thin_share_sweep` |
| Table 6 | `primary.pricing.tiers[*].fr_star_*`, `pricing_sensitivity.json` |
| Table 7 | `primary.pricing.tiers[*]` (bad rates, realised margins), `primary.pricing.portfolio_*` |
| Table 8 | `scenario_at_worked_example_fr`, `scenario_fr121` |
| Table A1 | `worked_example.json` → `draw.schedule` |
| Table 9 | `primary.monitoring.thr_*` |
| Section 6.8 | `primary.parity`, `primary.ablation_drop_declared_and_category`, `primary.permutation_importance_auc_drop` |
| Appendix A | `worked_example.json` |

## Reading the results correctly

1. **The learner is not the point.** On this DGP a calibrated logistic model on
   the same ten features matches or slightly exceeds gradient boosting
   (Table 4). The advantage over the bureau baseline comes from the features.
   Gradient boosting is retained for the interaction structure real platform
   data exhibits, which this DGP deliberately omits.
2. **The loss rates are the DGP's.** Loss-given-default is ≈ 0.09 of the
   obligation because a fixed holdback keeps collecting from slow payers; real
   MCA portfolios lose more. That is why the derived factor rates (1.10–1.16)
   sit below the industry range, and why Table 6 reports FR\* at LGD 0.20,
   0.35 and 0.50 as well.
3. **Out-of-time calibration is imperfect by construction.** The test cohort
   has 20% higher shock hazard than the training cohort. Realised lender margin
   under expected-loss pricing is therefore ≈ 4.4%, below the 5% target — this
   is the cost of drift, and the paper reports it rather than hiding it.
4. **Approval-rate parity across synthetic categories is a proxy exercise.**
   Categories are not protected classes. The experiment shows the mechanism
   (which features move the adverse-impact ratio) and nothing about any real
   population.

## Limitations of the artifact

* Anomalous cases a real platform sees — fraud rings, multi-account sellers,
  seasonal categories, macro shocks correlated across sellers — are not
  modelled. Shocks here are independent across sellers.
* The bureau proxy is a single linear signal; real bureau data is richer,
  which is why E3 sweeps its strength up to ρ = 0.85.
* Sampling-based Shapley values (Štrumbelj & Kononenko, 2014) approximate
  TreeSHAP; a production system would use the exact tree algorithm.
* Reject inference is simulated by truncating on the incumbent score; real
  reject inference must also handle the incumbent's own selection on
  unobservables.

## Licence and citation

MIT. See `LICENSE`.

R. Verma, "PBCM underwriting simulation for embedded merchant cash advances (v1.0.0)," Zenodo, 2026. doi:10.5281/zenodo.22712491 (concept DOI for all versions: 10.5281/zenodo.22712490). Machine-readable metadata in `CITATION.cff`.
