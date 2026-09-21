# Changelog

## v1.1.0 — September 2026 (manuscript version 3)

Data-generating process unchanged; evaluation rebuilt. Each item maps to a finding of the
independent referee report on manuscript version 2.

* **Labels.** Prediction target (reference product P0) and policy outcomes (the advance
  actually offered) are separated everywhere; the mixed tier table was removed.
* **Pricing.** Tier inputs (expected unpaid share, expected funded term) estimated under the
  offered terms on out-of-fold training scores by fixed-point iteration; single fee waterfall
  in identity and simulator (platform share of the collected fee; funding on the full
  advance for the realised term); pre-declared flatness tolerance; capital-weighted and
  equal-weighted margins; market cap tested on [0.50, 1] with an explicit infeasibility
  outcome; "cross-subsidy" language dropped.
* **Frozen policies.** Approval threshold (8% budget) and alert thresholds selected on a new
  validation cohort, frozen, evaluated on the test cohort across ten seeds; the retrospective
  frontier is reported separately.
* **Monitoring.** 1-indexed weeks; flags count only with ≥ 2 weeks' lead before the offer's
  own deadline; watch can fire from week 1; cumulative-progress and projected-payoff
  baselines; workload metrics; weekly windows and a clock-based missing-payout check in the
  design.
* **Cohort shift.** Paired no-shift test cohort; realised missed-window rate shift reported.
* **Robustness.** Feature-noise crossover, missingness, manipulation, common-shock and
  collection-base stresses.
* **Underwriting.** Paired bootstrap for logistic vs GBM; calibration slope/intercept and
  Wilson intervals; separately trained 52-week model; shuffled-label control;
  random-50% control for the selection experiment; per-feature ablations; parity at the
  operating threshold with group sizes.
* **Table 1.** Discrete weekly schedule, cash-flow IRR, fee annualisation labelled as such.
* **Scenarios.** Genuine truncated normal, impaired unsold inventory, no-financing
  counterfactual, adverse variant; supplement only.
* **Worked example.** Tier price; principal reasons separated from suggestions; prototype
  timing of inference and sampled Shapley values; stressed path at the frozen threshold.
* Figure source data written to `source_data/`.

## v1.0.0 — 11 September 2026 (manuscript version 2)

Initial release. Zenodo 10.5281/zenodo.22712491.
