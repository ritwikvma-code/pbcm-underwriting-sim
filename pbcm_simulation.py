"""
PBCM — Platform Behavioral Credit Model: policy-consistent synthetic evaluation suite
=====================================================================================
Artifact v1.1.0. Accompanies "Underwriting embedded merchant cash advances from
marketplace settlement data: a design blueprint with a policy-consistent synthetic
evaluation" (manuscript version 3, September 2026).

Everything here is synthetic. No platform, lender, customer, or company data is used
or required. The data-generating process (DGP) is unchanged from artifact v1.0.0, so
the underwriting comparison regenerates identically; the EVALUATION was rebuilt so that
every reported outcome is measured under the terms actually offered to the seller.

What changed from v1.0.0 (each item maps to a referee finding on manuscript v2)
  * Two label families are kept apart everywhere: the PREDICTION TARGET (repayment of a
    fixed reference product P0 within 1.5x its estimated window) and POLICY OUTCOMES
    (repayment of the advance actually offered — its own amount, price, holdback and
    deadline). Tables that mixed the two were removed.
  * Pricing inputs (expected unpaid share of the obligation and expected funded term)
    are estimated by risk tier UNDER THE OFFERED TERMS on out-of-fold training scores,
    by fixed-point iteration; the reference-product shortcut l = (1-E)*L0 is kept only
    as the comparison that shows why it fails. One fee waterfall is used in both the
    pricing identity and the simulator (platform paid a share of the COLLECTED fee;
    funding cost accrues on the full advance for the realised term). Flatness is judged
    against a pre-declared tolerance; capital-weighted and equal-weighted results are
    both reported; the market cap is tested on the feasible domain [0.50, 1] only.
  * Approval and alert thresholds are selected on a separate VALIDATION cohort, frozen,
    and then evaluated on the test cohort (frozen operating policy). The retrospective
    ranking frontier is reported separately and labelled as such.
  * Monitoring: calendar weeks are 1-indexed; a flag counts only if it arrives at least
    MIN_LEAD_WEEKS before the offer's own deadline; the event anticipated is a miss of
    that deadline under the funded terms; 'watch' can fire from the first event; two
    simple baselines (cumulative progress, projected payoff) are evaluated alongside.
  * Paired no-drift test cohort (identical draws, hazard multiplier 1.0) isolates the
    hazard-shift effect; realised default-rate shift is reported, not the hazard shift.
  * Robustness of the behavioural advantage: feature-noise sweep (crossover with the
    bureau proxy), missingness sweep, manipulation test, common-shock stress, and a
    collection-base shortfall — in addition to the bureau-strength sweeps.
  * Reject-inference experiment gains a random-50% control; ablations are per feature;
    parity is reported at the frozen operating threshold with group sizes; a 52-week
    model is trained separately; a shuffled-label negative control is included.
  * Table 1 arithmetic: discrete weekly schedule, cash-flow IRR, fee annualisation
    labelled as such; funding cost on the discrete term.
  * Deployment scenarios use a genuine truncated normal, impair unsold inventory, and
    are reported relative to a no-financing counterfactual as assumption-driven
    illustrations (supplement only).

Run:  python pbcm_simulation.py          (~15-25 minutes on a 2-core laptop)
Outputs: results.json, worked_example.json, figures fig2..fig8, source-data CSVs.
"""

import csv
import json
import math
import os
import time
from datetime import date, timedelta

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ARTIFACT_VERSION = "1.1.0"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42
SEEDS = list(range(42, 52))

# ----------------------------------------------------------------------
# Product terms and DGP constants (unchanged from v1.0.0)
# ----------------------------------------------------------------------
N_SELLERS = int(os.environ.get("PBCM_N_SELLERS", 12_000))   # per cohort (training, test, validation); env override is for smoke tests only
WEEKS_HIST = 13               # 90-day feature window
WEEKS_POST = 52               # post-origination horizon (accounting horizon; unpaid at 52 wk = write-off)
HOLDBACK = 0.15               # fixed holdback rate
ALPHA_LABEL = 0.15            # reference product P0: advance rate used to generate the prediction target
FR_LABEL = 1.25               # reference product P0: factor rate
LABEL_MULT = 1.5              # "full repayment within 1.5x estimated window"
OOT_HAZARD_SHIFT = 1.20       # test cohort hazard multiplier (simulated cohort shift)

CATEGORIES = {
    # name: (base return rate, platform category risk index, hazard multiplier)
    "electronics":  (0.10, 0.30, 1.00),
    "fashion":      (0.28, 0.55, 1.25),
    "home_garden":  (0.08, 0.35, 1.00),
    "collectibles": (0.05, 0.40, 1.10),
    "industrial":   (0.03, 0.25, 0.85),
    "media":        (0.06, 0.30, 0.95),
}
CAT_NAMES = list(CATEGORIES)

FEATURES = ["gmv_90d", "gmv_momentum", "revenue_cv", "return_rate_cohort_pct",
            "dispute_rate", "fulfillment_score", "tenure_months",
            "category_risk_index", "repeat_buyer_rate", "platform_concentration"]
QUALITY_FEATURES = ["return_rate_cohort_pct", "dispute_rate", "fulfillment_score", "repeat_buyer_rate"]

RHO_THICK = 0.55              # bureau-score correlation with latent health, thick file
RHO_THIN = 0.15               # ... thin file
THIN_EXTRA_SHARE = 0.25       # sellers with tenure >= 12 who are still thin-file

# Pricing parameters — ILLUSTRATIVE assumptions, not observed industry terms
COST_OF_FUNDS = 0.08          # annualised lender funding cost, accrues on the full advance for the realised term
PLATFORM_SHARE = 0.18         # platform share of the COLLECTED fee (the single waterfall used everywhere)
TARGET_MARGIN = 0.05          # lender target net margin on principal
FR_MARKET_CAP = 1.35          # ceiling the market will bear (assumption)
FR_BASE_LIN, GAMMA_LIN = 1.15, 0.30   # a judgemental linear rule, kept as the comparison
TIERS = [(0.90, 1.001, "very_low"), (0.80, 0.90, "low"), (0.70, 0.80, "moderate"),
         (0.60, 0.70, "elevated"), (0.50, 0.60, "high")]
E_FLOOR_SIZING = 0.50         # sizing rule alpha(E) is defined on [0.50, 1]

# Pre-declared evaluation policy
BAD_BUDGET = 0.08             # lender's budget: missed-window rate on the prediction target
FLAT_TOL = 0.010              # margin-flatness tolerance: |tier margin - target| <= 1.0 pt for tiers with n >= FLAT_MIN_N
FLAT_MIN_N = 300
MIN_LEAD_WEEKS = 2            # a flag must arrive at least this many weeks before the deadline to count
THR_GRID_A2 = [round(x, 2) for x in np.arange(0.30, 0.951, 0.05)]
THR_GRID_CUM = [round(x, 2) for x in np.arange(0.30, 0.951, 0.05)]
THR_GRID_PROJ = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ----------------------------------------------------------------------
# E1. Population and repayment data-generating process (UNCHANGED from v1.0.0)
# ----------------------------------------------------------------------
def gen_population(rng, n=N_SELLERS, hazard_scale=1.0, rho_thick=RHO_THICK,
                   rho_thin=RHO_THIN, thin_extra=THIN_EXTRA_SHARE):
    """Generate n sellers with a latent business-health variable h, the ten
    PBCM features derived from simulated 13-week transaction histories, a
    bureau-proxy score, and a 52-week post-origination GMV path."""
    h = rng.normal(0, 1, n)
    cat_idx = rng.integers(len(CAT_NAMES), size=n)
    cat_ret = np.array([CATEGORIES[c][0] for c in CAT_NAMES])[cat_idx]
    cat_risk = np.array([CATEGORIES[c][1] for c in CAT_NAMES])[cat_idx]
    cat_haz = np.array([CATEGORIES[c][2] for c in CAT_NAMES])[cat_idx]

    tenure = np.clip(np.exp(rng.normal(3.0, 0.8, n)), 1, 120)
    base_week = np.exp(rng.normal(6.75 + 0.15 * h + 0.10 * np.log(tenure), 0.85, n))
    base_week = np.clip(base_week, 120, 40_000)
    trend = np.clip(0.25 * h + rng.normal(0, 0.20, n), -0.6, 0.8)
    sigma_w = 0.12 + 0.15 * sigmoid(-h)
    t = np.arange(WEEKS_HIST)
    hist = (base_week[:, None] * (1 + trend[:, None] * (t[None, :] - 6) / WEEKS_HIST)
            * np.exp(rng.normal(0, 1, (n, WEEKS_HIST)) * sigma_w[:, None]))
    hist = np.maximum(hist, 1.0)
    gmv90 = hist.sum(1)
    gmv30 = hist[:, -4:].sum(1) * (30 / 28)
    momentum = (3 * gmv30 - gmv90) / gmv90
    rev_cv = hist.std(1, ddof=1) / hist.mean(1)

    ret_rate = cat_ret * np.exp(-0.35 * h + rng.normal(0, 0.30, n))
    ret_pct = np.zeros(n)
    for c in range(len(CAT_NAMES)):
        m = cat_idx == c
        r = ret_rate[m]
        order = r.argsort().argsort()
        ret_pct[m] = 100 * (1 - order / max(m.sum() - 1, 1))
    fulfil = np.clip(sigmoid(2.0 + 0.9 * h + rng.normal(0, 0.6, n)), 0.5, 1.0)
    dispute = np.clip(0.006 * np.exp(-0.6 * h + rng.normal(0, 0.5, n)) + 0.01 * (1 - fulfil), 0, 0.2)
    repeat = np.clip(0.15 + 0.06 * h + rng.normal(0, 0.05, n), 0.01, 0.8)
    conc_true = rng.beta(4, 2, n)
    misreport = rng.random(n) < 0.30
    conc_decl = np.clip(conc_true + misreport * rng.uniform(0.10, 0.35, n), 0, 1)

    thin = (tenure < 12) | (rng.random(n) < thin_extra)
    rho = np.where(thin, rho_thin, rho_thick)
    bureau = 680 + 70 * (rho * h + np.sqrt(1 - rho ** 2) * rng.normal(0, 1, n))

    start_week = gmv90 / WEEKS_HIST * (1 + trend / 2)
    mu = 0.003 * h + 0.02 * trend - 0.002
    eps = rng.normal(0, 1, (n, WEEKS_POST))
    logpath = np.cumsum(mu[:, None] + 0.06 * eps, axis=1)
    post = start_week[:, None] * np.exp(logpath)
    hazard = hazard_scale * 0.004 * np.exp(-0.9 * h) * cat_haz * (1 - 0.3 * conc_true)
    shock_draw = rng.random((n, WEEKS_POST)) < hazard[:, None]
    shock_week = np.where(shock_draw.any(1), shock_draw.argmax(1), -1)
    severity = rng.uniform(0.05, 0.60, n)
    shocked = shock_week >= 0
    wk = np.arange(WEEKS_POST)
    mult = np.where((shock_week[:, None] >= 0) & (wk[None, :] >= shock_week[:, None]),
                    severity[:, None], 1.0)
    post = post * mult

    X = np.column_stack([gmv90, momentum, rev_cv, ret_pct, dispute, fulfil,
                         tenure, cat_risk, repeat, conc_decl])
    return dict(X=X, h=h, cat=cat_idx, tenure=tenure, thin=thin, bureau=bureau,
                gmv90=gmv90, post=post, shock_week=shock_week, severity=severity,
                shocked=shocked, conc_true=conc_true, misreport=misreport)


def paired_no_shift_cohort(seed):
    """The test cohort regenerated with hazard multiplier 1.0 from the SAME random draws
    (features identical; the shocked set is a subset). Used to isolate the hazard shift."""
    rng = np.random.default_rng(seed)
    gen_population(rng)                              # consume the training draws
    return gen_population(rng, hazard_scale=1.0)


def repay_path(post, principal, factor):
    """Cumulative repayment (capped at the obligation) under the fixed holdback."""
    obligation = principal * factor
    cum = np.cumsum(HOLDBACK * post, axis=1)
    return np.minimum(cum, obligation[:, None]), obligation


def term_weeks(cum, T):
    full = cum >= T[:, None] - 1e-9
    return np.where(full.any(1), full.argmax(1) + 1, WEEKS_POST)


def make_labels(pop):
    """PREDICTION TARGET: repayment of the fixed reference product P0
    (A0 = 0.15*GMV90, FR 1.25, holdback 15%) within 1.5x its estimated window."""
    A0 = ALPHA_LABEL * pop["gmv90"]
    cum, T = repay_path(pop["post"], A0, FR_LABEL)
    est_window = FR_LABEL * ALPHA_LABEL * WEEKS_HIST / HOLDBACK          # 16.25 weeks (constant of P0)
    horizon = int(math.ceil(LABEL_MULT * est_window))                    # 25 weeks
    y_window = (cum[:, horizon - 1] >= T - 1e-9).astype(int)
    y_52 = (cum[:, -1] >= T - 1e-9).astype(int)
    unpaid = 1 - cum[:, -1] / T
    return dict(y_window=y_window, y_52=y_52, unpaid_share=unpaid, est_window=est_window,
                horizon=horizon, term=term_weeks(cum, T))


# ----------------------------------------------------------------------
# Feature-quality perturbations (robustness experiments)
# ----------------------------------------------------------------------
def add_feature_noise(X, kappa, rng):
    """Measurement noise: each feature gets N(0, (kappa*sd)^2) noise (stale/noisy feature store)."""
    if kappa == 0:
        return X
    sd = X.std(0, ddof=1)
    return X + kappa * sd[None, :] * rng.normal(0, 1, X.shape)


def apply_missingness(X, q, rng):
    """Missing-at-random blanks in the four quality features (NaN, handled natively by the learner)."""
    if q == 0:
        return X
    Xm = X.copy()
    cols = [FEATURES.index(f) for f in QUALITY_FEATURES]
    mask = rng.random((len(X), len(cols))) < q
    for k, j in enumerate(cols):
        Xm[mask[:, k], j] = np.nan
    return Xm


def manipulate(X, share, rng):
    """A share of applicants game the three features a seller can move quickly:
    momentum +0.15 (a discount-driven late-window spike), GMV90 x1.10, declared concentration = 1."""
    Xm = X.copy()
    who = rng.random(len(X)) < share
    Xm[who, FEATURES.index("gmv_momentum")] += 0.15
    Xm[who, FEATURES.index("gmv_90d")] *= 1.10
    Xm[who, FEATURES.index("platform_concentration")] = 1.0
    return Xm, who


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def ks_stat(y, p):
    order = np.argsort(p)
    y = y[order]
    P, N = y.sum(), len(y) - y.sum()
    return float(np.max(np.abs(np.cumsum(y) / P - np.cumsum(1 - y) / N)))


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] + (1e-12 if i == bins - 1 else 0))
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    hw = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (float(c - hw), float(c + hw))


def calib_bins(y, p, bins=10, min_n=60):
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] + (1e-12 if i == bins - 1 else 0))
        if m.sum():
            lo, hi = wilson(int(y[m].sum()), int(m.sum()))
            rows.append(dict(lo=float(edges[i]), hi=float(edges[i + 1]), n=int(m.sum()),
                             mean_pred=float(p[m].mean()), observed=float(y[m].mean()),
                             ci95_lo=lo, ci95_hi=hi, supported=bool(m.sum() >= min_n)))
    return rows


def calibration_slope_intercept(y, p):
    """Cox recalibration: logit(y) ~ a + b*logit(p). Perfect calibration: a = 0, b = 1."""
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(pc / (1 - pc))[:, None]
    lr = LogisticRegression(C=1e6, max_iter=5000).fit(z, y)
    return dict(slope=float(lr.coef_[0, 0]), intercept=float(lr.intercept_[0]))


def retrospective_frontier(y, p, approve_rate=0.60, bad_budget=BAD_BUDGET):
    """Descriptive ranking frontier computed WITH test outcomes (not an operating policy)."""
    n = len(y)
    order = np.argsort(-p)
    k = int(approve_rate * n)
    bad_at_rate = 1 - y[order[:k]].mean()
    cum_bad = np.cumsum(1 - y[order]) / np.arange(1, n + 1)
    ok = np.where(cum_bad <= bad_budget)[0]
    appr_at_budget = (ok.max() + 1) / n if len(ok) else 0.0
    return float(bad_at_rate), float(appr_at_budget)


def evaluate(y, p, thin=None):
    out = dict(auc=float(roc_auc_score(y, p)), ks=ks_stat(y, p),
               brier=float(brier_score_loss(y, p)), ece=ece(y, p))
    b, a = retrospective_frontier(y, p)
    out["bad_rate_at_60pct_approval"] = b
    out["retrospective_approval_at_8pct_budget"] = a
    if thin is not None:
        out["auc_thin_file"] = float(roc_auc_score(y[thin], p[thin]))
        out["auc_thick_file"] = float(roc_auc_score(y[~thin], p[~thin]))
    return out


def select_budget_threshold(y_val, p_val, bad_budget=BAD_BUDGET):
    """Frozen operating threshold: largest approval set on the VALIDATION cohort whose
    cumulative missed-window rate (prediction target) stays within the budget."""
    order = np.argsort(-p_val)
    cum_bad = np.cumsum(1 - y_val[order]) / np.arange(1, len(p_val) + 1)
    ok = np.where(cum_bad <= bad_budget)[0]
    if not len(ok):
        return 1.01, 0
    k = int(ok.max() + 1)
    return float(p_val[order[k - 1]]), k


def paired_bootstrap_auc_diff(y, p1, p2, reps=500, seed=1291):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(reps):
        idx = rng.integers(0, len(y), len(y))
        d.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx]))
    d = np.array(d)
    return dict(point=float(roc_auc_score(y, p1) - roc_auc_score(y, p2)),
                ci95=[float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))], replicates=reps)


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
def gbm(seed):
    """scikit-learn HistGradientBoostingClassifier wrapped in Platt (sigmoid) calibration, 5-fold."""
    base = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                          max_leaf_nodes=31, l2_regularization=1.0,
                                          random_state=seed)
    return CalibratedClassifierCV(base, method="sigmoid", cv=5)


def logit():
    """Standardised logistic regression. No post-hoc calibration wrapper: its output is a
    fitted probability already."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def fit_predict(model, Xtr, ytr, Xte):
    model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1]


def bureau_X(pop):
    return np.column_stack([pop["bureau"], np.log(pop["tenure"])])


# ----------------------------------------------------------------------
# E5. Sizing, pricing and policy-consistent outcomes
# ----------------------------------------------------------------------
def alpha_lin(E):
    """Advance rate: 10% of GMV90 at E = 0.50 rising to 25% at E = 1 (exposure policy, not a price)."""
    return 0.10 + 0.15 * (E - 0.50) / 0.50


def fr_lin(E):
    return FR_BASE_LIN + GAMMA_LIN * (1 - E)


def fr_star(ell, tau):
    """Expected-cash-flow identity under the single waterfall.
    Lender cash-in A*FR*(1-ell); platform paid s * (collected fee) ~ s*(A*FR*(1-ell) - A);
    funding cost A*c*tau/52 on the full advance; requirement net = m*A.  Solving:
        FR* = (1 + c*tau/52 + m - s) / ((1 - s) * (1 - ell))."""
    return (1 + COST_OF_FUNDS * np.asarray(tau) / 52 + TARGET_MARGIN - PLATFORM_SHARE) / \
           ((1 - PLATFORM_SHARE) * (1 - np.asarray(ell)))


def tier_index(E):
    """0..4 for E in [0.5, 1]; -1 below the sizing floor."""
    t = np.full(len(E), -1)
    for i, (lo, hi, _) in enumerate(TIERS):
        t[(E >= lo) & (E < hi)] = i
    return t


def own_deadline(FR, alpha):
    """Offer-specific deadline in calendar weeks: ceil(1.5 x estimated repayment weeks)."""
    return np.ceil(LABEL_MULT * FR * alpha * WEEKS_HIST / HOLDBACK).astype(int)


def realised_outcomes(post, gmv90, E, FR):
    """Everything about the advance ACTUALLY offered: amount, price, own deadline, repayment,
    unpaid share at 52 weeks, realised term, and realised lender margin under the single waterfall."""
    FR = np.where(np.isfinite(FR), FR, 1.0)              # rows below the sizing floor are never funded; placeholder keeps arithmetic finite
    alpha = alpha_lin(np.clip(E, E_FLOOR_SIZING, 1.0))
    A = alpha * gmv90
    cum, T = repay_path(post, A, FR)
    D = own_deadline(FR, alpha)
    idx = np.arange(len(E))
    repaid_by_deadline = cum[idx, np.clip(D, 1, WEEKS_POST) - 1] >= T - 1e-9
    repaid = cum[:, -1]
    unpaid_share = 1 - repaid / T
    term = term_weeks(cum, T)
    fee_collected = np.maximum(repaid - A, 0)
    platform_take = PLATFORM_SHARE * fee_collected
    funding = A * COST_OF_FUNDS * term / 52
    net = repaid - A - funding - platform_take
    return dict(A=A, FR=FR, alpha=alpha, D=D, T=T, cum=cum, repaid=repaid, unpaid_share=unpaid_share,
                term=term, net_margin=net / A, missed_own_window=~repaid_by_deadline,
                unpaid_any=unpaid_share > 1e-9, platform_take=platform_take, funding=funding)


def fit_tier_pricing(train, p_oof, max_iter=12, tol=1e-5):
    """Estimate per-tier expected unpaid share ell_t and expected funded term tau_t UNDER THE
    OFFERED TERMS on out-of-fold training scores, by fixed-point iteration on FR_t."""
    t = tier_index(p_oof)
    FR_t = np.full(len(TIERS), 1.15)
    hist = []
    for it in range(max_iter):
        FR = np.where(t >= 0, FR_t[np.clip(t, 0, None)], np.nan)
        m = t >= 0
        out = realised_outcomes(train["post"][m], train["gmv90"][m], p_oof[m], FR[m])
        ell = np.array([out["unpaid_share"][t[m] == i].mean() for i in range(len(TIERS))])
        tau = np.array([out["term"][t[m] == i].mean() for i in range(len(TIERS))])
        FR_new = fr_star(ell, tau)
        delta = float(np.max(np.abs(FR_new - FR_t)))
        hist.append(dict(iteration=it + 1, max_abs_change=delta, FR=[round(float(v), 5) for v in FR_new]))
        FR_t = FR_new
        if delta < tol:
            break
    # Jensen / waterfall approximation error on training data (platform paid on collected fee per seller)
    FR = FR_t[np.clip(t, 0, None)]
    out = realised_outcomes(train["post"][m], train["gmv90"][m], p_oof[m], FR[m])
    approx_err = []
    for i in range(len(TIERS)):
        mm = t[m] == i
        exact = out["platform_take"][mm].sum() / out["A"][mm].sum()
        lin = PLATFORM_SHARE * (out["repaid"][mm].sum() - out["A"][mm].sum()) / out["A"][mm].sum()
        approx_err.append(float(exact - lin))
    return dict(ell=[float(v) for v in ell], tau=[float(v) for v in tau], FR=[float(v) for v in FR_t],
                iterations=len(hist), converged=bool(delta < tol), history=hist,
                platform_take_approx_error=approx_err,
                train_margin_by_tier=[float(out["net_margin"][t[m] == i].mean()) for i in range(len(TIERS))])


def reference_shortcut_inputs(train, ltr):
    """The shortcut a lender might take: L0 = mean unpaid share among reference-product window-missers,
    tau0 = mean reference-product term; then ell(E) = (1 - E) * L0."""
    miss = ltr["y_window"] == 0
    L0 = float(ltr["unpaid_share"][miss].mean())
    tau0 = float(ltr["term"].mean())
    return L0, tau0


def fr_shortcut(E, L0, tau0):
    return fr_star((1 - E) * L0, tau0)


def cap_feasibility(FR_by_tier, cap=FR_MARKET_CAP):
    """Lowest tier whose price is at or below the cap, on the feasible domain [0.50, 1] only."""
    ok = [i for i, fr in enumerate(FR_by_tier) if fr <= cap]
    if len(ok) == len(TIERS):
        return dict(binds=False, lowest_feasible_tier=TIERS[-1][2], e_min=E_FLOOR_SIZING, infeasible_tiers=[])
    if not ok:
        return dict(binds=True, lowest_feasible_tier=None, e_min=None, infeasible_tiers=[t[2] for t in TIERS])
    j = max(ok)
    return dict(binds=True, lowest_feasible_tier=TIERS[j][2], e_min=TIERS[j][0],
                infeasible_tiers=[TIERS[i][2] for i in range(len(TIERS)) if i not in ok])


def tier_table(out, t, yref=None):
    rows = []
    for i, (lo, hi, name) in enumerate(TIERS):
        m = t == i
        if m.sum() == 0:
            rows.append(dict(band=f"{lo:.2f}-{min(hi, 1.0):.2f}", tier=name, n=0))
            continue
        row = dict(band=f"{lo:.2f}-{min(hi, 1.0):.2f}", tier=name, n=int(m.sum()),
                   alpha_lo=float(alpha_lin(lo)), alpha_hi=float(alpha_lin(min(hi, 1.0))),
                   fr=float(out["FR"][m].mean()), fr_min=float(out["FR"][m].min()), fr_max=float(out["FR"][m].max()),
                   deadline_min=int(out["D"][m].min()), deadline_max=int(out["D"][m].max()),
                   missed_own_window=float(out["missed_own_window"][m].mean()),
                   unpaid_any_52w=float(out["unpaid_any"][m].mean()),
                   unpaid_share_52w=float(out["unpaid_share"][m].mean()),
                   mean_term_weeks=float(out["term"][m].mean()),
                   margin_equal=float(out["net_margin"][m].mean()),
                   margin_capital=float(np.average(out["net_margin"][m], weights=out["A"][m])),
                   margin_sd=float(out["net_margin"][m].std(ddof=1)),
                   mean_advance=float(out["A"][m].mean()))
        if yref is not None:
            row["reference_target_miss_rate"] = float(1 - yref[m].mean())
        rows.append(row)
    return rows


def flatness(rows, target=TARGET_MARGIN, tol=FLAT_TOL, min_n=FLAT_MIN_N):
    dev = [(r["tier"], r["margin_equal"] - target) for r in rows if r.get("n", 0) >= min_n]
    if not dev:
        return dict(pass_=False, max_abs_dev=None, spread=None, tiers_tested=0)
    vals = [d for _, d in dev]
    return dict(pass_=bool(max(abs(v) for v in vals) <= tol), max_abs_dev=float(max(abs(v) for v in vals)),
                spread=float(max(vals) - min(vals)), tiers_tested=len(dev), tolerance=tol,
                deviations={k: float(v) for k, v in dev})


def portfolio(out, m):
    return dict(n=int(m.sum()), margin_equal=float(out["net_margin"][m].mean()),
                margin_capital=float(np.average(out["net_margin"][m], weights=out["A"][m])),
                missed_own_window=float(out["missed_own_window"][m].mean()),
                unpaid_any_52w=float(out["unpaid_any"][m].mean()),
                unpaid_share_52w=float(out["unpaid_share"][m].mean()),
                capital_deployed=float(out["A"][m].sum()), mean_term_weeks=float(out["term"][m].mean()))


# ----------------------------------------------------------------------
# E7. Monitoring (Algorithm 2, corrected) and two baselines
# ----------------------------------------------------------------------
def algorithm2_flags(post, gmv90, principal, factor, thr):
    """Repayment-velocity rule evaluated on weekly settlement windows.
    velocity_w = holdback collected in week w / expected weekly holdback fixed at origination.
    'watch'  : any of the last (up to three) weekly velocities below thr — can fire from week 1.
    'review' : three consecutive weekly velocities below thr (first possible: week 3).
    'escalate': review with more than 80% of principal still outstanding.
    Returns 1-INDEXED calendar weeks (0 = never)."""
    n, W = post.shape
    expected = HOLDBACK * gmv90 / WEEKS_HIST
    hold = HOLDBACK * post
    cum = np.cumsum(hold, axis=1)
    T = principal * factor
    outstanding = np.maximum(T[:, None] - cum, 0)
    vel = hold / expected[:, None]
    active = np.ones_like(cum, dtype=bool)
    active[:, 1:] = cum[:, :-1] < T[:, None] - 1e-9     # open at the start of the week
    first_rev = np.zeros(n, int)
    first_watch = np.zeros(n, int)
    for w in range(W):
        win = vel[:, max(0, w - 2):w + 1]
        slow_any = (win < thr).any(1) & active[:, w]
        wt = slow_any & (first_watch == 0)
        first_watch[wt] = w + 1
        if w >= 2:
            slow_all = (win < thr).all(1) & active[:, w]
            rv = slow_all & (first_rev == 0)
            first_rev[rv] = w + 1
    esc = np.zeros(n, bool)
    m = first_rev > 0
    esc[m] = outstanding[m, first_rev[m] - 1] > 0.80 * principal[m]
    return first_rev, first_watch, esc


def cumulative_progress_flags(post, gmv90, principal, factor, thr):
    """Baseline 1: flag when cumulative holdback collected falls below thr x the cumulative
    holdback expected at origination (first possible: week 3, for comparability)."""
    n, W = post.shape
    expected = HOLDBACK * gmv90 / WEEKS_HIST
    hold = HOLDBACK * post
    cum = np.cumsum(hold, axis=1)
    T = principal * factor
    active = np.ones_like(cum, dtype=bool)
    active[:, 1:] = cum[:, :-1] < T[:, None] - 1e-9
    exp_cum = np.minimum(expected[:, None] * np.arange(1, W + 1)[None, :], T[:, None])
    ratio = cum / exp_cum
    first = np.zeros(n, int)
    for w in range(2, W):
        f = (ratio[:, w] < thr) & active[:, w] & (first == 0)
        first[f] = w + 1
    return first


def projected_payoff_flags(post, gmv90, principal, factor, D, kappa):
    """Baseline 2: from week 4, project the payoff week from the trailing four-week mean
    holdback; flag when the projection exceeds kappa x the offer's own deadline."""
    n, W = post.shape
    hold = HOLDBACK * post
    cum = np.cumsum(hold, axis=1)
    T = principal * factor
    active = np.ones_like(cum, dtype=bool)
    active[:, 1:] = cum[:, :-1] < T[:, None] - 1e-9
    first = np.zeros(n, int)
    for w in range(3, W):
        recent = hold[:, w - 3:w + 1].mean(1)
        outstanding = np.maximum(T - cum[:, w], 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            proj = np.where(recent > 0, (w + 1) + outstanding / recent, np.inf)
        f = (proj > kappa * D) & active[:, w] & (first == 0)
        first[f] = w + 1
    return first


def monitor_eval(flag_week, bad, D, min_lead=MIN_LEAD_WEEKS):
    """flag_week: 1-indexed calendar week of the first review flag (0 = none). bad: missed own deadline.
    A flag counts only if it arrives at least min_lead weeks before the deadline."""
    flagged = flag_week > 0
    early = flagged & (flag_week <= D - min_lead)
    tp = early & bad
    fp = early & ~bad
    late_on_bad = flagged & bad & ~early
    n_good = int((~bad).sum())
    prec = float(tp.sum() / max(early.sum(), 1))
    rec = float(tp.sum() / max(bad.sum(), 1))
    lead = (D - flag_week)[tp]
    return dict(n=int(len(bad)), event_rate=float(bad.mean()), flags_any=int(flagged.sum()),
                flags_actionable=int(early.sum()), tp=int(tp.sum()), fp=int(fp.sum()),
                fn=int((bad & ~early).sum()), late_flags_on_events=int(late_on_bad.sum()),
                precision=prec, recall=rec, f1=float(2 * prec * rec / max(prec + rec, 1e-12)),
                false_alarm_rate=float(fp.sum() / max(n_good, 1)),
                flags_per_100_advances=float(100 * early.sum() / len(bad)),
                flags_per_event_caught=float(early.sum() / max(tp.sum(), 1)),
                median_lead_weeks=float(np.median(lead)) if len(lead) else None,
                p25_lead_weeks=float(np.quantile(lead, 0.25)) if len(lead) else None)


def monitoring_suite(pop, m, E, FR, thr_a2=None, thr_cum=None, kappa=None, sweep=False):
    """Run Algorithm 2 and the two baselines on funded sellers m; select thresholds by F1 if not given."""
    out = realised_outcomes(pop["post"][m], pop["gmv90"][m], E[m], FR[m])
    bad = out["missed_own_window"]
    post, g, A, F, D = pop["post"][m], pop["gmv90"][m], out["A"], out["FR"], out["D"]

    def best(grid, fn):
        scores = {}
        for th in grid:
            scores[th] = monitor_eval(fn(th), bad, D)["f1"]
        th = max(scores, key=scores.get)
        return th, scores

    res = {}
    if thr_a2 is None:
        thr_a2, res["a2_f1_by_threshold_validation"] = best(THR_GRID_A2, lambda th: algorithm2_flags(post, g, A, F, th)[0])
    if thr_cum is None:
        thr_cum, res["cum_f1_by_threshold_validation"] = best(THR_GRID_CUM, lambda th: cumulative_progress_flags(post, g, A, F, th))
    if kappa is None:
        kappa, res["proj_f1_by_kappa_validation"] = best(THR_GRID_PROJ, lambda k: projected_payoff_flags(post, g, A, F, D, k))
    fr_, fw_, esc_ = algorithm2_flags(post, g, A, F, thr_a2)
    a2 = monitor_eval(fr_, bad, D)
    a2["threshold"] = thr_a2
    a2["escalate_share_of_flags"] = float(esc_[fr_ > 0].mean()) if (fr_ > 0).any() else 0.0
    a2["watch_share_before_deadline"] = float(((fw_ > 0) & (fw_ <= D - MIN_LEAD_WEEKS)).mean())
    a2["recall_52w_unpaid"] = float(((fr_ > 0) & (fr_ <= D - MIN_LEAD_WEEKS) & out["unpaid_any"]).sum() / max(out["unpaid_any"].sum(), 1))
    res["algorithm2"] = a2
    cb = monitor_eval(cumulative_progress_flags(post, g, A, F, thr_cum), bad, D); cb["threshold"] = thr_cum
    res["baseline_cumulative_progress"] = cb
    pb = monitor_eval(projected_payoff_flags(post, g, A, F, D, kappa), bad, D); pb["kappa"] = kappa
    res["baseline_projected_payoff"] = pb
    res["frozen"] = dict(thr_a2=thr_a2, thr_cum=thr_cum, kappa=kappa)
    if sweep:
        res["a2_sweep"] = {f"thr_{th:.2f}": monitor_eval(algorithm2_flags(post, g, A, F, th)[0], bad, D)
                           for th in (0.40, 0.50, 0.60, 0.70, 0.80, 0.90)}
        res["cum_sweep"] = {f"thr_{th:.2f}": monitor_eval(cumulative_progress_flags(post, g, A, F, th), bad, D)
                            for th in (0.40, 0.50, 0.60, 0.70, 0.80, 0.90)}
    return res


# ----------------------------------------------------------------------
# E8. Parity helpers (descriptive diagnostics; synthetic groups are not protected classes)
# ----------------------------------------------------------------------
def approval_parity(p, groups, thr=None, approve_rate=None):
    if thr is None:
        thr = np.quantile(p, 1 - approve_rate)
    appr = p >= thr
    rates, sizes = {}, {}
    for g in np.unique(groups):
        mg = groups == g
        rates[str(g)] = float(appr[mg].mean()); sizes[str(g)] = int(mg.sum())
    return dict(rates=rates, sizes=sizes, air=float(min(rates.values()) / max(rates.values())),
                overall_approval=float(appr.mean()), threshold=float(thr))


# ----------------------------------------------------------------------
# One full run for a seed
# ----------------------------------------------------------------------
def run_seed(seed, hazard_shift=OOT_HAZARD_SHIFT, extras=False):
    rng = np.random.default_rng(seed)
    train = gen_population(rng)
    test = gen_population(rng, hazard_scale=hazard_shift)      # held-out cohort with simulated shift
    val = gen_population(rng, hazard_scale=1.0)                # validation cohort for threshold selection (no shift)
    ltr, lte, lva = make_labels(train), make_labels(test), make_labels(val)
    ytr, yte, yva = ltr["y_window"], lte["y_window"], lva["y_window"]

    res = {}
    # --- E2 held-out comparison (prediction target = reference product P0)
    models = {
        "bureau_proxy": (logit(), bureau_X(train), bureau_X(test), bureau_X(val)),
        "logit_behavioral": (logit(), train["X"], test["X"], val["X"]),
        "pbcm_gbm": (gbm(seed), train["X"], test["X"], val["X"]),
        "pbcm_plus_bureau": (gbm(seed), np.column_stack([train["X"], train["bureau"]]),
                             np.column_stack([test["X"], test["bureau"]]), np.column_stack([val["X"], val["bureau"]])),
    }
    preds, pval, fitted = {}, {}, {}
    for name, (mdl, Xa, Xb, Xv) in models.items():
        mdl.fit(Xa, ytr)
        fitted[name] = mdl
        preds[name] = mdl.predict_proba(Xb)[:, 1]
        pval[name] = mdl.predict_proba(Xv)[:, 1]
        res[name] = evaluate(yte, preds[name], test["thin"])
    res["auc_diff_logit_minus_gbm"] = paired_bootstrap_auc_diff(yte, preds["logit_behavioral"], preds["pbcm_gbm"])
    res["calibration"] = dict(pbcm_gbm=calibration_slope_intercept(yte, preds["pbcm_gbm"]),
                              logit_behavioral=calibration_slope_intercept(yte, preds["logit_behavioral"]),
                              bureau_proxy=calibration_slope_intercept(yte, preds["bureau_proxy"]),
                              bins_pbcm=calib_bins(yte, preds["pbcm_gbm"]))
    # ranking robustness to the 52-week outcome AND a separately trained 52-week model
    res["label52"] = dict(auc_25wk_model_on_52wk_outcome={k: float(roc_auc_score(lte["y_52"], preds[k])) for k in ("bureau_proxy", "pbcm_gbm")},
                          auc_52wk_model_on_52wk_outcome=float(roc_auc_score(lte["y_52"], fit_predict(gbm(seed), train["X"], ltr["y_52"], test["X"]))),
                          base_rate_52wk_test=float(lte["y_52"].mean()))
    # in-time 5-fold (also provides out-of-fold scores for pricing inputs)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    p_oof = cross_val_predict(gbm(seed), train["X"], ytr, cv=skf, method="predict_proba")[:, 1]
    res["pbcm_gbm_in_time_cv"] = evaluate(ytr, p_oof)
    # shuffled-label negative control
    p_null = fit_predict(gbm(seed), train["X"], np.random.default_rng(seed + 99).permutation(ytr), test["X"])
    res["shuffled_label_control_auc"] = float(roc_auc_score(yte, p_null))

    # --- Frozen operating approval policy (threshold chosen on validation, applied to test)
    frozen = {}
    for name in ("bureau_proxy", "logit_behavioral", "pbcm_gbm"):
        thr, k = select_budget_threshold(yva, pval[name])
        appr = preds[name] >= thr
        frozen[name] = dict(threshold=thr, validation_approval=float(k / len(yva)),
                            test_approval=float(appr.mean()),
                            test_bad_rate_reference_target=float(1 - yte[appr].mean()) if appr.any() else None,
                            validation_bad_rate=float(1 - yva[pval[name] >= thr].mean()) if (pval[name] >= thr).any() else None)
    res["frozen_policy"] = frozen
    E = preds["pbcm_gbm"]
    thr_A = frozen["pbcm_gbm"]["threshold"]
    book = E >= max(thr_A, E_FLOOR_SIZING)          # the operating book

    # --- E4 selection: approved-only training vs a random-50% control
    pb_tr = fit_predict(logit(), bureau_X(train), ytr, bureau_X(train))
    keep = pb_tr >= np.quantile(pb_tr, 0.50)
    rnd = np.random.default_rng(seed + 7).random(len(ytr)) < 0.50
    p_ri = fit_predict(gbm(seed), train["X"][keep], ytr[keep], test["X"])
    p_rc = fit_predict(gbm(seed), train["X"][rnd], ytr[rnd], test["X"])
    res["selection"] = dict(auc_full=res["pbcm_gbm"]["auc"], auc_incumbent_approved_50=float(roc_auc_score(yte, p_ri)),
                            auc_random_50_control=float(roc_auc_score(yte, p_rc)),
                            approved_share_used=float(keep.mean()), random_share_used=float(rnd.mean()),
                            bad60_full=res["pbcm_gbm"]["bad_rate_at_60pct_approval"],
                            bad60_incumbent_approved=retrospective_frontier(yte, p_ri)[0],
                            bad60_random_control=retrospective_frontier(yte, p_rc)[0])

    # --- E8 per-feature ablations + parity at the frozen operating threshold and at a fixed 60% rate
    cat_names = np.array(CAT_NAMES)[test["cat"]]
    ten_grp = np.where(test["tenure"] < 12, "tenure<12", "tenure>=12")
    abl_preds = {"pbcm_gbm": E}
    for label, drop in (("drop_declared_concentration", ["platform_concentration"]),
                        ("drop_category_index", ["category_risk_index"]),
                        ("drop_both", ["platform_concentration", "category_risk_index"])):
        cols = [i for i, f in enumerate(FEATURES) if f not in drop]
        abl_preds[label] = fit_predict(gbm(seed), train["X"][:, cols], ytr, test["X"][:, cols])
    res["ablation_auc"] = {k: float(roc_auc_score(yte, v)) for k, v in abl_preds.items()}
    par = {}
    for name, p in list(abl_preds.items()) + [("bureau_proxy", preds["bureau_proxy"])]:
        thr_use = frozen["pbcm_gbm"]["threshold"] if name != "bureau_proxy" else frozen["bureau_proxy"]["threshold"]
        par[name] = dict(operating=dict(category=approval_parity(p, cat_names, thr=thr_use),
                                        tenure=approval_parity(p, ten_grp, thr=thr_use)),
                         fixed60=dict(category=approval_parity(p, cat_names, approve_rate=0.60),
                                      tenure=approval_parity(p, ten_grp, approve_rate=0.60)))
    res["parity"] = par
    m_full = fitted["pbcm_gbm"]
    base_auc = res["pbcm_gbm"]["auc"]
    imp, prng = {}, np.random.default_rng(seed + 1000)
    for j, f in enumerate(FEATURES):
        Xp = test["X"].copy(); Xp[:, j] = prng.permutation(Xp[:, j])
        imp[f] = float(base_auc - roc_auc_score(yte, m_full.predict_proba(Xp)[:, 1]))
    res["permutation_importance_auc_drop"] = imp

    # --- E5 pricing: tier inputs estimated under the offered terms (train, out-of-fold), evaluated on test
    tp = fit_tier_pricing(train, p_oof)
    L0, tau0 = reference_shortcut_inputs(train, ltr)
    t_te = tier_index(E)
    fund = t_te >= 0                                    # "if funded" view over the whole sizing domain
    FR_tier = np.where(fund, np.array(tp["FR"])[np.clip(t_te, 0, None)], np.nan)
    FR_short = fr_shortcut(np.clip(E, 0.5, 1), L0, tau0)
    FR_linear = fr_lin(E)
    rules = {"tier_expected_loss": FR_tier, "reference_shortcut": FR_short, "linear_rule": FR_linear}
    outs = {k: realised_outcomes(test["post"], test["gmv90"], E, v) for k, v in rules.items()}
    pricing = dict(tier_inputs=tp, reference_shortcut=dict(L0=L0, tau0=tau0),
                   cap=cap_feasibility(tp["FR"]), market_cap=FR_MARKET_CAP,
                   tiers={k: tier_table(o, np.where(fund, t_te, -1), yref=yte) for k, o in outs.items()},
                   flatness={k: flatness(tier_table(o, np.where(fund, t_te, -1))) for k, o in outs.items()},
                   portfolio_if_funded_all_tiers={k: portfolio(o, fund) for k, o in outs.items()},
                   portfolio_operating_book={k: portfolio(o, book) for k, o in outs.items()},
                   operating_book_threshold=float(max(thr_A, E_FLOOR_SIZING)))
    # paired no-shift cohort: identical features and draws, hazard multiplier 1.0
    ns = paired_no_shift_cohort(seed)
    assert np.array_equal(ns["X"], test["X"])
    o_ns = realised_outcomes(ns["post"], ns["gmv90"], E, FR_tier)
    lns = make_labels(ns)
    pricing["no_shift_paired"] = dict(tiers=tier_table(o_ns, np.where(fund, t_te, -1)),
                                      portfolio_operating_book=portfolio(o_ns, book),
                                      reference_target_miss_rate_no_shift=float(1 - lns["y_window"].mean()),
                                      reference_target_miss_rate_shift=float(1 - yte.mean()))
    # common-shock and collection-base stresses (deterministic, illustrative, no probabilities attached)
    stress = []
    for mult in (1.0, 0.9, 0.8, 0.7, 0.6):
        o = realised_outcomes(test["post"] * mult, test["gmv90"], E, FR_tier)
        stress.append(dict(kind="all_funded_sellers_gmv_x", value=mult, **portfolio(o, book)))
    o = realised_outcomes(test["post"] * 0.88, test["gmv90"], E, FR_tier)
    stress.append(dict(kind="collection_base_12pct_below_sizing_base", value=0.88, **portfolio(o, book)))
    pricing["stress"] = stress
    res["pricing"] = pricing

    # --- E7 monitoring: thresholds selected on validation (funded at the tier prices), frozen, evaluated on test
    Ev = pval["pbcm_gbm"]
    t_va = tier_index(Ev)
    book_v = Ev >= max(thr_A, E_FLOOR_SIZING)
    FR_v = np.where(t_va >= 0, np.array(tp["FR"])[np.clip(t_va, 0, None)], np.nan)
    sel = monitoring_suite(val, book_v, Ev, FR_v)
    mon = monitoring_suite(test, book, E, FR_tier, thr_a2=sel["frozen"]["thr_a2"], thr_cum=sel["frozen"]["thr_cum"],
                           kappa=sel["frozen"]["kappa"], sweep=extras)
    mon["selection_on_validation"] = {k: v for k, v in sel.items() if k.endswith("validation")}
    res["monitoring"] = mon

    # --- population descriptives
    res["population"] = dict(
        n_train=int(N_SELLERS), n_test=int(N_SELLERS), n_validation=int(N_SELLERS),
        reference_target_repay_rate_train=float(ytr.mean()), reference_target_repay_rate_test=float(yte.mean()),
        reference_target_repay_rate_validation=float(yva.mean()),
        repay_52wk_rate_test=float(lte["y_52"].mean()),
        shock_share_test=float(test["shocked"].mean()), shock_share_train=float(train["shocked"].mean()),
        thin_file_share=float(test["thin"].mean()),
        est_window_weeks=float(ltr["est_window"]), label_horizon_weeks=int(ltr["horizon"]),
        median_gmv90=float(np.median(test["gmv90"])), median_tenure=float(np.median(test["tenure"])),
        misreport_share_platform_conc=float(test["misreport"].mean()),
        feature_medians={f: float(np.median(test["X"][:, j])) for j, f in enumerate(FEATURES)},
        operating_book_share=float(book.mean()))
    state = dict(train=train, test=test, val=val, ltr=ltr, lte=lte, preds=preds, E=E, tier_pricing=tp,
                 L0=L0, tau0=tau0, model=m_full, seed=seed, thr_A=thr_A, book=book, frozen_mon=mon["frozen"],
                 FR_tier=FR_tier, p_oof=p_oof)
    return res, state


# ----------------------------------------------------------------------
# E3. Robustness sweeps (primary seed)
# ----------------------------------------------------------------------
def robustness(seed, thr_A):
    out = dict(rho_thick_sweep=[], thin_share_sweep=[], feature_noise_sweep=[], missingness_sweep=[], manipulation=[])
    for rho in (0.30, 0.40, 0.55, 0.70, 0.85):
        rng = np.random.default_rng(seed)
        tr = gen_population(rng, rho_thick=rho)
        te = gen_population(rng, hazard_scale=OOT_HAZARD_SHIFT, rho_thick=rho)
        ytr, yte = make_labels(tr)["y_window"], make_labels(te)["y_window"]
        pb = fit_predict(logit(), bureau_X(tr), ytr, bureau_X(te))
        pp = fit_predict(gbm(seed), tr["X"], ytr, te["X"])
        pc = fit_predict(gbm(seed), np.column_stack([tr["X"], tr["bureau"]]), ytr, np.column_stack([te["X"], te["bureau"]]))
        out["rho_thick_sweep"].append(dict(rho_thick=rho, auc_bureau=float(roc_auc_score(yte, pb)),
                                           auc_pbcm=float(roc_auc_score(yte, pp)), auc_combined=float(roc_auc_score(yte, pc)),
                                           auc_bureau_thick_only=float(roc_auc_score(yte[~te["thin"]], pb[~te["thin"]])),
                                           auc_pbcm_thick_only=float(roc_auc_score(yte[~te["thin"]], pp[~te["thin"]])),
                                           auc_combined_thick_only=float(roc_auc_score(yte[~te["thin"]], pc[~te["thin"]]))))
    for share in (0.0, 0.25, 0.50, 0.75):
        rng = np.random.default_rng(seed)
        tr = gen_population(rng, thin_extra=share)
        te = gen_population(rng, hazard_scale=OOT_HAZARD_SHIFT, thin_extra=share)
        ytr, yte = make_labels(tr)["y_window"], make_labels(te)["y_window"]
        pb = fit_predict(logit(), bureau_X(tr), ytr, bureau_X(te))
        pp = fit_predict(gbm(seed), tr["X"], ytr, te["X"])
        out["thin_share_sweep"].append(dict(thin_extra_share=share, realised_thin_share=float(te["thin"].mean()),
                                            auc_bureau=float(roc_auc_score(yte, pb)), auc_pbcm=float(roc_auc_score(yte, pp))))
    # base cohorts for the feature-quality experiments
    rng = np.random.default_rng(seed)
    tr = gen_population(rng); te = gen_population(rng, hazard_scale=OOT_HAZARD_SHIFT)
    ytr, yte = make_labels(tr)["y_window"], make_labels(te)["y_window"]
    pb = fit_predict(logit(), bureau_X(tr), ytr, bureau_X(te))
    auc_bureau = float(roc_auc_score(yte, pb))
    for kappa in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        nrng = np.random.default_rng(seed + 500)
        Xtr = add_feature_noise(tr["X"], kappa, nrng); Xte = add_feature_noise(te["X"], kappa, nrng)
        pp = fit_predict(gbm(seed), Xtr, ytr, Xte)
        pl = fit_predict(logit(), Xtr, ytr, Xte)
        pc = fit_predict(gbm(seed), np.column_stack([Xtr, tr["bureau"]]), ytr, np.column_stack([Xte, te["bureau"]]))
        out["feature_noise_sweep"].append(dict(noise_sd_multiple=kappa, auc_pbcm=float(roc_auc_score(yte, pp)),
                                               auc_logit=float(roc_auc_score(yte, pl)), auc_bureau=auc_bureau,
                                               auc_combined=float(roc_auc_score(yte, pc))))
    for q in (0.0, 0.10, 0.25, 0.50, 0.75):
        mrng = np.random.default_rng(seed + 600)
        Xtr = apply_missingness(tr["X"], q, mrng); Xte = apply_missingness(te["X"], q, mrng)
        pp = fit_predict(gbm(seed), Xtr, ytr, Xte)
        out["missingness_sweep"].append(dict(missing_share_quality_features=q, auc_pbcm=float(roc_auc_score(yte, pp)),
                                             auc_bureau=auc_bureau))
    # manipulation: model trained on clean data; a share of test applicants game three features
    model = gbm(seed).fit(tr["X"], ytr)
    p_clean = model.predict_proba(te["X"])[:, 1]
    for share in (0.10, 0.25):
        Xm, who = manipulate(te["X"], share, np.random.default_rng(seed + 700))
        p_m = model.predict_proba(Xm)[:, 1]
        flipped = who & (p_clean < thr_A) & (p_m >= thr_A)
        out["manipulation"].append(dict(manipulator_share=share, auc_clean=float(roc_auc_score(yte, p_clean)),
                                        auc_manipulated=float(roc_auc_score(yte, p_m)),
                                        mean_score_uplift_manipulators=float((p_m - p_clean)[who].mean()),
                                        approval_rate_manipulators_clean=float((p_clean[who] >= thr_A).mean()),
                                        approval_rate_manipulators_gamed=float((p_m[who] >= thr_A).mean()),
                                        share_of_manipulators_flipped_to_approve=float(flipped[who].mean()),
                                        miss_rate_among_flipped=float(1 - yte[flipped].mean()) if flipped.any() else None,
                                        miss_rate_among_approved_clean=float(1 - yte[p_clean >= thr_A].mean()),
                                        operating_threshold=float(thr_A)))
    return out


# ----------------------------------------------------------------------
# E6. Deployment scenarios (supplement; assumption-driven illustrations)
# ----------------------------------------------------------------------
def truncnorm(rng, mean, sd, lo, hi, n):
    x = np.empty(0)
    while len(x) < n:
        d = rng.normal(mean, sd, 2 * n)
        x = np.concatenate([x, d[(d >= lo) & (d <= hi)]])
    return x[:n]


def scenario_mc(rng, advance=1000.0, factor=1.10, n=200_000, adverse=False):
    """Twelve-week inventory deployment, net of the fee, RELATIVE TO NOT BORROWING (no restock:
    zero incremental profit, zero fee). Unsold inventory is impaired, not carried at cost."""
    fee = advance * (factor - 1)
    if adverse:
        margin = truncnorm(rng, 0.25, 0.08, 0.05, 0.60, n); sell = rng.beta(4, 3, n); impair = 0.50
    else:
        margin = truncnorm(rng, 0.35, 0.08, 0.10, 0.60, n); sell = rng.beta(6, 2, n); impair = 0.30
    carry = 0.015 * 3.0
    gross_profit = sell * advance * margin / (1 - margin)
    holding = (1 - sell) * advance * carry
    impairment = (1 - sell) * advance * impair
    net = gross_profit - fee - holding - impairment
    deficit = -fee * np.ones(n)          # funds absorbed by operating deficit: sales unchanged by assumption

    def s(x):
        return dict(mean=float(x.mean()), median=float(np.median(x)), p10=float(np.quantile(x, 0.10)),
                    p90=float(np.quantile(x, 0.90)), prob_positive=float((x > 0).mean()))
    be = fee / (net.mean() + fee) if net.mean() + fee > 0 else None
    return dict(advance=advance, factor=factor, fee=fee, adverse=adverse, growth=s(net), deficit=s(deficit),
                breakeven_growth_share_identity=float(be) if be else None,
                assumptions=dict(margin=("TruncNormal(0.25, 0.08) on [0.05, 0.60]" if adverse else "TruncNormal(0.35, 0.08) on [0.10, 0.60]") + " (rejection-sampled)",
                                 sell_through_12w=("Beta(4, 3), mean 0.57" if adverse else "Beta(6, 2), mean 0.75"),
                                 unsold_inventory=f"impaired by {impair:.0%} of cost plus 1.5%/month carrying cost for 3 months",
                                 counterfactual="no financing: no restock, zero incremental profit, zero fee",
                                 not_modelled="liquidity over time; displacement of other financing; effect on repayment path; any causal effect of seller education"))


# ----------------------------------------------------------------------
# Table 1 arithmetic: discrete schedule, cash-flow IRR, fee annualisation
# ----------------------------------------------------------------------
def weekly_irr(schedule, principal):
    lo, hi = 0.0, 1.0
    f = lambda r: sum(c / (1 + r) ** (k + 1) for k, c in enumerate(schedule)) - principal
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def apr_fixture(advance=1000.0, factor=1.25, weekly_settlement=1153.85):
    T = round(advance * factor, 2)
    hb = round(HOLDBACK * weekly_settlement, 2)
    sched, cum = [], 0.0
    while cum < T - 1e-9:
        pay = min(hb, round(T - cum, 2)); sched.append(pay); cum = round(cum + pay, 2)
    r = weekly_irr(sched, advance)
    weeks = len(sched)
    return dict(advance=advance, factor=factor, obligation=T, fee=round(T - advance, 2), weekly_holdback=hb,
                schedule=sched, payoff_week=weeks, continuous_weeks=round(T / hb, 2),
                weekly_irr=r, nominal_annual_irr_52x=52 * r, effective_annual_irr=(1 + r) ** 52 - 1,
                fee_annualisation_continuous=(factor - 1) * 52 / (T / hb),
                lender_funding_cost_discrete=round(advance * COST_OF_FUNDS * weeks / 52, 2),
                lender_funding_cost_continuous=round(advance * COST_OF_FUNDS * (T / hb) / 52, 2),
                platform_share_of_fee=round(PLATFORM_SHARE * (T - advance), 2),
                lender_gross_fee=round((1 - PLATFORM_SHARE) * (T - advance), 2),
                lender_net_discrete=round((1 - PLATFORM_SHARE) * (T - advance) - advance * COST_OF_FUNDS * weeks / 52, 2))


# ----------------------------------------------------------------------
# WX. Worked example
# ----------------------------------------------------------------------
def shapley_sampling(model, x, background, rng, n_samples=400):
    d = len(x)
    phi = np.zeros(d)
    for _ in range(n_samples):
        z = background[rng.integers(len(background))]
        perm = rng.permutation(d)
        x_with = z.copy(); x_without = z.copy()
        for j in perm:
            x_with[j] = x[j]
            p_with = model.predict_proba(x_with[None, :])[0, 1]
            p_without = model.predict_proba(x_without[None, :])[0, 1]
            phi[j] += p_with - p_without
            x_without[j] = x[j]
    return phi / n_samples


def worked_example(state, rng):
    model, test, tp = state["model"], state["test"], state["tier_pricing"]
    x = np.array([15000.0, 0.083, 0.21, 72.0, 0.009, 0.97, 18.0, 0.34, 0.18, 0.85])
    E = float(model.predict_proba(x[None, :])[0, 1])
    E_r = round(E, 3)
    t = int(tier_index(np.array([E_r]))[0])
    alpha = float(alpha_lin(E_r))
    A_max = round(alpha * x[0], 2)
    FR = round(float(tp["FR"][t]), 2)
    FR_l = round(float(fr_lin(E_r)), 2)
    FR_s = round(float(fr_shortcut(E_r, state["L0"], state["tau0"])), 3)
    # timing (prototype measurement on this machine, not a benchmark)
    t0 = time.perf_counter()
    for _ in range(100):
        model.predict_proba(x[None, :])
    infer_ms = (time.perf_counter() - t0) / 100 * 1000
    bg = test["X"][rng.choice(len(test["X"]), 200, replace=False)]
    t0 = time.perf_counter(); phi = shapley_sampling(model, x, bg, rng, 400); shap400_s = time.perf_counter() - t0
    t0 = time.perf_counter(); phi50 = shapley_sampling(model, x, bg, np.random.default_rng(1), 50); shap50_s = time.perf_counter() - t0
    contrib = sorted(zip(FEATURES, phi.tolist()), key=lambda z: -abs(z[1]))
    draw = 1000.0
    T = round(draw * FR, 2)
    weekly_settle = round(x[0] / WEEKS_HIST, 2)
    weekly_hold = round(HOLDBACK * weekly_settle, 2)
    weeks = math.ceil(T / weekly_hold)
    disb = date(2025, 7, 1)
    first_settle = disb + timedelta(days=6)
    schedule, cum = [], 0.0
    for k in range(weeks):
        d = first_settle + timedelta(weeks=k)
        hb = min(weekly_hold, round(T - cum, 2)); cum = round(cum + hb, 2)
        schedule.append(dict(settlement_date=d.isoformat(), gross=weekly_settle, holdback=hb,
                             net_payout=round(weekly_settle - hb, 2), cumulative=cum,
                             outstanding=round(T - cum, 2), velocity=round(hb / weekly_hold, 2)))
    payoff = schedule[-1]["settlement_date"]
    apr = apr_fixture(draw, FR, weekly_settle)
    thr = state["frozen_mon"]["thr_a2"]
    # stressed path: sales fall 60% from week 5 (velocity 0.40) — evaluated at the FROZEN threshold
    stressed_post = np.array([[weekly_settle * (0.40 if k >= 4 else 1.0) for k in range(WEEKS_POST)]])
    fr_, fw_, esc_ = algorithm2_flags(stressed_post, np.array([x[0]]), np.array([draw]), np.array([FR]), thr)
    cum_s, T_s = repay_path(stressed_post, np.array([draw]), np.array([FR]))
    stressed = [dict(week=k + 1, gross=round(float(stressed_post[0, k]), 2), holdback=round(float(HOLDBACK * stressed_post[0, k]), 2),
                     velocity=round(float(HOLDBACK * stressed_post[0, k] / weekly_hold), 2),
                     outstanding=round(float(max(T_s[0] - cum_s[0, k], 0)), 2)) for k in range(12)]
    flag = dict(threshold=thr, first_watch_week=int(fw_[0]) or None, first_review_week=int(fr_[0]) or None,
                decision_at_review=("escalate" if esc_[0] else "review") if fr_[0] > 0 else None,
                outstanding_at_review=round(float(max(T_s[0] - cum_s[0, fr_[0] - 1], 0)), 2) if fr_[0] > 0 else None,
                own_deadline_weeks=int(own_deadline(np.array([FR]), np.array([alpha]))[0]),
                own_deadline_for_this_draw_weeks=int(math.ceil(LABEL_MULT * T / weekly_hold)))
    return dict(profile={f: float(v) for f, v in zip(FEATURES, x)}, eligibility_score=E_r, tier=TIERS[t][2],
                alpha=round(alpha, 4), max_advance=A_max, factor_rate_tier=FR, factor_rate_linear_rule=FR_l,
                factor_rate_reference_shortcut=FR_s, total_if_max=round(A_max * FR, 2),
                principal_reasons=[dict(feature=f, phi=round(v, 4)) for f, v in contrib],
                explanation_timing=dict(inference_ms_per_call=infer_ms, shapley_400_permutations_s=shap400_s,
                                        shapley_50_permutations_s=shap50_s, predict_calls_400=8000,
                                        note="single-process prototype measurement; explanations are computed in the nightly feature refresh and cached, not on the eligibility path"),
                shapley_50_vs_400_max_abs_diff=float(np.max(np.abs(phi50 - phi))),
                draw=dict(requested=draw, factor_rate=FR, total_repayment=T, holdback_rate=HOLDBACK,
                          weekly_settlement=weekly_settle, weekly_holdback=weekly_hold, est_repayment_weeks=weeks,
                          disbursement_date=disb.isoformat(), est_payoff_date=payoff, third_event=schedule[2], schedule=schedule,
                          cash_flow_apr=apr),
                max_draw_est_weeks=math.ceil(round(A_max * FR, 2) / weekly_hold),
                stressed_path_flag=flag, stressed_path=stressed)


# ----------------------------------------------------------------------
# Self-tests (run at start; failures abort)
# ----------------------------------------------------------------------
def _selftest():
    # watch can fire from week 1 and week 2; review needs three consecutive weeks
    post = np.array([[40., 40., 100., 100., 100.], [100., 40., 100., 100., 100.], [40., 40., 40., 100., 100.]])
    fr, fw, esc = algorithm2_flags(post, np.array([1300., 1300., 1300.]), np.array([1000.] * 3), np.array([1.1] * 3), 0.5)
    assert fw.tolist() == [1, 2, 1], fw
    assert fr.tolist() == [0, 0, 3], fr
    # repayment conservation and capping
    cum, T = repay_path(np.array([[0., 100., 10000.]]), np.array([100.]), np.array([1.1]))
    assert np.all(np.diff(cum, axis=1) >= 0) and np.all(cum <= T[:, None]) and np.isclose(cum[0, -1], 110)
    # pricing identity under the single waterfall (deterministic loss, expected term)
    for ell in (0.0, 0.05, 0.2):
        for tau in (16.0, 24.0):
            fr_ = fr_star(ell, tau)
            net = fr_ * (1 - ell) - PLATFORM_SHARE * (fr_ * (1 - ell) - 1) - 1 - COST_OF_FUNDS * tau / 52
            assert abs(net - TARGET_MARGIN) < 1e-12
    # monitoring evaluation: late flags do not count
    ev = monitor_eval(np.array([5, 24, 0, 10]), np.array([True, True, True, False]), np.array([25, 25, 25, 25]))
    assert ev["tp"] == 1 and ev["fn"] == 2 and ev["fp"] == 1 and ev["late_flags_on_events"] == 1
    # IRR fixture reproduces the independent check (7 x 173.08 + 38.44 at FR 1.25)
    a = apr_fixture()
    assert a["schedule"][-1] == 38.44 and a["payoff_week"] == 8 and abs(a["weekly_irr"] - 0.057342) < 1e-5


# ----------------------------------------------------------------------
# Figures and source data
# ----------------------------------------------------------------------
def write_csv(name, header, rows):
    with open(os.path.join(OUT_DIR, "source_data", name), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def figures(primary, state, rob, multi, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    os.makedirs(os.path.join(OUT_DIR, "source_data"), exist_ok=True)
    yte = state["lte"]["y_window"]
    # Fig 2: ROC
    fig, ax = plt.subplots(figsize=(6.2, 4.6)); rows = []
    for name, lab, ls in (("bureau_proxy", "Bureau-proxy score", "-"), ("logit_behavioral", "Logistic, behavioural", "--"),
                          ("pbcm_gbm", "PBCM (gradient boosting)", "-"), ("pbcm_plus_bureau", "PBCM + bureau", ":")):
        fpr, tpr, _ = roc_curve(yte, state["preds"][name])
        ax.plot(fpr, tpr, ls, label=f"{lab} (AUC {primary[name]['auc']:.3f})")
        rows += [(name, float(a), float(b)) for a, b in zip(fpr[::50], tpr[::50])]
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Held-out cohort ROC (prediction target: reference product P0)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig2_roc.png"), dpi=200); plt.close(fig)
    write_csv("fig2_roc.csv", ["model", "fpr", "tpr"], rows)
    # Fig 3: subgroup AUC with 10-seed error bars
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    names = ["bureau_proxy", "logit_behavioral", "pbcm_gbm"]
    keys_thin = {"bureau_proxy": "auc_bureau_thin", "logit_behavioral": "auc_logit_thin", "pbcm_gbm": "auc_pbcm_thin"}
    keys_thick = {"bureau_proxy": "auc_bureau_thick", "logit_behavioral": "auc_logit_thick", "pbcm_gbm": "auc_pbcm_thick"}
    xk = np.arange(len(names)); w = 0.38
    ax.bar(xk - w / 2, [multi[keys_thin[n]]["mean"] for n in names], w, yerr=[multi[keys_thin[n]]["sd"] for n in names], capsize=3, label="Thin-file sellers", hatch="//")
    ax.bar(xk + w / 2, [multi[keys_thick[n]]["mean"] for n in names], w, yerr=[multi[keys_thick[n]]["sd"] for n in names], capsize=3, label="Thick-file sellers")
    ax.set_xticks(xk); ax.set_xticklabels(["Bureau proxy", "Logistic, behavioural", "PBCM"]); ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUC, held-out cohort (mean ± sd, 10 seeds)"); ax.set_title("AUC by bureau-file thickness"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig3_subgroup_auc.png"), dpi=200); plt.close(fig)
    write_csv("fig3_subgroup_auc.csv", ["model", "group", "auc_mean", "auc_sd"],
              [(n, g, multi[k[n]]["mean"], multi[k[n]]["sd"]) for n in names for g, k in (("thin", keys_thin), ("thick", keys_thick))])
    # Fig 4: crossover — feature noise vs bureau strength
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.0))
    r = rob["feature_noise_sweep"]
    axs[0].plot([d["noise_sd_multiple"] for d in r], [d["auc_pbcm"] for d in r], "s-", label="PBCM")
    axs[0].plot([d["noise_sd_multiple"] for d in r], [d["auc_logit"] for d in r], "d--", label="Logistic, behavioural")
    axs[0].plot([d["noise_sd_multiple"] for d in r], [d["auc_combined"] for d in r], "^:", label="PBCM + bureau")
    axs[0].axhline(r[0]["auc_bureau"], color="gray", ls="-.", label=f"Bureau proxy (ρ = 0.55): {r[0]['auc_bureau']:.3f}")
    axs[0].set_xlabel("Feature measurement noise (multiples of each feature's sd)"); axs[0].set_ylabel("AUC, held-out cohort")
    axs[0].set_title("(a) Behavioural signal degraded"); axs[0].legend(fontsize=7); axs[0].grid(alpha=0.3)
    r2 = rob["rho_thick_sweep"]
    axs[1].plot([d["rho_thick"] for d in r2], [d["auc_bureau"] for d in r2], "o-", label="Bureau proxy, all")
    axs[1].plot([d["rho_thick"] for d in r2], [d["auc_bureau_thick_only"] for d in r2], "o--", label="Bureau proxy, thick files")
    axs[1].plot([d["rho_thick"] for d in r2], [d["auc_pbcm"] for d in r2], "s-", label="PBCM, all")
    axs[1].plot([d["rho_thick"] for d in r2], [d["auc_combined_thick_only"] for d in r2], "^:", label="PBCM + bureau, thick files")
    axs[1].set_xlabel("Bureau signal strength for thick-file sellers (ρ)"); axs[1].set_title("(b) Bureau signal strengthened")
    axs[1].legend(fontsize=7); axs[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig4_robustness.png"), dpi=200); plt.close(fig)
    write_csv("fig4_robustness.csv", ["panel", "x", "series", "auc"],
              [("noise", d["noise_sd_multiple"], k, d[k]) for d in r for k in ("auc_pbcm", "auc_logit", "auc_combined", "auc_bureau")] +
              [("rho", d["rho_thick"], k, d[k]) for d in r2 for k in ("auc_bureau", "auc_bureau_thick_only", "auc_pbcm", "auc_combined_thick_only")])
    # Fig 5: calibration with Wilson intervals
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    cb = primary["calibration"]["bins_pbcm"]
    xs = [b["mean_pred"] for b in cb]; ys = [b["observed"] for b in cb]
    err = [[b["observed"] - b["ci95_lo"] for b in cb], [b["ci95_hi"] - b["observed"] for b in cb]]
    ax.errorbar(xs, ys, yerr=err, fmt="o-", capsize=3, label="PBCM (Platt-scaled), 95% Wilson interval")
    ax.plot([0, 1], [0, 1], "k--", lw=0.7, label="Perfect calibration")
    for b in cb:
        if not b["supported"]:
            ax.annotate(f"n={b['n']}", (b["mean_pred"], b["observed"]), fontsize=7, xytext=(4, -10), textcoords="offset points")
    c = primary["calibration"]["pbcm_gbm"]
    ax.set_xlabel("Predicted probability of repaying reference product P0 on time"); ax.set_ylabel("Observed rate")
    ax.set_title(f"Reliability, held-out cohort (ECE {primary['pbcm_gbm']['ece']:.3f}; slope {c['slope']:.2f}, intercept {c['intercept']:.2f})", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig5_calibration.png"), dpi=200); plt.close(fig)
    write_csv("fig5_calibration.csv", ["bin_lo", "bin_hi", "n", "mean_pred", "observed", "ci95_lo", "ci95_hi"],
              [(b["lo"], b["hi"], b["n"], b["mean_pred"], b["observed"], b["ci95_lo"], b["ci95_hi"]) for b in cb])
    # Fig 6: realised margin by tier under three rules (10 seeds) + price curves
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.0))
    tiers = [t[2] for t in TIERS]; xk = np.arange(len(tiers)); w = 0.27
    for i, (rule, lab) in enumerate((("tier_expected_loss", "Tier expected-loss price (offered terms)"),
                                     ("reference_shortcut", "Reference-product shortcut"), ("linear_rule", "Linear rule"))):
        means = [multi[f"margin_{rule}_{t}"]["mean"] for t in tiers]; sds = [multi[f"margin_{rule}_{t}"]["sd"] for t in tiers]
        axs[0].bar(xk + (i - 1) * w, [100 * v for v in means], w, yerr=[100 * s for s in sds], capsize=2, label=lab)
    axs[0].axhline(100 * TARGET_MARGIN, color="k", lw=0.8, ls="--", label="Target 5%")
    axs[0].axhspan(100 * (TARGET_MARGIN - FLAT_TOL), 100 * (TARGET_MARGIN + FLAT_TOL), color="gray", alpha=0.15, label="Pre-declared tolerance ±1 pt")
    axs[0].set_xticks(xk); axs[0].set_xticklabels([f"{t[0]:.1f}–{min(t[1], 1):.1f}" for t in TIERS], fontsize=8)
    axs[0].set_xlabel("Score band E"); axs[0].set_ylabel("Realised lender margin on principal (%)")
    axs[0].set_title("(a) Realised margin by tier, held-out cohort (10 seeds)"); axs[0].legend(fontsize=6.5); axs[0].grid(axis="y", alpha=0.3)
    tp = primary["pricing"]["tier_inputs"]; Es = np.linspace(0.5, 1.0, 200)
    axs[1].step([t[0] for t in TIERS][::-1] + [1.0], [tp["FR"][i] for i in range(len(TIERS))][::-1] + [tp["FR"][0]], where="post", label="Tier expected-loss price")
    axs[1].plot(Es, fr_shortcut(Es, state["L0"], state["tau0"]), "--", label="Reference-product shortcut")
    axs[1].plot(Es, fr_lin(Es), ":", label="Linear rule")
    axs[1].axhline(FR_MARKET_CAP, color="gray", lw=0.8, ls="-."); axs[1].text(0.51, FR_MARKET_CAP + 0.004, "assumed market cap 1.35", fontsize=7, color="gray")
    axs[1].set_xlabel("Calibrated score E"); axs[1].set_ylabel("Factor rate"); axs[1].set_ylim(1.0, 1.4)
    axs[1].set_title("(b) Price by score under the three rules"); axs[1].legend(fontsize=7); axs[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig6_pricing.png"), dpi=200); plt.close(fig)
    write_csv("fig6_pricing.csv", ["rule", "tier", "margin_mean", "margin_sd"],
              [(rule, t, multi[f"margin_{rule}_{t}"]["mean"], multi[f"margin_{rule}_{t}"]["sd"]) for rule in ("tier_expected_loss", "reference_shortcut", "linear_rule") for t in tiers])
    # Fig 7: monitoring — Algorithm 2 sweep vs baselines at frozen thresholds
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    sw = primary["monitoring"]["a2_sweep"]; ths = sorted(sw)
    ax.plot([float(t[4:]) for t in ths], [sw[t]["precision"] for t in ths], "o-", label="Algorithm 2 precision")
    ax.plot([float(t[4:]) for t in ths], [sw[t]["recall"] for t in ths], "s-", label="Algorithm 2 recall")
    ax.plot([float(t[4:]) for t in ths], [sw[t]["false_alarm_rate"] for t in ths], "^-", label="Algorithm 2 false-alarm rate")
    cs = primary["monitoring"]["cum_sweep"]
    ax.plot([float(t[4:]) for t in ths], [cs[t]["precision"] for t in ths], "o--", color="gray", label="Cumulative-progress precision")
    ax.plot([float(t[4:]) for t in ths], [cs[t]["recall"] for t in ths], "s--", color="gray", label="Cumulative-progress recall")
    fz = primary["monitoring"]["frozen"]["thr_a2"]
    ax.axvline(fz, color="k", lw=0.8, ls=":"); ax.text(fz + 0.005, 0.02, f"frozen θ = {fz:.2f}", fontsize=7)
    ax.set_xlabel("Velocity threshold"); ax.set_ylabel("Rate"); ax.set_ylim(0, 1.05)
    ax.set_title("Early warning of an own-deadline miss (flags with ≥ 2 weeks' lead)", fontsize=10); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig7_monitoring.png"), dpi=200); plt.close(fig)
    write_csv("fig7_monitoring.csv", ["rule", "threshold", "precision", "recall", "false_alarm_rate", "median_lead_weeks"],
              [("algorithm2", float(t[4:]), sw[t]["precision"], sw[t]["recall"], sw[t]["false_alarm_rate"], sw[t]["median_lead_weeks"]) for t in ths] +
              [("cumulative_progress", float(t[4:]), cs[t]["precision"], cs[t]["recall"], cs[t]["false_alarm_rate"], cs[t]["median_lead_weeks"]) for t in ths])
    # Fig 8 (supplement): deployment scenario distributions
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    rng = np.random.default_rng(7)
    for adverse, lab, col in ((False, "Base assumptions", "#2b6cb0"), (True, "Adverse assumptions", "#c05621")):
        margin = truncnorm(rng, 0.25 if adverse else 0.35, 0.08, 0.05 if adverse else 0.10, 0.60, 100_000)
        sell = rng.beta(4, 3, 100_000) if adverse else rng.beta(6, 2, 100_000)
        net = sell * 1000 * margin / (1 - margin) - 100 - (1 - sell) * 1000 * 0.045 - (1 - sell) * 1000 * (0.5 if adverse else 0.3)
        ax.hist(net, bins=80, alpha=0.6, color=col, label=lab)
    ax.axvline(0, color="k", lw=0.8); ax.axvline(-100, color="red", ls="--", lw=0.8, label="Deficit deployment: −fee")
    ax.set_xlabel("12-week net gain vs no financing, $1,000 advance at FR 1.10 (USD)"); ax.set_ylabel("Simulated deployments")
    ax.set_title("Supplement: assumption-driven deployment scenarios"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig8_scenario.png"), dpi=200); plt.close(fig)
    # multi-seed source data
    keys = sorted(runs[0].keys())
    write_csv("multiseed_runs.csv", keys, [[r[k] for k in keys] for r in runs])


# ----------------------------------------------------------------------
def summarise_run(r):
    """Flat dictionary of headline metrics for the multi-seed table."""
    d = dict(seed=r["_seed"],
             auc_bureau=r["bureau_proxy"]["auc"], auc_logit=r["logit_behavioral"]["auc"], auc_pbcm=r["pbcm_gbm"]["auc"],
             auc_combined=r["pbcm_plus_bureau"]["auc"], auc_diff_logit_minus_gbm=r["auc_diff_logit_minus_gbm"]["point"],
             auc_pbcm_thin=r["pbcm_gbm"]["auc_thin_file"], auc_bureau_thin=r["bureau_proxy"]["auc_thin_file"], auc_logit_thin=r["logit_behavioral"]["auc_thin_file"],
             auc_pbcm_thick=r["pbcm_gbm"]["auc_thick_file"], auc_bureau_thick=r["bureau_proxy"]["auc_thick_file"], auc_logit_thick=r["logit_behavioral"]["auc_thick_file"],
             ks_pbcm=r["pbcm_gbm"]["ks"], ece_pbcm=r["pbcm_gbm"]["ece"], brier_pbcm=r["pbcm_gbm"]["brier"],
             calib_slope_pbcm=r["calibration"]["pbcm_gbm"]["slope"], calib_intercept_pbcm=r["calibration"]["pbcm_gbm"]["intercept"],
             bad60_bureau=r["bureau_proxy"]["bad_rate_at_60pct_approval"], bad60_pbcm=r["pbcm_gbm"]["bad_rate_at_60pct_approval"],
             retro8_bureau=r["bureau_proxy"]["retrospective_approval_at_8pct_budget"], retro8_pbcm=r["pbcm_gbm"]["retrospective_approval_at_8pct_budget"],
             frozen_thr_pbcm=r["frozen_policy"]["pbcm_gbm"]["threshold"], frozen_appr_pbcm=r["frozen_policy"]["pbcm_gbm"]["test_approval"],
             frozen_bad_pbcm=r["frozen_policy"]["pbcm_gbm"]["test_bad_rate_reference_target"],
             frozen_thr_bureau=r["frozen_policy"]["bureau_proxy"]["threshold"], frozen_appr_bureau=r["frozen_policy"]["bureau_proxy"]["test_approval"],
             frozen_bad_bureau=r["frozen_policy"]["bureau_proxy"]["test_bad_rate_reference_target"],
             frozen_appr_logit=r["frozen_policy"]["logit_behavioral"]["test_approval"], frozen_bad_logit=r["frozen_policy"]["logit_behavioral"]["test_bad_rate_reference_target"],
             auc_52wk_model=r["label52"]["auc_52wk_model_on_52wk_outcome"], auc_25wk_model_on_52wk=r["label52"]["auc_25wk_model_on_52wk_outcome"]["pbcm_gbm"],
             shuffled_control_auc=r["shuffled_label_control_auc"],
             auc_sel_approved50=r["selection"]["auc_incumbent_approved_50"], auc_sel_random50=r["selection"]["auc_random_50_control"],
             auc_abl_declared=r["ablation_auc"]["drop_declared_concentration"], auc_abl_category=r["ablation_auc"]["drop_category_index"], auc_abl_both=r["ablation_auc"]["drop_both"],
             air_cat_op_pbcm=r["parity"]["pbcm_gbm"]["operating"]["category"]["air"], air_ten_op_pbcm=r["parity"]["pbcm_gbm"]["operating"]["tenure"]["air"],
             air_cat_op_both=r["parity"]["drop_both"]["operating"]["category"]["air"], air_ten_op_both=r["parity"]["drop_both"]["operating"]["tenure"]["air"],
             air_cat_op_declared=r["parity"]["drop_declared_concentration"]["operating"]["category"]["air"], air_ten_op_declared=r["parity"]["drop_declared_concentration"]["operating"]["tenure"]["air"],
             air_cat_op_category=r["parity"]["drop_category_index"]["operating"]["category"]["air"], air_ten_op_category=r["parity"]["drop_category_index"]["operating"]["tenure"]["air"],
             air_cat_op_bureau=r["parity"]["bureau_proxy"]["operating"]["category"]["air"], air_ten_op_bureau=r["parity"]["bureau_proxy"]["operating"]["tenure"]["air"],
             air_cat_60_pbcm=r["parity"]["pbcm_gbm"]["fixed60"]["category"]["air"], air_ten_60_pbcm=r["parity"]["pbcm_gbm"]["fixed60"]["tenure"]["air"],
             air_cat_60_both=r["parity"]["drop_both"]["fixed60"]["category"]["air"], air_ten_60_both=r["parity"]["drop_both"]["fixed60"]["tenure"]["air"],
             air_cat_60_bureau=r["parity"]["bureau_proxy"]["fixed60"]["category"]["air"],
             L0=r["pricing"]["reference_shortcut"]["L0"], tau0=r["pricing"]["reference_shortcut"]["tau0"],
             pricing_iterations=r["pricing"]["tier_inputs"]["iterations"],
             cap_binds=int(r["pricing"]["cap"]["binds"]),
             book_share=r["population"]["operating_book_share"],
             mon_thr_a2=r["monitoring"]["frozen"]["thr_a2"], mon_thr_cum=r["monitoring"]["frozen"]["thr_cum"], mon_kappa=r["monitoring"]["frozen"]["kappa"],
             mon_event_rate=r["monitoring"]["algorithm2"]["event_rate"],
             mon_a2_precision=r["monitoring"]["algorithm2"]["precision"], mon_a2_recall=r["monitoring"]["algorithm2"]["recall"],
             mon_a2_false_alarm=r["monitoring"]["algorithm2"]["false_alarm_rate"], mon_a2_lead=r["monitoring"]["algorithm2"]["median_lead_weeks"],
             mon_a2_flags_per_100=r["monitoring"]["algorithm2"]["flags_per_100_advances"], mon_a2_f1=r["monitoring"]["algorithm2"]["f1"],
             mon_cum_precision=r["monitoring"]["baseline_cumulative_progress"]["precision"], mon_cum_recall=r["monitoring"]["baseline_cumulative_progress"]["recall"],
             mon_cum_lead=r["monitoring"]["baseline_cumulative_progress"]["median_lead_weeks"], mon_cum_f1=r["monitoring"]["baseline_cumulative_progress"]["f1"],
             mon_cum_false_alarm=r["monitoring"]["baseline_cumulative_progress"]["false_alarm_rate"],
             mon_proj_precision=r["monitoring"]["baseline_projected_payoff"]["precision"], mon_proj_recall=r["monitoring"]["baseline_projected_payoff"]["recall"],
             mon_proj_lead=r["monitoring"]["baseline_projected_payoff"]["median_lead_weeks"], mon_proj_f1=r["monitoring"]["baseline_projected_payoff"]["f1"],
             mon_proj_false_alarm=r["monitoring"]["baseline_projected_payoff"]["false_alarm_rate"],
             ref_miss_test=1 - r["population"]["reference_target_repay_rate_test"],
             ref_miss_no_shift=r["pricing"]["no_shift_paired"]["reference_target_miss_rate_no_shift"])
    for rule in ("tier_expected_loss", "reference_shortcut", "linear_rule"):
        for row in r["pricing"]["tiers"][rule]:
            d[f"margin_{rule}_{row['tier']}"] = row.get("margin_equal", float("nan"))
            d[f"margincap_{rule}_{row['tier']}"] = row.get("margin_capital", float("nan"))
            if rule == "tier_expected_loss":
                d[f"miss_own_{row['tier']}"] = row.get("missed_own_window", float("nan"))
                d[f"unpaidshare_{row['tier']}"] = row.get("unpaid_share_52w", float("nan"))
                d[f"unpaidany_{row['tier']}"] = row.get("unpaid_any_52w", float("nan"))
                d[f"fr_{row['tier']}"] = row.get("fr", float("nan"))
                d[f"n_{row['tier']}"] = row.get("n", 0)
        d[f"flat_pass_{rule}"] = int(r["pricing"]["flatness"][rule]["pass_"])
        d[f"flat_spread_{rule}"] = r["pricing"]["flatness"][rule]["spread"]
        d[f"portfolio_equal_{rule}"] = r["pricing"]["portfolio_operating_book"][rule]["margin_equal"]
        d[f"portfolio_capital_{rule}"] = r["pricing"]["portfolio_operating_book"][rule]["margin_capital"]
        d[f"portfolio_all_equal_{rule}"] = r["pricing"]["portfolio_if_funded_all_tiers"][rule]["margin_equal"]
    d["portfolio_capital_no_shift"] = r["pricing"]["no_shift_paired"]["portfolio_operating_book"]["margin_capital"]
    d["portfolio_equal_no_shift"] = r["pricing"]["no_shift_paired"]["portfolio_operating_book"]["margin_equal"]
    for row in r["pricing"]["no_shift_paired"]["tiers"]:
        d[f"margin_noshift_{row['tier']}"] = row.get("margin_equal", float("nan"))
    for s in r["pricing"]["stress"]:
        d[f"stress_{s['kind']}_{s['value']}"] = s["margin_capital"]
    return d


def main():
    t0 = time.time()
    _selftest()
    print("Primary run (seed 42) ...", flush=True)
    primary, state = run_seed(SEED, extras=True)
    primary["_seed"] = SEED
    print(f"  done {time.time() - t0:.0f}s; frozen thr {state['thr_A']:.3f}; tier FR {np.round(state['tier_pricing']['FR'], 3)}", flush=True)
    print("Robustness sweeps ...", flush=True)
    rob = robustness(SEED, state["thr_A"])
    print("Worked example ...", flush=True)
    wx = worked_example(state, np.random.default_rng(SEED))
    scen = dict(base_fr_wx=scenario_mc(np.random.default_rng(SEED), factor=wx["factor_rate_tier"]),
                adverse_fr_wx=scenario_mc(np.random.default_rng(SEED), factor=wx["factor_rate_tier"], adverse=True),
                base_fr121=scenario_mc(np.random.default_rng(SEED), factor=1.21),
                adverse_fr121=scenario_mc(np.random.default_rng(SEED), factor=1.21, adverse=True))
    print("Multi-seed (10 seeds) ...", flush=True)
    runs = [summarise_run(primary)]
    per_seed = {SEED: None}
    for s in SEEDS[1:]:
        r, _ = run_seed(s)
        r["_seed"] = s
        runs.append(summarise_run(r))
        print(f"  seed {s} done {time.time() - t0:.0f}s", flush=True)

    def ms(k):
        v = np.array([x.get(k, np.nan) for x in runs], dtype=float)
        return dict(mean=round(float(np.nanmean(v)), 5), sd=round(float(np.nanstd(v, ddof=1)), 5),
                    min=round(float(np.nanmin(v)), 5), max=round(float(np.nanmax(v)), 5), n=int(np.isfinite(v).sum()))
    keys = set().union(*[r.keys() for r in runs]) - {"seed"}
    multi = {k: ms(k) for k in sorted(keys)}
    multi["n_seeds"] = len(runs); multi["seeds"] = [r["seed"] for r in runs]
    results = dict(artifact_version=ARTIFACT_VERSION, seed=SEED, primary=primary, robustness=rob, scenarios_supplement=scen,
                   table1_apr=dict(fr125=apr_fixture(), fr110=apr_fixture(factor=wx["factor_rate_tier"])),
                   multiseed=multi,
                   constants=dict(N_SELLERS=N_SELLERS, HOLDBACK=HOLDBACK, ALPHA_LABEL=ALPHA_LABEL, FR_LABEL=FR_LABEL, LABEL_MULT=LABEL_MULT,
                                  RHO_THICK=RHO_THICK, RHO_THIN=RHO_THIN, COST_OF_FUNDS=COST_OF_FUNDS, PLATFORM_SHARE=PLATFORM_SHARE,
                                  TARGET_MARGIN=TARGET_MARGIN, FR_MARKET_CAP=FR_MARKET_CAP, OOT_HAZARD_SHIFT=OOT_HAZARD_SHIFT,
                                  BAD_BUDGET=BAD_BUDGET, FLAT_TOL=FLAT_TOL, FLAT_MIN_N=FLAT_MIN_N, MIN_LEAD_WEEKS=MIN_LEAD_WEEKS,
                                  TIERS=[list(t) for t in TIERS]))
    def _np(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not serialisable: {type(o)}")
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=_np)
    with open(os.path.join(OUT_DIR, "worked_example.json"), "w") as f:
        json.dump(wx, f, indent=2, default=_np)
    print("Figures ...", flush=True)
    figures(primary, state, rob, multi, runs)
    print(f"Done in {time.time() - t0:.0f}s. Wrote results.json, worked_example.json, figures and source_data/ to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
