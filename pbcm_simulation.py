"""
PBCM — Platform Behavioral Credit Model: synthetic evaluation suite
====================================================================
Accompanies "AI-Driven Underwriting for Embedded Merchant Cash Advances:
A Financial Platform Engineering Framework for E-Commerce Marketplaces" (v2).

Everything here is synthetic. No platform, lender, customer, or company data is
used or required. The data-generating process (DGP) is stated in full in
Section 6.1 of the paper and in README.md; every result in the paper's Tables
4-9 and Figures 2-7, and every value in the API worked example (Appendix A),
is written by this script into results.json / worked_example.json.

Experiments
  E1  Population + repayment DGP; label construction (two label definitions)
  E2  Underwriting comparison: PBCM (gradient boosting on 10 behavioural
      features) vs bureau-proxy score vs logistic-on-behavioural vs combined;
      in-time (5-fold) and out-of-time (next origination cohort) validation;
      calibration; thin-file vs thick-file subgroups
  E3  Sensitivity: how the PBCM-vs-bureau gap changes with bureau signal
      strength (rho sweep) and with the thin-file share
  E4  Reject inference: training only on incumbent-approved sellers
  E5  Pricing: expected-loss factor rate FR*(E) vs the linear rule; realised
      lender margin by tier; eligibility floor E_min implied by a market cap
  E6  Capital-deployment scenario as a distribution (not two point cases)
  E7  Repayment-velocity monitoring (Algorithm 2): precision / recall /
      lead time vs eventual default; threshold sweep
  E8  Feature ablation (seller-declared and category features) and
      approval-rate parity by category and tenure group
  E9  Multi-seed robustness (seeds 42-51) for the headline metrics
  WX  Worked example: the representative seller of Section 5, scored by the
      trained model, with a consistent repayment schedule for Appendix A

Dependencies: numpy, scikit-learn, matplotlib (see requirements.txt).
XGBoost / LightGBM are drop-in alternatives for the boosting learner; the
artifact uses scikit-learn's HistGradientBoostingClassifier so that a single
`pip install -r requirements.txt` suffices.

Run:  python pbcm_simulation.py          (~2-3 minutes on a laptop)
"""

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

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42

# ----------------------------------------------------------------------
# Product terms and DGP constants (all stated in the paper, Section 6.1)
# ----------------------------------------------------------------------
N_SELLERS = 12_000            # per origination cohort
WEEKS_HIST = 13               # 90-day feature window
WEEKS_POST = 52               # post-origination horizon
HOLDBACK = 0.15               # fixed holdback rate
ALPHA_LABEL = 0.15            # advance rate used to generate labels (policy-neutral)
FR_LABEL = 1.25               # factor rate used to generate labels
LABEL_MULT = 1.5              # "full repayment within 1.5x estimated window"

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

RHO_THICK = 0.55              # bureau-score correlation with latent health, thick file
RHO_THIN = 0.15               # ... thin file
THIN_EXTRA_SHARE = 0.25       # sellers with tenure >= 12 who are still thin-file

# Pricing parameters (Section 4.3)
COST_OF_FUNDS = 0.08          # annualised lender funding cost
PLATFORM_SHARE = 0.18         # platform share of the fee
TARGET_MARGIN = 0.05          # lender target net margin on principal
FR_MARKET_CAP = 1.35          # ceiling the market will bear for platform MCA
FR_BASE_LIN, GAMMA_LIN = 1.15, 0.30   # the v1 linear rule, kept for comparison


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ----------------------------------------------------------------------
# E1. Population and repayment data-generating process
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
    # 13-week history
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
    # within-category percentile: share of same-category sellers with a WORSE (higher) return rate
    ret_pct = np.zeros(n)
    for c in range(len(CAT_NAMES)):
        m = cat_idx == c
        r = ret_rate[m]
        order = r.argsort().argsort()          # rank 0 = lowest return rate
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

    # post-origination path: 52 weeks, GBM-like drift + a permanent shock process
    start_week = gmv90 / WEEKS_HIST * (1 + trend / 2)
    mu = 0.003 * h + 0.02 * trend - 0.002
    eps = rng.normal(0, 1, (n, WEEKS_POST))
    logpath = np.cumsum(mu[:, None] + 0.06 * eps, axis=1)
    post = start_week[:, None] * np.exp(logpath)
    hazard = hazard_scale * 0.004 * np.exp(-0.9 * h) * cat_haz * (1 - 0.3 * conc_true)
    shock_draw = rng.random((n, WEEKS_POST)) < hazard[:, None]
    shock_week = np.where(shock_draw.any(1), shock_draw.argmax(1), -1)
    severity = rng.uniform(0.05, 0.60, n)      # remaining share of GMV after the shock
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


def repay_path(post, principal, factor):
    """Cumulative repayment (capped at obligation) under a fixed holdback."""
    obligation = principal * factor
    cum = np.cumsum(HOLDBACK * post, axis=1)
    return np.minimum(cum, obligation[:, None]), obligation


def mean_term_weeks(pop, principal, factor):
    """Mean weeks to full repayment (52 if never) under a given policy."""
    cum, T = repay_path(pop["post"], principal, factor)
    full = cum >= T[:, None] - 1e-9
    term = np.where(full.any(1), full.argmax(1) + 1, WEEKS_POST)
    return float(term.mean())


def make_labels(pop):
    A0 = ALPHA_LABEL * pop["gmv90"]
    cum, T = repay_path(pop["post"], A0, FR_LABEL)
    # estimated window is a constant of the product terms because A0 is proportional to GMV90
    est_window = FR_LABEL * ALPHA_LABEL * WEEKS_HIST / HOLDBACK          # = 16.25 weeks
    horizon = int(math.ceil(LABEL_MULT * est_window))                    # = 25 weeks
    y_window = (cum[:, horizon - 1] >= T - 1e-9).astype(int)             # label 1 (paper's definition)
    y_52 = (cum[:, -1] >= T - 1e-9).astype(int)                          # label 2 (write-off at 52 weeks)
    lgd = 1 - cum[:, -1] / T                                             # unpaid share of obligation at 52w
    return dict(y_window=y_window, y_52=y_52, lgd=lgd, est_window=est_window, horizon=horizon)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def ks_stat(y, p):
    order = np.argsort(p)
    y = y[order]
    P, N = y.sum(), len(y) - y.sum()
    cdf_pos = np.cumsum(y) / P
    cdf_neg = np.cumsum(1 - y) / N
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] + (1e-12 if i == bins - 1 else 0))
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def approval_metrics(y, p, approve_rate=0.60, bad_budget=0.08):
    n = len(y)
    order = np.argsort(-p)
    k = int(approve_rate * n)
    bad_at_rate = 1 - y[order[:k]].mean()
    # largest approval share whose cumulative bad rate stays within the budget
    cum_bad = np.cumsum(1 - y[order]) / np.arange(1, n + 1)
    ok = np.where(cum_bad <= bad_budget)[0]
    appr_at_budget = (ok.max() + 1) / n if len(ok) else 0.0
    return float(bad_at_rate), float(appr_at_budget)


def evaluate(y, p, thin=None):
    out = dict(auc=float(roc_auc_score(y, p)), ks=ks_stat(y, p),
               brier=float(brier_score_loss(y, p)), ece=ece(y, p))
    b, a = approval_metrics(y, p)
    out["bad_rate_at_60pct_approval"] = b
    out["approval_at_8pct_bad_budget"] = a
    if thin is not None:
        out["auc_thin_file"] = float(roc_auc_score(y[thin], p[thin]))
        out["auc_thick_file"] = float(roc_auc_score(y[~thin], p[~thin]))
    return out


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
def gbm(seed):
    base = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                          max_leaf_nodes=31, l2_regularization=1.0,
                                          random_state=seed)
    return CalibratedClassifierCV(base, method="sigmoid", cv=5)     # Platt scaling


def logit():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def fit_predict_oot(model, Xtr, ytr, Xte):
    model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1]


def bureau_X(pop):
    return np.column_stack([pop["bureau"], np.log(pop["tenure"])])


# ----------------------------------------------------------------------
# E5. Pricing
# ----------------------------------------------------------------------
def fr_star(E, lgd_mean, term_weeks):
    """Break-even-plus-margin factor rate. Lender cash-in = A[FR(1-l) - s(FR-1)],
    cost = A(1 + c*tau/52), requirement cash-in - cost >= m*A, with l = (1-E)*LGD."""
    loss = (1 - E) * lgd_mean
    num = 1 + COST_OF_FUNDS * term_weeks / 52 + TARGET_MARGIN - PLATFORM_SHARE
    den = (1 - loss) - PLATFORM_SHARE
    return num / den


def alpha_lin(E):
    return 0.10 + 0.15 * (E - 0.50) / 0.50


def fr_lin(E):
    return FR_BASE_LIN + GAMMA_LIN * (1 - E)


def e_min_for_cap(lgd_mean, term_weeks, cap=FR_MARKET_CAP):
    """Smallest calibrated E whose break-even factor rate is at or below the market cap."""
    for E in np.linspace(0.30, 1.0, 7001):
        if fr_star(E, lgd_mean, term_weeks) <= cap:
            return float(E)
    return 1.0


def realised_margin(pop, E, approve_mask, rule):
    """Realised lender net margin on principal, by seller, for a pricing rule."""
    lgd_mean = None
    A = np.clip(alpha_lin(np.clip(E, 0.5, 1.0)), 0.10, 0.25) * pop["gmv90"]
    FR = rule(E)
    cum, T = repay_path(pop["post"], A, FR)
    repaid = cum[:, -1]
    # term actually taken (weeks to full repayment, else 52)
    full = cum >= T[:, None] - 1e-9
    term = np.where(full.any(1), full.argmax(1) + 1, WEEKS_POST)
    fee_collected = np.maximum(repaid - A, 0)          # fee is the last thing collected
    platform_take = PLATFORM_SHARE * fee_collected
    funding = A * COST_OF_FUNDS * term / 52
    net = repaid - A - funding - platform_take
    return net / A, A, FR, term


# ----------------------------------------------------------------------
# E7. Monitoring (Algorithm 2)
# ----------------------------------------------------------------------
def algorithm2_flags(post, gmv90, principal, factor, thr=0.70):
    """Apply the repayment-velocity rule weekly; return first week a 'review' or
    'escalate' flag is raised (or -1), and first 'watch' week."""
    n, W = post.shape
    expected = HOLDBACK * gmv90 / WEEKS_HIST
    hold = HOLDBACK * post
    cum = np.cumsum(hold, axis=1)
    T = principal * factor
    outstanding = np.maximum(T[:, None] - cum, 0)
    vel = hold / expected[:, None]
    active = cum < T[:, None] - 1e-9                    # advance still open
    first_rev = np.full(n, -1)
    first_watch = np.full(n, -1)
    for w in range(2, W):
        last3 = vel[:, w - 2:w + 1]
        slow_all = (last3 < thr).all(1) & active[:, w]
        slow_any = (last3 < thr).any(1) & active[:, w]
        rev = slow_all & (first_rev < 0)
        first_rev[rev] = w
        wt = slow_any & ~slow_all & (first_watch < 0)
        first_watch[wt] = w
    # severity split at flag time
    esc = np.zeros(n, bool)
    m = first_rev >= 0
    esc[m] = outstanding[m, first_rev[m]] > 0.80 * principal[m]
    return first_rev, first_watch, esc


# ----------------------------------------------------------------------
# E8. Fairness helpers
# ----------------------------------------------------------------------
def approval_parity(p, groups, approve_rate=0.60):
    thr = np.quantile(p, 1 - approve_rate)
    appr = p >= thr
    rates = {g: float(appr[groups == g].mean()) for g in np.unique(groups)}
    air = min(rates.values()) / max(rates.values())
    return rates, float(air)


# ----------------------------------------------------------------------
# One full run for a seed
# ----------------------------------------------------------------------
def run_seed(seed, hazard_shift=1.20, verbose=False):
    rng = np.random.default_rng(seed)
    train = gen_population(rng)
    test = gen_population(rng, hazard_scale=hazard_shift)      # out-of-time cohort, tighter conditions
    ltr, lte = make_labels(train), make_labels(test)
    ytr, yte = ltr["y_window"], lte["y_window"]

    res = {}
    # --- E2 out-of-time comparison
    models = {
        "bureau_proxy": (logit(), bureau_X(train), bureau_X(test)),
        "logit_behavioral": (logit(), train["X"], test["X"]),
        "pbcm_gbm": (gbm(seed), train["X"], test["X"]),
        "pbcm_plus_bureau": (gbm(seed), np.column_stack([train["X"], train["bureau"]]),
                             np.column_stack([test["X"], test["bureau"]])),
    }
    preds = {}
    for name, (m, Xa, Xb) in models.items():
        preds[name] = fit_predict_oot(m, Xa, ytr, Xb)
        res[name] = evaluate(yte, preds[name], test["thin"])
    # label-2 robustness for the two headline models
    res["label52_auc"] = {k: float(roc_auc_score(lte["y_52"], preds[k])) for k in ("bureau_proxy", "pbcm_gbm")}
    # in-time 5-fold for PBCM (to show OOT degradation honestly)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    p_cv = cross_val_predict(gbm(seed), train["X"], ytr, cv=skf, method="predict_proba")[:, 1]
    res["pbcm_gbm_in_time_cv"] = evaluate(ytr, p_cv)

    # --- E4 reject inference: train only on sellers the bureau incumbent would approve
    pb_tr = fit_predict_oot(logit(), bureau_X(train), ytr, bureau_X(train))
    keep = pb_tr >= np.quantile(pb_tr, 0.50)
    p_ri = fit_predict_oot(gbm(seed), train["X"][keep], ytr[keep], test["X"])
    res["reject_inference"] = dict(auc_full_training=res["pbcm_gbm"]["auc"],
                                   auc_approved_only_training=float(roc_auc_score(yte, p_ri)),
                                   approved_share_used=float(keep.mean()),
                                   bad_rate_at_60_full=res["pbcm_gbm"]["bad_rate_at_60pct_approval"],
                                   bad_rate_at_60_approved_only=approval_metrics(yte, p_ri)[0])

    # --- E8 ablation + parity
    drop = [FEATURES.index("category_risk_index"), FEATURES.index("platform_concentration")]
    keep_cols = [i for i in range(len(FEATURES)) if i not in drop]
    p_abl = fit_predict_oot(gbm(seed), train["X"][:, keep_cols], ytr, test["X"][:, keep_cols])
    res["ablation_drop_declared_and_category"] = evaluate(yte, p_abl, test["thin"])
    cat_names = np.array(CAT_NAMES)[test["cat"]]
    ten_grp = np.where(test["tenure"] < 12, "tenure<12", "tenure>=12")
    par = {}
    for name, p in (("bureau_proxy", preds["bureau_proxy"]), ("pbcm_gbm", preds["pbcm_gbm"]), ("pbcm_ablated", p_abl)):
        rc, airc = approval_parity(p, cat_names)
        rt, airt = approval_parity(p, ten_grp)
        # outcome-conditioned view: approval rate among sellers who actually repay (equal opportunity)
        good = yte == 1
        thr = np.quantile(p, 0.40)
        eo = {g: float((p[good & (cat_names == g)] >= thr).mean()) for g in CAT_NAMES}
        par[name] = dict(approval_by_category=rc, air_category=airc,
                         approval_by_tenure=rt, air_tenure=airt,
                         tpr_by_category=eo, tpr_ratio_category=min(eo.values()) / max(eo.values()))
    res["parity"] = par
    # feature importance via permutation on the OOT set (AUC drop)
    m_full = gbm(seed).fit(train["X"], ytr)
    base_auc = roc_auc_score(yte, m_full.predict_proba(test["X"])[:, 1])
    imp = {}
    prng = np.random.default_rng(seed + 1000)
    for j, f in enumerate(FEATURES):
        Xp = test["X"].copy()
        Xp[:, j] = prng.permutation(Xp[:, j])
        imp[f] = float(base_auc - roc_auc_score(yte, m_full.predict_proba(Xp)[:, 1]))
    res["permutation_importance_auc_drop"] = imp

    # --- E5 pricing on the OOT cohort using PBCM scores
    E = preds["pbcm_gbm"]
    # Loss parameters are estimated on the TRAINING cohort and paired with the label E predicts:
    # LGD_w = expected unpaid share of the obligation at 52 weeks, given the seller missed the
    # 1.5x window (includes the zeros of slow-but-full payers); term = mean weeks to full repayment.
    win_def_tr = ytr == 0
    lgd_mean = float(ltr["lgd"][win_def_tr].mean()) if win_def_tr.any() else 0.0
    term_nom = mean_term_weeks(train, ALPHA_LABEL * train["gmv90"], FR_LABEL)
    e_min = e_min_for_cap(lgd_mean, term_nom)
    defaulters = yte == 0
    lgd52_test = float(lte["lgd"][lte["y_52"] == 0].mean())
    tiers = [(0.90, 1.001), (0.80, 0.90), (0.70, 0.80), (0.60, 0.70), (0.50, 0.60)]
    tier_rows = []
    approve = E >= 0.50
    net_star, A_star, FR_s, term_s = realised_margin(test, E, approve, lambda e: fr_star(e, lgd_mean, term_nom))
    net_lin, _, FR_l, _ = realised_margin(test, E, approve, fr_lin)
    for lo, hi in tiers:
        m = (E >= lo) & (E < hi)
        if m.sum() == 0:
            continue
        tier_rows.append(dict(
            band=f"{lo:.2f}-{min(hi, 1.0):.2f}", n=int(m.sum()), share_of_scored=float(m.mean()),
            realised_bad_rate=float(1 - yte[m].mean()), realised_writeoff_rate_52w=float(1 - lte["y_52"][m].mean()),
            mean_unpaid_share_if_window_default=float(lte["lgd"][m & defaulters].mean()) if (m & defaulters).any() else 0.0,
            fr_star_lo=float(fr_star(lo, lgd_mean, term_nom)), fr_star_hi=float(fr_star(min(hi, 1.0), lgd_mean, term_nom)),
            fr_linear_lo=float(fr_lin(min(hi, 1.0))), fr_linear_hi=float(fr_lin(lo)),
            alpha_lo=float(alpha_lin(lo)), alpha_hi=float(alpha_lin(min(hi, 1.0))),
            realised_net_margin_fr_star=float(net_star[m].mean()),
            realised_net_margin_fr_linear=float(net_lin[m].mean()),
            mean_term_weeks=float(term_s[m].mean()),
        ))
    res["pricing"] = dict(lgd_mean_given_default=lgd_mean, nominal_term_weeks=term_nom,
                          est_window_weeks=float(ltr["est_window"]), lgd_52w_given_writeoff_test=lgd52_test,
                          e_min_at_market_cap=e_min, market_cap=FR_MARKET_CAP,
                          portfolio_net_margin_fr_star=float(net_star[approve].mean()),
                          portfolio_net_margin_fr_linear=float(net_lin[approve].mean()),
                          tiers=tier_rows,
                          calibration_bins=calib_bins(yte, E))

    # --- E7 monitoring on approved OOT sellers
    A = np.clip(alpha_lin(np.clip(E, 0.5, 1.0)), 0.10, 0.25) * test["gmv90"]
    mon = {}
    for thr in (0.50, 0.60, 0.70, 0.80, 0.90):
        fr_, fw_, esc_ = algorithm2_flags(test["post"][approve], test["gmv90"][approve], A[approve],
                                          FR_s[approve], thr=thr)
        ydef = yte[approve] == 0                       # missed the 1.5x window (the paper's label)
        ywo = lte["y_52"][approve] == 0               # still unpaid at 52 weeks (write-off)
        flagged = fr_ >= 0
        lead = np.where(flagged & ydef, lte["horizon"] - fr_, np.nan)
        mon[f"thr_{thr:.2f}"] = dict(
            precision=float((flagged & ydef).sum() / max(flagged.sum(), 1)),
            recall=float((flagged & ydef).sum() / max(ydef.sum(), 1)),
            false_alarm_rate_nondefaulters=float((flagged & ~ydef).mean() / max((~ydef).mean(), 1e-9)),
            median_lead_weeks_before_label=float(np.nanmedian(lead)) if np.isfinite(lead).any() else None,
            escalate_share_of_flags=float(esc_[flagged].mean()) if flagged.any() else 0.0,
            recall_writeoffs=float((flagged & ywo).sum() / max(ywo.sum(), 1)),
            n_approved=int(approve.sum()), window_default_rate_approved=float(ydef.mean()),
            writeoff_rate_approved=float(ywo.mean()))
    res["monitoring"] = mon

    # --- population descriptives
    res["population"] = dict(
        n_train=int(N_SELLERS), n_test=int(N_SELLERS),
        base_rate_label_window_train=float(ytr.mean()), base_rate_label_window_test=float(yte.mean()),
        base_rate_label_52_test=float(lte["y_52"].mean()),
        shock_share_test=float(test["shocked"].mean()), thin_file_share=float(test["thin"].mean()),
        est_window_weeks=float(ltr["est_window"]), label_horizon_weeks=int(ltr["horizon"]),
        median_gmv90=float(np.median(test["gmv90"])), median_tenure=float(np.median(test["tenure"])),
        misreport_share_platform_conc=float(test["misreport"].mean()),
        feature_medians={f: float(np.median(test["X"][:, j])) for j, f in enumerate(FEATURES)},
    )
    return res, dict(train=train, test=test, ltr=ltr, lte=lte, preds=preds, E=E, lgd_mean=lgd_mean,
                     term_nom=term_nom, model=m_full, seed=seed)


def calib_bins(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] + (1e-12 if i == bins - 1 else 0))
        if m.sum():
            rows.append(dict(lo=float(edges[i]), hi=float(edges[i + 1]), n=int(m.sum()),
                             mean_pred=float(p[m].mean()), observed=float(y[m].mean())))
    return rows


# ----------------------------------------------------------------------
# E3. Sensitivity sweeps (primary seed)
# ----------------------------------------------------------------------
def sensitivity(seed):
    out = dict(rho_thick_sweep=[], thin_share_sweep=[])
    for rho in (0.30, 0.40, 0.55, 0.70, 0.85):
        rng = np.random.default_rng(seed)
        tr = gen_population(rng, rho_thick=rho)
        te = gen_population(rng, hazard_scale=1.20, rho_thick=rho)
        ytr, yte = make_labels(tr)["y_window"], make_labels(te)["y_window"]
        pb = fit_predict_oot(logit(), bureau_X(tr), ytr, bureau_X(te))
        pp = fit_predict_oot(gbm(seed), tr["X"], ytr, te["X"])
        pc = fit_predict_oot(gbm(seed), np.column_stack([tr["X"], tr["bureau"]]), ytr,
                             np.column_stack([te["X"], te["bureau"]]))
        out["rho_thick_sweep"].append(dict(rho_thick=rho, auc_bureau=float(roc_auc_score(yte, pb)),
                                           auc_pbcm=float(roc_auc_score(yte, pp)),
                                           auc_combined=float(roc_auc_score(yte, pc)),
                                           auc_bureau_thick_only=float(roc_auc_score(yte[~te["thin"]], pb[~te["thin"]])),
                                           auc_pbcm_thick_only=float(roc_auc_score(yte[~te["thin"]], pp[~te["thin"]]))))
    for share in (0.0, 0.25, 0.50, 0.75):
        rng = np.random.default_rng(seed)
        tr = gen_population(rng, thin_extra=share)
        te = gen_population(rng, hazard_scale=1.20, thin_extra=share)
        ytr, yte = make_labels(tr)["y_window"], make_labels(te)["y_window"]
        pb = fit_predict_oot(logit(), bureau_X(tr), ytr, bureau_X(te))
        pp = fit_predict_oot(gbm(seed), tr["X"], ytr, te["X"])
        out["thin_share_sweep"].append(dict(thin_extra_share=share, realised_thin_share=float(te["thin"].mean()),
                                            auc_bureau=float(roc_auc_score(yte, pb)), auc_pbcm=float(roc_auc_score(yte, pp))))
    return out


# ----------------------------------------------------------------------
# E6. Deployment scenario as a distribution
# ----------------------------------------------------------------------
def scenario_mc(rng, advance=1000.0, factor=1.21, n=200_000):
    fee = advance * (factor - 1)
    margin = np.clip(rng.normal(0.35, 0.08, n), 0.10, 0.60)
    sell_through = rng.beta(6, 2, n)                    # share of inventory sold in the 12-week window
    hold_months = 3.0
    carry = 0.015 * hold_months                         # carrying cost on unsold inventory
    gross_profit = sell_through * advance * margin / (1 - margin)
    holding_cost = (1 - sell_through) * advance * carry
    net_growth = gross_profit - fee - holding_cost
    net_deficit = -fee * np.ones(n)                     # operational-deficit deployment
    def summarise(x):
        return dict(mean=float(x.mean()), median=float(np.median(x)), p10=float(np.quantile(x, 0.10)),
                    p90=float(np.quantile(x, 0.90)), prob_positive=float((x > 0).mean()),
                    roi_on_advance_median=float(np.median(x) / advance),
                    roi_on_fee_median=float(np.median(x) / fee))
    breakeven_p = fee / (net_growth.mean() + fee) if net_growth.mean() + fee > 0 else None
    mix = [dict(p_growth=p, expected_net=float(p * net_growth.mean() + (1 - p) * net_deficit.mean()))
           for p in (0.0, 0.25, 0.5, 0.75, 1.0)]
    return dict(advance=advance, factor=factor, fee=fee, growth=summarise(net_growth),
                deficit=summarise(net_deficit), breakeven_growth_share=float(breakeven_p) if breakeven_p else None,
                mixture=mix, assumptions=dict(margin="TruncNormal(0.35, 0.08) on [0.10, 0.60]",
                                              sell_through_12w="Beta(6, 2), mean 0.75",
                                              carrying_cost="1.5% per month on unsold inventory for 3 months",
                                              unsold_inventory="carried at cost, no write-down"))


# ----------------------------------------------------------------------
# WX. Worked example for Appendix A
# ----------------------------------------------------------------------
def shapley_sampling(model, x, background, rng, n_samples=400):
    """Sampling-based Shapley values (Strumbelj & Kononenko 2014) for one prediction."""
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
    model, test = state["model"], state["test"]
    x = np.array([15000.0, 0.083, 0.21, 72.0, 0.009, 0.97, 18.0, 0.34, 0.18, 0.85])
    cal = gbm(state["seed"]).fit(state["train"]["X"], state["ltr"]["y_window"])
    E = float(cal.predict_proba(x[None, :])[0, 1])
    E_r = round(E, 3)
    lgd_mean, term_nom = state["lgd_mean"], state["term_nom"]
    e_min = e_min_for_cap(lgd_mean, term_nom)
    alpha = float(alpha_lin(E_r))
    A_max = round(alpha * x[0], 2)
    FR = round(float(fr_star(E_r, lgd_mean, term_nom)), 2)
    FR_l = round(float(fr_lin(E_r)), 2)
    phi = shapley_sampling(cal, x, test["X"][rng.choice(len(test["X"]), 200, replace=False)], rng)
    contrib = sorted(zip(FEATURES, phi.tolist()), key=lambda t: -abs(t[1]))
    # repayment schedule for a $1,000 draw
    draw = 1000.0
    T = round(draw * FR, 2)
    weekly_settle = round(x[0] / WEEKS_HIST, 2)
    weekly_hold = round(HOLDBACK * weekly_settle, 2)
    weeks = math.ceil(T / weekly_hold)
    disb = date(2025, 7, 1)
    first_settle = disb + timedelta(days=6)              # first weekly settlement after disbursement
    schedule = []
    cum = 0.0
    for k in range(weeks):
        d = first_settle + timedelta(weeks=k)
        hb = min(weekly_hold, round(T - cum, 2))
        cum = round(cum + hb, 2)
        schedule.append(dict(settlement_date=d.isoformat(), gross=weekly_settle, holdback=hb,
                             net_payout=round(weekly_settle - hb, 2), cumulative=cum,
                             outstanding=round(T - cum, 2), velocity=round(hb / weekly_hold, 2)))
    payoff = schedule[-1]["settlement_date"]
    ev = schedule[2]                                     # third settlement event shown in the API example
    # deficit scenario: unchanged sales -> velocity 1.0 -> no flag. Add a stressed variant: sales -40% from week 5
    stressed = []
    cum = 0.0
    for k in range(60):
        g = weekly_settle * (0.60 if k >= 4 else 1.0)
        hb = min(round(HOLDBACK * g, 2), round(T - cum, 2))
        if hb <= 0:
            break
        cum = round(cum + hb, 2)
        stressed.append(dict(week=k + 1, gross=round(g, 2), holdback=hb, velocity=round(hb / weekly_hold, 2), outstanding=round(T - cum, 2)))
    # Algorithm 2 on the stressed path
    flag = None
    for k in range(2, len(stressed)):
        v = [stressed[i]["velocity"] for i in (k - 2, k - 1, k)]
        if all(vi < 0.70 for vi in v):
            flag = dict(week=k + 1, decision="escalate" if stressed[k]["outstanding"] > 0.80 * draw else "review")
            break
        if any(vi < 0.70 for vi in v) and flag is None:
            flag = dict(week=k + 1, decision="watch")
            # keep looking for escalation
            flag_watch = flag; flag = None
            for kk in range(k + 1, len(stressed)):
                v2 = [stressed[i]["velocity"] for i in (kk - 2, kk - 1, kk)]
                if all(vi < 0.70 for vi in v2):
                    flag = dict(week=kk + 1, decision="escalate" if stressed[kk]["outstanding"] > 0.80 * draw else "review", first_watch_week=flag_watch["week"])
                    break
            if flag is None:
                flag = flag_watch
            break
    return dict(profile={f: float(v) for f, v in zip(FEATURES, x)}, eligibility_score=E_r,
                e_min_at_market_cap=e_min, risk_tier=("very_low" if E_r >= 0.9 else "low" if E_r >= 0.8 else "moderate" if E_r >= 0.7 else "elevated" if E_r >= 0.6 else "high"),
                alpha=round(alpha, 4), max_advance=A_max, factor_rate_expected_loss=FR, factor_rate_linear_rule_for_comparison=FR_l,
                total_if_max=round(A_max * FR, 2), shapley_contributions=[dict(feature=f, phi=round(v, 4)) for f, v in contrib],
                draw=dict(requested=draw, factor_rate=FR, total_repayment=T, holdback_rate=HOLDBACK,
                          weekly_settlement=weekly_settle, weekly_holdback=weekly_hold, est_repayment_weeks=weeks,
                          disbursement_date=disb.isoformat(), est_payoff_date=payoff,
                          third_event=ev, schedule=schedule),
                max_draw_est_weeks=math.ceil(round(A_max * FR, 2) / weekly_hold),
                stressed_path_flag=flag, stressed_path=stressed[:12])


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def figures(primary, state, sens, scen, multi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    yte = state["lte"]["y_window"]
    # Fig 2: ROC curves
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for name, lab in (("bureau_proxy", "Bureau-proxy score"), ("logit_behavioral", "Logistic, behavioural"),
                      ("pbcm_gbm", "PBCM (gradient boosting)"), ("pbcm_plus_bureau", "PBCM + bureau")):
        fpr, tpr, _ = roc_curve(yte, state["preds"][name])
        ax.plot(fpr, tpr, label=f"{lab} (AUC {primary[name]['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_xlabel("False positive rate (approved sellers who do not repay)"); ax.set_ylabel("True positive rate")
    ax.set_title("E2: Out-of-time ROC, next origination cohort (n = 12,000)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig2_roc_oot.png"), dpi=200); plt.close(fig)
    # Fig 3: subgroup AUC
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    names = ["bureau_proxy", "logit_behavioral", "pbcm_gbm", "pbcm_plus_bureau"]
    x = np.arange(len(names)); w = 0.38
    ax.bar(x - w / 2, [primary[n]["auc_thin_file"] for n in names], w, label="Thin-file sellers")
    ax.bar(x + w / 2, [primary[n]["auc_thick_file"] for n in names], w, label="Thick-file sellers")
    ax.set_xticks(x); ax.set_xticklabels(["Bureau", "Logit-beh.", "PBCM", "PBCM+bureau"]); ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUC (out-of-time)"); ax.set_title("E2: Where the behavioural advantage concentrates"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig3_subgroup_auc.png"), dpi=200); plt.close(fig)
    # Fig 4: rho sweep
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    r = sens["rho_thick_sweep"]
    ax.plot([d["rho_thick"] for d in r], [d["auc_bureau"] for d in r], "o-", label="Bureau-proxy")
    ax.plot([d["rho_thick"] for d in r], [d["auc_pbcm"] for d in r], "s-", label="PBCM")
    ax.plot([d["rho_thick"] for d in r], [d["auc_combined"] for d in r], "^-", label="PBCM + bureau")
    ax.set_xlabel("Bureau signal strength for thick-file sellers (rho)"); ax.set_ylabel("AUC (out-of-time)")
    ax.set_title("E3: The PBCM advantage shrinks as bureau data improves"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig4_rho_sweep.png"), dpi=200); plt.close(fig)
    # Fig 5: calibration
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    cb = primary["pricing"]["calibration_bins"]
    ax.plot([b["mean_pred"] for b in cb], [b["observed"] for b in cb], "o-", label="PBCM (Platt-scaled)")
    ax.plot([0, 1], [0, 1], "k--", lw=0.7, label="Perfect calibration")
    ax.set_xlabel("Predicted repayment probability E(s)"); ax.set_ylabel("Observed repayment rate")
    ax.set_title(f"E2: Reliability diagram, out-of-time (ECE {primary['pbcm_gbm']['ece']:.3f})"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig5_calibration.png"), dpi=200); plt.close(fig)
    # Fig 6: pricing curves
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    Es = np.linspace(0.40, 1.0, 200)
    pr = primary["pricing"]
    ax.plot(Es, fr_star(Es, pr["lgd_mean_given_default"], pr["nominal_term_weeks"]), label="Expected-loss factor rate FR*(E)")
    ax.plot(Es, fr_lin(Es), "--", label="Linear rule 1.15 + 0.30(1 - E) (v1)")
    ax.axhline(FR_MARKET_CAP, color="gray", lw=0.8, ls=":"); ax.text(0.41, FR_MARKET_CAP + 0.005, "market cap 1.35", fontsize=8, color="gray")
    ax.text(0.41, 1.02, "cap does not bind for E >= 0.30 at this LGD", fontsize=8, color="red")
    ax.set_xlabel("Calibrated repayment probability E"); ax.set_ylabel("Factor rate"); ax.set_ylim(1.0, 1.6)
    ax.set_title("E5: Pricing derived from expected loss vs the linear rule"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig6_pricing.png"), dpi=200); plt.close(fig)
    # Fig 7: scenario distribution
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    rng = np.random.default_rng(7)
    margin = np.clip(rng.normal(0.35, 0.08, 100_000), 0.10, 0.60); st = rng.beta(6, 2, 100_000)
    net = st * 1000 * margin / (1 - margin) - 1000 * (scen["factor"] - 1) - (1 - st) * 1000 * 0.045
    ax.hist(net, bins=80, color="#2b6cb0", alpha=0.85)
    ax.axvline(0, color="k", lw=0.8); ax.axvline(-scen["fee"], color="red", ls="--", lw=0.8, label=f"Deficit deployment: -${scen['fee']:.0f}")
    ax.set_xlabel("12-week net gain on a $1,000 advance (USD)"); ax.set_ylabel("Simulated sellers")
    ax.set_title(f"E6: Growth deployment is a distribution (P(net > 0) = {scen['growth']['prob_positive']:.2f})"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig7_scenario.png"), dpi=200); plt.close(fig)
    # Fig 8: monitoring threshold sweep
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    mon = primary["monitoring"]; ths = sorted(mon)
    ax.plot([float(t[4:]) for t in ths], [mon[t]["precision"] for t in ths], "o-", label="Precision (flag -> default)")
    ax.plot([float(t[4:]) for t in ths], [mon[t]["recall"] for t in ths], "s-", label="Recall (defaults flagged)")
    ax.plot([float(t[4:]) for t in ths], [mon[t]["false_alarm_rate_nondefaulters"] for t in ths], "^-", label="False-alarm rate (non-defaulters)")
    ax.set_xlabel("Velocity threshold"); ax.set_ylabel("Rate"); ax.set_ylim(0, 1.05)
    ax.set_title("E7: Algorithm 2 early-warning trade-off"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig8_monitoring.png"), dpi=200); plt.close(fig)


# ----------------------------------------------------------------------
def main():
    t0 = time.time()
    print("Primary run (seed 42) ...")
    primary, state = run_seed(SEED, verbose=True)
    print("Sensitivity sweeps ...")
    sens = sensitivity(SEED)
    print("Scenario Monte Carlo ...")
    scen = scenario_mc(np.random.default_rng(SEED), factor=1.21)
    print("Worked example ...")
    wx = worked_example(state, np.random.default_rng(SEED))
    scen_wx = scenario_mc(np.random.default_rng(SEED), factor=wx["factor_rate_expected_loss"])
    print("Multi-seed robustness (10 seeds) ...")
    runs = []
    for s in range(42, 52):
        r, _ = run_seed(s)
        runs.append(dict(seed=s,
                         auc_bureau=r["bureau_proxy"]["auc"], auc_logit=r["logit_behavioral"]["auc"],
                         auc_pbcm=r["pbcm_gbm"]["auc"], auc_combined=r["pbcm_plus_bureau"]["auc"],
                         auc_pbcm_thin=r["pbcm_gbm"]["auc_thin_file"], auc_bureau_thin=r["bureau_proxy"]["auc_thin_file"],
                         auc_pbcm_thick=r["pbcm_gbm"]["auc_thick_file"], auc_bureau_thick=r["bureau_proxy"]["auc_thick_file"],
                         ks_pbcm=r["pbcm_gbm"]["ks"], ece_pbcm=r["pbcm_gbm"]["ece"], brier_pbcm=r["pbcm_gbm"]["brier"],
                         bad60_bureau=r["bureau_proxy"]["bad_rate_at_60pct_approval"], bad60_pbcm=r["pbcm_gbm"]["bad_rate_at_60pct_approval"],
                         appr8_bureau=r["bureau_proxy"]["approval_at_8pct_bad_budget"], appr8_pbcm=r["pbcm_gbm"]["approval_at_8pct_bad_budget"],
                         auc_ablated=r["ablation_drop_declared_and_category"]["auc"],
                         air_cat_bureau=r["parity"]["bureau_proxy"]["air_category"], air_cat_pbcm=r["parity"]["pbcm_gbm"]["air_category"],
                         air_cat_ablated=r["parity"]["pbcm_ablated"]["air_category"],
                         air_ten_pbcm=r["parity"]["pbcm_gbm"]["air_tenure"], air_ten_bureau=r["parity"]["bureau_proxy"]["air_tenure"],
                         auc_ri=r["reject_inference"]["auc_approved_only_training"],
                         margin_star=r["pricing"]["portfolio_net_margin_fr_star"], margin_lin=r["pricing"]["portfolio_net_margin_fr_linear"],
                         e_min=r["pricing"]["e_min_at_market_cap"], lgd=r["pricing"]["lgd_mean_given_default"],
                         mon_prec=r["monitoring"]["thr_0.70"]["precision"], mon_rec=r["monitoring"]["thr_0.70"]["recall"],
                         mon_lead=r["monitoring"]["thr_0.70"]["median_lead_weeks_before_label"],
                         base_rate=r["population"]["base_rate_label_window_test"]))
    def ms(k):
        v = np.array([x[k] for x in runs], dtype=float)
        return dict(mean=round(float(np.nanmean(v)), 4), sd=round(float(np.nanstd(v, ddof=1)), 4),
                    min=round(float(np.nanmin(v)), 4), max=round(float(np.nanmax(v)), 4))
    multi = {k: ms(k) for k in runs[0] if k != "seed"}
    multi["n_seeds"] = len(runs)
    multi["seeds"] = [r["seed"] for r in runs]

    results = dict(seed=SEED, primary=primary, sensitivity=sens, scenario_fr121=scen,
                   scenario_at_worked_example_fr=scen_wx, multiseed=multi,
                   constants=dict(N_SELLERS=N_SELLERS, HOLDBACK=HOLDBACK, ALPHA_LABEL=ALPHA_LABEL, FR_LABEL=FR_LABEL,
                                  LABEL_MULT=LABEL_MULT, RHO_THICK=RHO_THICK, RHO_THIN=RHO_THIN,
                                  COST_OF_FUNDS=COST_OF_FUNDS, PLATFORM_SHARE=PLATFORM_SHARE, TARGET_MARGIN=TARGET_MARGIN,
                                  FR_MARKET_CAP=FR_MARKET_CAP, OOT_HAZARD_SHIFT=1.20))
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(OUT_DIR, "worked_example.json"), "w") as f:
        json.dump(wx, f, indent=2)
    print("Figures ...")
    figures(primary, state, sens, scen_wx, multi)
    print(f"Done in {time.time() - t0:.0f}s. Wrote results.json, worked_example.json and figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
