"""Analytic sensitivity of the tier expected-loss price to the loss and term inputs.
Reads the tier inputs estimated under the offered terms (results.json -> primary.pricing.tier_inputs)
and re-evaluates the closed form at multiples of the estimated unpaid share and term, reporting
whether the assumed market cap binds on the feasible domain. No simulation is run.
Writes pricing_sensitivity.json."""
import json
import os

from pbcm_simulation import (COST_OF_FUNDS, FR_MARKET_CAP, PLATFORM_SHARE, TARGET_MARGIN, TIERS,
                             cap_feasibility, fr_star)

OUT = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(os.path.join(OUT, "results.json")))
tp = res["primary"]["pricing"]["tier_inputs"]
rows = []
for lmult in (1.0, 2.0, 4.0, 8.0):
    for tmult in (1.0, 1.5):
        ell = [min(e * lmult, 0.95) for e in tp["ell"]]
        tau = [t * tmult for t in tp["tau"]]
        FR = [float(fr_star(e, t)) for e, t in zip(ell, tau)]
        rows.append(dict(unpaid_share_multiple=lmult, term_multiple=tmult,
                         ell_by_tier={TIERS[i][2]: round(ell[i], 4) for i in range(len(TIERS))},
                         tau_by_tier={TIERS[i][2]: round(tau[i], 1) for i in range(len(TIERS))},
                         fr_by_tier={TIERS[i][2]: round(FR[i], 3) for i in range(len(TIERS))},
                         cap=cap_feasibility(FR)))
out = dict(parameters=dict(cost_of_funds=COST_OF_FUNDS, platform_share=PLATFORM_SHARE,
                           target_margin=TARGET_MARGIN, market_cap=FR_MARKET_CAP),
           estimated_inputs=dict(ell=tp["ell"], tau=tp["tau"], FR=tp["FR"]), rows=rows)
json.dump(out, open(os.path.join(OUT, "pricing_sensitivity.json"), "w"), indent=2)
for r in rows:
    print(r["unpaid_share_multiple"], r["term_multiple"], r["fr_by_tier"], "cap binds:", r["cap"]["binds"])
