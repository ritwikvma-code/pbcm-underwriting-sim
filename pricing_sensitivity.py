"""Analytic sensitivity of the expected-loss factor rate FR*(E) to loss-given-default
and term (Table 6 of the paper). Uses the same closed form as pbcm_simulation.fr_star;
no simulation is run. Writes pricing_sensitivity.json."""
import json, os
from pbcm_simulation import fr_star, COST_OF_FUNDS, PLATFORM_SHARE, TARGET_MARGIN, FR_MARKET_CAP, e_min_for_cap

OUT = os.path.dirname(os.path.abspath(__file__))
rows = []
for lgd in (0.09, 0.20, 0.35, 0.50):
    for term in (16.25, 20.2, 30.0):
        rows.append(dict(lgd=lgd, term_weeks=term,
                         fr_star={f"E={E:.2f}": round(float(fr_star(E, lgd, term)), 3) for E in (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)},
                         e_min_at_cap=round(e_min_for_cap(lgd, term), 3)))
out = dict(parameters=dict(cost_of_funds=COST_OF_FUNDS, platform_share=PLATFORM_SHARE,
                           target_margin=TARGET_MARGIN, market_cap=FR_MARKET_CAP), rows=rows)
json.dump(out, open(os.path.join(OUT, "pricing_sensitivity.json"), "w"), indent=2)
for r in rows:
    print(r)
