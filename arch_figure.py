"""Renders Figure 1 (proposed MCA lifecycle and integration layers). Illustration only; computes no
results. The integration shown is a PROPOSAL: the artifact contains no service implementation."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.dirname(os.path.abspath(__file__))
NAVY, LIGHT, ORANGE, GREEN, GREY = "#1f3a5f", "#e8eef6", "#c05621", "#2f855a", "#6b7280"


def box(ax, x, y, w, h, text, fc=LIGHT, ec=NAVY, tc="black", bold=False, fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, p, q, color=NAVY, ls="-", lw=1.3):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color=color, lw=lw, ls=ls))


fig, ax = plt.subplots(figsize=(12, 6.9))
ax.set_xlim(0, 12); ax.set_ylim(0, 6.9); ax.axis("off")
ax.text(6, 6.62, "Proposed embedded-MCA lifecycle: five layers, seven endpoints, one authoritative service per state",
        ha="center", fontsize=11.5, fontweight="bold", color=NAVY)

# ---- left column: data -> features -> inference -> offer
ax.text(0.3, 6.0, "L1  Platform data sources", fontsize=9, color=GREY, fontweight="bold")
srcs = ["Settlement /\npayouts (net)", "Orders &\nfulfilment", "Returns &\ndisputes", "Seller profile\n& category"]
for i, t in enumerate(srcs):
    box(ax, 0.3 + i * 1.4, 5.2, 1.25, 0.65, t)
ax.text(0.3, 4.85, "L2  Feature and explanation pipeline (batch)", fontsize=9, color=GREY, fontweight="bold")
box(ax, 0.3, 3.95, 5.45, 0.75,
    "POST /internal/v1/features/refresh   nightly + sales-event trigger\nx(s): 10 features on the trailing 13 weeks (as-of peer cohort for percentiles)\nscore E(s), tier, principal reasons pre-computed and cached (24 h TTL, versioned)", fs=7.3)
for i in range(4):
    arrow(ax, (0.925 + i * 1.4, 5.2), (0.925 + i * 1.4, 4.72))
ax.text(0.3, 3.6, "L3  Scoring, sizing, pricing (stateless reads)", fontsize=9, color=GREY, fontweight="bold")
box(ax, 0.3, 2.7, 2.6, 0.75, "Calibrated gradient-boosted\nclassifier  ->  E(s)", fc=NAVY, tc="white", bold=True)
box(ax, 3.15, 2.7, 2.6, 0.75, "Sizing alpha(E) x GMV90 (capped)\nTier price FR_t from expected loss\nunder the offered terms", fs=7.3)
arrow(ax, (1.6, 3.95), (1.6, 3.47)); arrow(ax, (2.9, 3.075), (3.15, 3.075))
box(ax, 0.3, 1.6, 5.45, 0.8,
    "GET /v1/advance/eligibility/{seller_id}  (cached decision, decision_version, offer expiry)\noffer: max advance, factor rate, principal reasons\ndecline: principal reasons (+ optional, separately labelled suggestions)", fs=7.0)
arrow(ax, (3.0, 2.7), (3.0, 2.42))

# ---- right column: lifecycle + settlement
ax.text(6.35, 6.0, "L4  Advance lifecycle service (authoritative for offer -> advance state)", fontsize=9, color=GREY, fontweight="bold")
box(ax, 6.35, 5.2, 2.6, 0.65, "POST /v1/advance/apply\natomic offer lock, idempotency key", fs=7.3)
box(ax, 9.1, 5.2, 2.6, 0.65, "POST /v1/advance/disburse\nledger entry, schedule start", fs=7.3)
box(ax, 6.35, 3.95, 2.6, 0.75, "GET /v1/advance/status/{id}\nstate, schedule, flags", fs=7.3)
box(ax, 9.1, 3.95, 2.6, 0.75, "GET /v1/advance/risk/portfolio\nexposure by tier, open flags,\ncalibration and parity drift", fs=7.3)
arrow(ax, (8.95, 5.525), (9.1, 5.525))
arrow(ax, (5.75, 2.05), (6.05, 2.05), color=GREEN)
ax.plot([6.05, 6.05], [2.05, 5.525], color=GREEN, lw=1.3)
arrow(ax, (6.05, 5.525), (6.35, 5.525), color=GREEN)
ax.text(6.1, 3.3, "seller accepts", fontsize=7.5, color=GREEN, rotation=90, va="center")

ax.text(6.35, 3.6, "L5  Settlement integration (authoritative for money movement)", fontsize=9, color=GREY, fontweight="bold")
box(ax, 6.35, 2.6, 5.35, 0.85,
    "Payout engine calls  POST /v1/advance/repayment/event  per settlement\n(idempotent by settlement_id; out-of-order and duplicate events rejected)\nholdback = 15% x net payout; final payment capped at balance; reversals netted; daily reconciliation", fc="#fdf3ec", ec=ORANGE, fs=7.0)
arrow(ax, (10.4, 5.2), (10.4, 4.72), color=ORANGE)
arrow(ax, (10.4, 3.95), (10.4, 3.47), color=ORANGE)
arrow(ax, (7.65, 3.95), (7.65, 3.47), color=ORANGE)
box(ax, 6.35, 1.6, 5.35, 0.8,
    "Weekly monitor (scheduler, not per event): velocity and projected-payoff rules on weekly windows;\nmissing-payout check by clock; watch / review / escalate\nrealised outcomes become the next cohort's labels", fc="#fdf3ec", ec=ORANGE, fs=7.0)
arrow(ax, (9.0, 2.6), (9.0, 2.42), color=ORANGE)
ax.plot([6.35, 6.2, 6.2], [1.9, 1.9, 1.2], color=GREY, lw=1.1, ls="--")
ax.plot([6.2, 1.6, 1.6], [1.2, 1.2, 1.7], color=GREY, lw=1.1, ls="--")
arrow(ax, (1.6, 1.2), (1.6, 1.7), color=GREY, ls="--", lw=1.1)
ax.text(3.9, 1.28, "realised outcomes -> labels -> retraining (with monitored approval expansion)", fontsize=7.3, color=GREY)
box(ax, 0.3, 0.25, 11.4, 0.7,
    "All endpoints: REST, OAuth 2.0 bearer tokens, JSON; every decision logged with inputs, model and decision version, thresholds, reasons.\n"
    "Latency budgets (targets, not measurements): eligibility read < 300 ms; repayment event < 100 ms; apply < 500 ms.  Proposal: no service is implemented in the artifact.",
    fc="white", ec=GREY, fs=7.2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_architecture.png"), dpi=200)
print("wrote fig1_architecture.png")
