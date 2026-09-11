"""Renders Figure 1 (MCA lifecycle and PBCM integration layers). Illustration only; computes no results."""
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


fig, ax = plt.subplots(figsize=(12, 6.6))
ax.set_xlim(0, 12); ax.set_ylim(0, 6.6); ax.axis("off")
ax.text(6, 6.35, "Embedded MCA lifecycle: five system layers and the seven-endpoint integration surface",
        ha="center", fontsize=12, fontweight="bold", color=NAVY)

# ---- left column: data -> features -> inference -> offer
ax.text(0.3, 5.75, "L1  Platform data sources", fontsize=9, color=GREY, fontweight="bold")
srcs = ["Settlement /\npayouts", "Orders &\nfulfilment", "Returns &\ndisputes", "Seller profile\n& category"]
for i, t in enumerate(srcs):
    box(ax, 0.3 + i * 1.4, 4.95, 1.25, 0.65, t)
ax.text(0.3, 4.6, "L2  Feature pipeline", fontsize=9, color=GREY, fontweight="bold")
box(ax, 0.3, 3.75, 5.45, 0.7,
    "POST /internal/v1/features/refresh\nnightly batch + event trigger (week GMV > 1.5x trailing 4-wk avg)\nfeature store, 24 h TTL  ->  x(s): 10 behavioural features", fs=7.5)
for i in range(4):
    arrow(ax, (0.925 + i * 1.4, 4.95), (0.925 + i * 1.4, 4.47))
ax.text(0.3, 3.4, "L3  PBCM inference and pricing", fontsize=9, color=GREY, fontweight="bold")
box(ax, 0.3, 2.5, 2.6, 0.75, "Calibrated gradient-boosted\nclassifier  ->  E(s)", fc=NAVY, tc="white", bold=True)
box(ax, 3.15, 2.5, 2.6, 0.75, "Sizing alpha(E)\nPricing FR*(E) from expected loss\nFloor E_min", fs=7.5)
arrow(ax, (1.6, 3.75), (1.6, 3.27)); arrow(ax, (2.9, 2.875), (3.15, 2.875))
box(ax, 0.3, 1.5, 5.45, 0.7,
    "GET /v1/advance/eligibility/{seller_id}\noffer (max advance, factor rate, Shapley explanation)\nor improvement hints if E < E_min", fs=7.5)
arrow(ax, (3.0, 2.5), (3.0, 2.22))

# ---- right column: lifecycle + settlement
ax.text(6.35, 5.75, "L4  Advance lifecycle", fontsize=9, color=GREY, fontweight="bold")
box(ax, 6.35, 4.95, 2.6, 0.65, "POST /v1/advance/apply\noffer lock, lender pre-auth", fs=7.5)
box(ax, 9.1, 4.95, 2.6, 0.65, "POST /v1/advance/disburse\nplatform wallet", fs=7.5)
box(ax, 6.35, 3.75, 2.6, 0.7, "GET /v1/advance/status/{id}\nseller and risk-ops view", fs=7.5)
box(ax, 9.1, 3.75, 2.6, 0.7, "GET /v1/advance/risk/portfolio\nexposure, open flags", fs=7.5)
arrow(ax, (8.95, 5.275), (9.1, 5.275))
arrow(ax, (5.75, 1.85), (6.05, 1.85), color=GREEN)
# seller-accepts path drawn as an elbow
ax.plot([6.05, 6.05], [1.85, 5.275], color=GREEN, lw=1.3)
arrow(ax, (6.05, 5.275), (6.35, 5.275), color=GREEN)
ax.text(6.1, 3.0, "seller accepts", fontsize=7.5, color=GREEN, rotation=90, va="center")

ax.text(6.35, 3.4, "L5  Settlement integration", fontsize=9, color=GREY, fontweight="bold")
box(ax, 6.35, 2.5, 5.35, 0.75,
    "Existing payout engine calls  POST /v1/advance/repayment/event  on every settlement\nholdback = 15% x gross payout; balance, velocity, Algorithm 2 flag", fc="#fdf3ec", ec=ORANGE, fs=7.5)
arrow(ax, (10.4, 4.95), (10.4, 4.47), color=ORANGE)   # disburse -> risk view (advance created)
arrow(ax, (10.4, 3.75), (10.4, 3.27), color=ORANGE)   # -> settlement events
arrow(ax, (7.65, 3.75), (7.65, 3.27), color=ORANGE)
box(ax, 6.35, 1.5, 5.35, 0.7,
    "Risk operations: watch / review / escalate\nrepayment outcomes become labels for the next training cohort (data flywheel)", fc="#fdf3ec", ec=ORANGE, fs=7.5)
arrow(ax, (9.0, 2.5), (9.0, 2.22), color=ORANGE)
# flywheel back to L3
ax.plot([6.35, 6.2, 6.2], [1.7, 1.7, 1.0], color=GREY, lw=1.1, ls="--")
ax.plot([6.2, 1.6, 1.6], [1.0, 1.0, 1.5], color=GREY, lw=1.1, ls="--")
arrow(ax, (1.6, 1.0), (1.6, 1.5), color=GREY, ls="--", lw=1.1)
ax.text(3.9, 1.08, "labels -> retraining", fontsize=7.5, color=GREY)
box(ax, 0.3, 0.2, 11.4, 0.55,
    "All endpoints: REST, OAuth 2.0 bearer tokens, JSON.  Every decision is audit-logged with inputs, score, thresholds and explanation.",
    fc="white", ec=GREY, fs=7.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_architecture.png"), dpi=200)
print("wrote fig1_architecture.png")
