"""Ladder chart: max expected signal photons at 1% error versus background counts per window.
Reads results/ladder.json (src/ladder_points.py). Writes figures/fig_ladder.png and .svg.
Usage: python3 src/figure_ladder.py
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = json.load(open("results/ladder.json"))
STYLE = {"threshold": ("One detector, threshold", "#b04a3a", "o"),
         "banded": ("One detector, banded", "#d98c1f", "s"),
         "two_channel": ("Two separable channels", "#2f7f5f", "^"),
         "arrival_time": ("Arrival time, 4 slots", "#3a6ea5", "D")}

fig, ax = plt.subplots(figsize=(7.5, 5.2))
for key, (label, col, mk) in STYLE.items():
    pts = sorted(D["lines"][key], key=lambda p: p["bg"])
    ax.plot([p["bg"] for p in pts], [p["cost"] for p in pts], "-", marker=mk, color=col, label=label, lw=1.8, ms=6)
    f = D["certified_floor"].get(key)
    if f:
        ax.plot([f["bg"]], [f["cost"]], marker="_", color=col, ms=16, mew=2.5, ls="none")
ax.plot([], [], marker="_", color="gray", ms=14, mew=2.5, ls="none", label="Certified floor at zero background\n(arrival time floor is loose)")
ax.set_xscale("symlog", linthresh=0.05, linscale=0.35)
ax.set_yscale("log")
ax.set_xlim(-0.01, 4.5); ax.set_ylim(7, 600)
ax.set_xticks([0, 0.1, 1, 3]); ax.set_xticklabels(["0", "0.1", "1", "3"])
ax.set_yticks([10, 20, 50, 100, 200, 500]); ax.set_yticklabels(["10", "20", "50", "100", "200", "500"]); ax.minorticks_off()
ax.set_xlabel("Background counts per window")
ax.set_ylabel("Max expected photons for 1% error")
ax.set_title("Cost of the A > B comparison, K = 4")
ax.grid(True, which="major", alpha=0.3)
ax.legend(fontsize=8.5, loc="lower right")
fig.tight_layout()
fig.savefig("figures/fig_ladder.png", dpi=150)
fig.savefig("figures/fig_ladder.svg")
print("wrote figures/fig_ladder.png and .svg")
