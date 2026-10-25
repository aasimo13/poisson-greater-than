"""Headline figure: where the 16 pair means sit, and what the decoder says, for both designs.

x axis is sqrt(detected photons): Poisson shot noise has about the same width everywhere on
this scale, so 'distinguishable' looks like 'a fixed gap apart'.
Usage: python3 src/figure_bands.py [outfile_stem]   (default figures/fig_bands)
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from core import labels, pair_means, lp_minimax, budget_for_shape

for f in font_manager.findSystemFonts():
    if "Inter" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Inter", "font.size": 11})

INK = "#1f2328"; MUTED = "#6b7280"; GRID = "#e5e7eb"
TRUE_C = "#2563eb"; FALSE_C = "#ea580c"
TRUE_BG = "#dbeafe"; FALSE_BG = "#ffedd5"

K = 4
LAB = labels(K)
designs = [
    ("Best known design (reordered levels, four alternating decoder bands)",
     np.array([98.2976, 4.6013, 0.0, 28.7852]), np.array([28.7852, 0.0, 4.6013, 98.2976])),
    ("Best possible single threshold design",
     np.array([1.5, 1.0, 0.5, 0.0]), np.array([0.0, 0.5, 1.0, 1.5])),
]

fig, axes = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True,
                         gridspec_kw={"hspace": 0.55})
XMAX = np.sqrt(335)
for ax, (title, x, y) in zip(axes, designs):
    P, s = budget_for_shape(x, y)
    xs, ys = s * x, s * y
    m = pair_means(xs, ys)
    err, dec = lp_minimax(m, LAB, return_decoder=True)
    if "threshold" in title:
        # the actual single threshold rule: say true iff N < T (randomized at T)
        from scipy.stats import poisson
        Lt = m[LAB == 1].max(); Lf = m[LAB == 0].min()
        nn = np.arange(len(dec))
        Ft = poisson.cdf(nn - 1, Lt); pt = poisson.pmf(nn, Lt)
        Ff = poisson.cdf(nn - 1, Lf); pf = poisson.pmf(nn, Lf)
        # error on true = 1 - Ft - g pt, on false = Ff + g pf; equalize: g = (1 - Ft - Ff)/(pt + pf)
        g_all = np.clip((1 - Ft - Ff) / (pt + pf + 1e-300), 0, 1)
        worst = np.maximum(1 - Ft - g_all * pt, Ff + g_all * pf)
        T = int(np.argmin(worst)); g = g_all[T]
        print("threshold rule: T =", T, "gamma =", round(g, 3), "worst error", worst[T])
        dec = np.where(nn < T, 1.0, 0.0); dec[T] = g
    n = np.arange(len(dec) - 1)
    # shade decoder regions: P(say true) per count n
    for k in n:
        lo, hi = np.sqrt(max(k - 0.5, 0)), np.sqrt(k + 0.5)
        if hi > XMAX:
            break
        p = dec[k]
        col = TRUE_BG if p >= 0.5 else FALSE_BG
        ax.axvspan(lo, hi, ymin=0, ymax=1, color=col, lw=0)
    # pair means, stacked when they coincide
    order = np.argsort(m)
    placed = []
    for j in order:
        v = np.sqrt(max(m[j], 0))
        level = sum(1 for u in placed if abs(u - v) < 0.12)
        placed.append(v)
        c = TRUE_C if LAB[j] else FALSE_C
        ax.plot(v, 0.25 + 0.17 * level, "o", ms=8, mfc=c if LAB[j] else "white",
                mec=c, mew=1.8, zorder=3)
    # mark the close calls: neighbors on the count axis whose answers differ
    srt = np.argsort(m)
    for a_, b_ in zip(srt[:-1], srt[1:]):
        if LAB[a_] != LAB[b_] and m[b_] - m[a_] > 1e-6:
            u0, u1 = np.sqrt(max(m[a_], 0)), np.sqrt(m[b_])
            ax.plot([u0, u0, u1, u1], [0.90, 0.95, 0.95, 0.90], color=INK, lw=1.1, zorder=4)
            ax.text((u0 + u1) / 2, 0.98, f"{m[a_]:.0f} vs {m[b_]:.0f}" if m[a_] >= 1 else f"0 vs {m[b_]:.1f}",
                    ha="center", va="bottom", fontsize=9.5, color=INK, zorder=4)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([])
    for sp in ["left", "right", "top"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_title(f"{title}: {P:.1f} photons", loc="left", color=INK, fontsize=12,
                 fontweight="semibold", pad=6)

ticks = [0, 1, 5, 10, 25, 50, 100, 200, 320]
axes[-1].set_xticks(np.sqrt(ticks))
axes[-1].set_xticklabels([str(t) for t in ticks], color=MUTED)
axes[-1].set_xlim(-0.15, XMAX)
axes[-1].set_xlabel("Expected detected photons for the pair (square root scale, so shot noise has equal width everywhere)",
                    color=MUTED, fontsize=10)
axes[0].tick_params(axis="x", colors=MUTED)

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
handles = [Line2D([], [], marker="o", ls="", mfc=TRUE_C, mec=TRUE_C, ms=8, label="Pair where A > B"),
           Line2D([], [], marker="o", ls="", mfc="white", mec=FALSE_C, mew=1.8, ms=8, label="Pair where A ≤ B"),
           Patch(color=TRUE_BG, label="Decoder says A > B"),
           Patch(color=FALSE_BG, label="Decoder says A ≤ B")]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10,
           bbox_to_anchor=(0.5, -0.02), labelcolor=INK)
fig.subplots_adjust(left=0.03, right=0.99, top=0.9, bottom=0.25)
stem = sys.argv[1] if len(sys.argv) > 1 else "figures/fig_bands"
fig.savefig(stem + ".png", dpi=200)
fig.savefig(stem + ".svg")
print("saved", stem)
