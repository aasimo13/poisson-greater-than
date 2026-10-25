"""Plot the matched-rate resource benchmark from its saved numerical results.

Run from the repository root: python3 src/figure_resources.py
Curves connect the tested settings; adaptive points are not a continuous sweep.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def draw(data, out_dir):
    backgrounds = sorted({s["background_rate"] for s in data["scenarios"]})
    if len(backgrounds) != 2:
        raise ValueError("The publication figure expects the complete two-background benchmark")
    styles = {
        "fixed_threshold": ("Threshold, one count", "#64748b", "s"),
        "fixed_banded": ("Nominal banded, one count", "#2563eb", "o"),
        "fixed_banded_bg1": ("Retuned banded, one count", "#7c3aed", "D"),
        "four_serial_bit_slots": ("Four bit slots", "#059669", "^"),
    }
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#334155", "text.color": "#0f172a",
                         "xtick.color": "#475569", "ytick.color": "#475569",
                         "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.3), sharey=True)
        legend_items = {}
        for ax, bg in zip(axes, backgrounds):
            scenarios = sorted([s for s in data["scenarios"] if s["background_rate"] == bg],
                               key=lambda s: s["readout_pause"])
            rate = scenarios[0]["peak_signal_rate_per_transmitter"]
            x = np.array([rate * s["readout_pause"] for s in scenarios])
            deadline = rate * scenarios[0]["common_wall_deadline"]
            ax.axhspan(deadline, 235, color="#f8fafc", zorder=0)
            ax.axhline(deadline, color="#94a3b8", lw=1.2, ls="--")
            ax.text(1, deadline + 4, "Common deadline", fontsize=8.5, color="#64748b")
            for name, (label, color, marker) in styles.items():
                rows = [next((d for d in s["fixed"] if d["name"] == name), None) for s in scenarios]
                y = np.array([np.nan if d is None else rate * d["uniform_mean_wall_time"] for d in rows])
                if not np.any(np.isfinite(y) & (y <= 235)):
                    continue
                line, = ax.plot(x, y, color=color, lw=1.7, alpha=.95, label=label)
                legend_items[label] = line
                for xx, yy, d in zip(x, y, rows):
                    if d is not None and yy <= 235:
                        ax.plot(xx, yy, marker=marker, ms=6, color=color,
                                markerfacecolor=color if d["meets_wall_deadline"] else "white",
                                markeredgewidth=1.5)
            adaptive_y = []
            for s in scenarios:
                feasible = [a for a in s["adaptive"] if a.get("feasible") and a.get("meets_wall_deadline")]
                adaptive_y.append(rate * min(a["uniform_mean_wall_time"] for a in feasible)
                                  if feasible else np.nan)
            if np.any(np.isfinite(adaptive_y)):
                line, = ax.plot(x, adaptive_y, color="#ea580c", marker="o", lw=2.3, ms=6,
                                label="Adaptive, best tested grid")
                legend_items["Adaptive, best tested grid"] = line
            ax.set_title("No background" if bg == 0 else "Background rate = 1% of source peak",
                         loc="left", fontsize=12, fontweight="bold", pad=13)
            ax.set_xlim(-2, 52)
            ax.set_ylim(0, 235)
            ax.set_xticks([0, 10, 25, 50])
            ax.set_xlabel(r"Readout pause, $R\tau$", labelpad=10)
            ax.grid(axis="y", color="#e2e8f0", lw=.7)
            ax.set_axisbelow(True)
            if bg:
                ax.text(.02, .035, "Nominal banded shape needs >279 active units;\nretuning restores a feasible one-count design.",
                        transform=ax.transAxes, fontsize=8.4, color="#475569", va="bottom")
        axes[0].set_ylabel(r"Mean decision time, $R\,\mathbb{E}[T]$", labelpad=10)
        fig.suptitle("Readout cost changes which tested design is fastest", x=.08, ha="left",
                     y=.98, fontsize=17, fontweight="bold")
        fig.text(.08, .922, "Same source peak rate, 1% error limit for every input pair, and a common deadline of 200 units.",
                 fontsize=10, color="#475569")
        fig.legend(legend_items.values(), legend_items.keys(), loc="lower center",
                   bbox_to_anchor=(.5, .035), ncol=3, frameon=False, fontsize=9)
        fig.text(.08, .012, "Filled markers meet the deadline; open markers miss it. Mean time uses uniform inputs. Lines connect tested settings.",
                 fontsize=8, color="#64748b")
        fig.subplots_adjust(left=.08, right=.98, top=.85, bottom=.23, wspace=.12)
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "svg"):
            target = out_dir / ("fig_resources." + ext)
            fig.savefig(target, dpi=220, facecolor="white")
            print(target)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/resource_benchmark.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    draw(json.loads(args.input.read_text()), args.out_dir)


if __name__ == "__main__":
    main()
