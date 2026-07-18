"""The one-figure summary: seconds of training to match baseline quality.

For each recipe, solve the fitted power law L(t) = L_target for
L_target = baseline's loss after 80 s. Renders results/headline.png.
"""
import os

import numpy as np

from analyze import load, fit_variant, power_law

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# dataviz palette — colors match the entities' slots in scaling.png
COLORS = {"baseline": "#2a78d6", "lr_3e3": "#eda100",
          "combo": "#008300", "tr_best": "#eb6834", "tr2_best": "#1baf7a"}
LABELS = {"baseline": "Baseline GPT-Nano",
          "lr_3e3": "Best single change (lr 3e-3)",
          "combo": "Greedy autoresearch stack",
          "tr_best": "Trust-region search (v1)",
          "tr2_best": "Trust-region v2 (verified accepts)"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = ("#0b0b0b", "#52514e", "#898781",
                                         "#e1e0d9", "#c3c2b7", "#fcfcfb")


def time_to_match(params, target, t_max_search=500.0):
    L_inf, A, alpha = params
    if target <= L_inf:
        return float("inf")
    return float((A / (target - L_inf)) ** (1.0 / alpha))


def main():
    by_variant, _ = load(os.path.join(RESULTS, "results.jsonl"))
    target = by_variant["baseline"][80.0]

    rows = []
    for v in ["baseline", "lr_3e3", "combo", "tr_best", "tr2_best"]:
        if v not in by_variant or len(by_variant[v]) < 3:
            continue
        budgets = sorted(by_variant[v])
        params, _ = fit_variant(budgets, [by_variant[v][b] for b in budgets])
        t = 80.0 if v == "baseline" else time_to_match(params, target)
        rows.append((v, t))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows.sort(key=lambda r: -r[1])
    fig, ax = plt.subplots(figsize=(8.5, 0.9 + 0.75 * len(rows)), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ys = range(len(rows))
    for y, (v, t) in zip(ys, rows):
        ax.barh(y, t, height=0.55, color=COLORS[v], zorder=3)
        mult = 80.0 / t
        note = "%.0f s" % t if v == "baseline" else "%.0f s   %.1f× less training" % (t, mult)
        ax.text(t + 1.2, y, note, va="center", ha="left", fontsize=10.5,
                color=INK, fontweight="bold" if v != "baseline" else "normal")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([LABELS[v] for v, _ in rows], fontsize=10.5, color=INK)
    ax.invert_yaxis()

    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xlabel("seconds of training to reach the same val loss (%.2f nats/char)"
                  % target, color=INK2, fontsize=10)
    ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title("What autoresearch bought: same quality, ~10× less training time\n"
                 "GPT-Nano, char-level Shakespeare, M1 Pro — times from fitted "
                 "scaling laws L(t) = L∞ + A·t⁻ᵅ",
                 loc="left", color=INK, fontsize=11.5)
    fig.tight_layout()
    out = os.path.join(RESULTS, "headline.png")
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", out)
    for v, t in rows:
        print("%-10s %6.1f s   x%.1f" % (v, t, 80.0 / t))


if __name__ == "__main__":
    main()
