"""The counterexample figure: when low-fidelity verdicts don't transfer.

Left panel:  single-change sweep — improvement according to a 10 s run vs
             measured improvement at 80 s (both vs baseline).
Right panel: inside the search loops, at campaign step sizes — surrogate-
             predicted improvement vs replicated 80 s truth, with the
             measurement-noise band. The v2 iteration-4 false positive
             (predicted +0.008, measured −0.004) is the counterexample.

Renders results/counterexample.png and prints a markdown table.
"""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

INK, INK2, MUTED, GRID, AXIS, SURFACE = ("#0b0b0b", "#52514e", "#898781",
                                         "#e1e0d9", "#c3c2b7", "#fcfcfb")
C_AGREE, C_FN, C_FP = "#2a78d6", "#eb6834", "#e34948"  # + marker shapes as
# secondary encoding (circle / diamond / X) so color never carries alone


def sweep_points():
    L = defaultdict(dict)
    with open(os.path.join(RESULTS, "results.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            L[r["variant"]][r["budget_s"]] = r["final_val_loss"]
    base10, base80 = L["baseline"][10.0], L["baseline"][80.0]
    singles = ["lr_1e3", "lr_3e3", "cosine", "dropout0", "wtie",
               "llama_mlp", "rotary", "bs128"]
    return [(v, base10 - L[v][10.0], base80 - L[v][80.0]) for v in singles]


# (label, cheap-predicted improvement, replicated-80s improvement)
CAMPAIGN = [
    ("v1 it1", 0.0807, 0.0089),
    ("v1 it2", -0.0229, 0.0139),
    ("v1 it3", -0.0065, -0.0132),
    ("v1 it4", -0.0082, -0.0114),
    ("v1 it5", -0.0031, -0.0073),
    ("v1 it6", -0.0172, 0.0001),
    ("v2 it3", 0.0006, 0.0027),
    ("v2 it4", 0.0079, -0.0035),
]


def category(x, y):
    if x < 0 and y > 0:
        return "fn"
    if x > 0 and y < 0:
        return "fp"
    return "agree"


def draw_points(ax, pts, label_all=False, ms=9):
    style = {"agree": dict(marker="o", color=C_AGREE),
             "fn": dict(marker="D", color=C_FN),
             "fp": dict(marker="X", color=C_FP)}
    for name, x, y in pts:
        s = style[category(x, y)]
        ax.plot(x, y, s["marker"], color=s["color"], ms=ms,
                mec=SURFACE, mew=1.0, zorder=4)


def style_axes(ax, xlabel, ylabel):
    ax.axhline(0, color=AXIS, lw=1.0, zorder=1)
    ax.axvline(0, color=AXIS, lw=1.0, zorder=1)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    for s in ax.spines.values():
        s.set_color(AXIS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9.5)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)
    ax.set_facecolor(SURFACE)


def main():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    # ---- left: sweep transfer ----
    pts = sweep_points()
    lim = 0.33
    axL.fill_betweenx([0, 0.27], -lim, 0, color=C_FN, alpha=0.06, zorder=0)
    axL.plot([-lim, lim], [-lim, lim], ls="--", color=MUTED, lw=1, zorder=1)
    draw_points(axL, pts)
    axL.set_xlim(-lim, lim)
    axL.set_ylim(-0.19, 0.27)
    ann = {"rotary": (8, 10), "llama_mlp": (8, 8), "cosine": (8, -12),
           "lr_3e3": (-30, 10), "wtie": (8, -4)}
    for name, x, y in pts:
        if name in ann:
            axL.annotate(name, (x, y), textcoords="offset points",
                         xytext=ann[name], fontsize=9, color=INK)
    axL.text(-0.315, 0.245, "FALSE NEGATIVES\ncheap judge discards a winner",
             fontsize=9, color=C_FN, fontweight="bold", va="top")
    axL.text(0.06, -0.175, "dashed = perfect transfer", fontsize=8.5, color=MUTED)
    style_axes(axL, "improvement according to a 10 s run  (nats, + = better)",
               "measured improvement at 80 s  (nats, + = better)")
    axL.set_title("Single changes: the cheap judge inverts\non the changes that matter most",
                  loc="left", color=INK, fontsize=10.5)

    # ---- right: campaign step sizes ----
    axR.axhspan(-0.019, 0.019, color=MUTED, alpha=0.13, zorder=0)
    axR.text(-0.029, -0.0182, "replicate noise (±2σ):\n1-seed accepts live here",
             fontsize=8.5, color=INK2, ha="left", va="bottom")
    axR.fill_between([0, 0.095], -0.032, 0, color=C_FP, alpha=0.06, zorder=0)
    draw_points(axR, CAMPAIGN, ms=10)
    axR.set_xlim(-0.032, 0.095)
    axR.set_ylim(-0.032, 0.032)
    axR.annotate("v2 it4 — THE COUNTEREXAMPLE\nsurrogate: +0.008 better\nreplicated truth: −0.004 worse\n(rejected by the CI test)",
                 (0.0079, -0.0035), textcoords="offset points", xytext=(14, -34),
                 fontsize=9, color=C_FP, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C_FP, lw=1.2))
    axR.annotate("v1 it1: 9× over-promise\n(accepted anyway)", (0.0807, 0.0089),
                 textcoords="offset points", xytext=(-118, -34), fontsize=9, color=INK,
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))
    axR.annotate("v1 it2: predicted worse,\nwas better", (-0.0229, 0.0139),
                 textcoords="offset points", xytext=(4, 14), fontsize=9, color=C_FN)
    axR.text(0.09, -0.0305, "FALSE POSITIVES", fontsize=9, color=C_FP,
             fontweight="bold", ha="right")
    style_axes(axR, "surrogate-predicted improvement  (nats, + = better)",
               "replicated 80 s truth  (nats, + = better)")
    axR.set_title("Inside the loops: at real step sizes the cheap\nsignal errs in both directions",
                  loc="left", color=INK, fontsize=10.5)

    # shared legend
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=C_AGREE, ms=8, mec=SURFACE, label="verdicts agree"),
               Line2D([], [], marker="D", ls="", color=C_FN, ms=8, mec=SURFACE, label="false negative (cheap says worse, truth better)"),
               Line2D([], [], marker="X", ls="", color=C_FP, ms=9, mec=SURFACE, label="false positive (cheap says better, truth worse)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("The cheap judge vs the real judge — why autoresearch needs distrust logic",
                 x=0.02, ha="left", color=INK, fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0.045, 1, 0.94])
    out = os.path.join(RESULTS, "counterexample.png")
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", out)

    print("\n| change / decision | cheap verdict | replicated 80 s truth | outcome |")
    print("|---|---|---|---|")
    for name, x, y in sweep_points() + CAMPAIGN:
        cat = {"agree": "transfer OK", "fn": "**false negative**",
               "fp": "**false positive**"}[category(x, y)]
        print("| %s | %+.3f | %+.3f | %s |" % (name, x, y, cat))


if __name__ == "__main__":
    main()
