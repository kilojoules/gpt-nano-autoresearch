"""Fit scaling laws to sweep results and render the report.

For each variant, fits  L(t) = L_inf + A * t^(-alpha)  to (budget, final val
loss) points, then quantifies each improvement as an *effective time multiplier*:
how much longer the baseline would need to train to match the variant's loss at
the largest budget. Writes results/report.md and results/scaling.png.

Usage: python analyze.py [--results results/results.jsonl] [--plot-top 5]
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from scipy.optimize import curve_fit, brentq

HERE = os.path.dirname(os.path.abspath(__file__))

# dataviz reference palette (light mode), fixed slot order — assigned to
# variants by registry order (color follows the entity, never its rank)
PALETTE = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def power_law(t, L_inf, A, alpha):
    return L_inf + A * np.power(t, -alpha)


def fit_variant(budgets, losses):
    """Fit L(t) = L_inf + A t^-alpha; returns (params, ok)."""
    t = np.asarray(budgets, dtype=float)
    L = np.asarray(losses, dtype=float)
    try:
        p0 = [max(0.5, L.min() - 0.3), (L.max() - L.min()) * t.min() ** 0.5, 0.5]
        params, _ = curve_fit(
            power_law, t, L, p0=p0,
            bounds=([0.0, 1e-6, 0.01], [L.min(), 100.0, 3.0]), maxfev=20000)
        return tuple(params), True
    except Exception:
        # fallback: grid-search L_inf, log-linear fit for A, alpha
        best = None
        for L_inf in np.linspace(0.0, L.min() - 1e-3, 60):
            y = np.log(L - L_inf)
            b, a = np.polyfit(np.log(t), y, 1)
            pred = L_inf + np.exp(a) * t ** b
            sse = float(np.sum((pred - L) ** 2))
            if best is None or sse < best[0]:
                best = (sse, (L_inf, float(np.exp(a)), float(-b)))
        return best[1], False


def load(results_path):
    rows = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    # average over seeds
    agg = defaultdict(list)
    for r in rows:
        agg[(r["variant"], r["budget_s"])].append(r["final_val_loss"])
    by_variant = defaultdict(dict)
    for (v, b), losses in agg.items():
        by_variant[v][b] = float(np.mean(losses))
    meta = {}
    for r in rows:
        meta.setdefault(r["variant"], r)  # first row carries desc/params
    return by_variant, meta


def analyze(results_path, plot_top=5, out_dir=None, plot_include=None):
    out_dir = out_dir or os.path.join(HERE, "results")
    by_variant, meta = load(results_path)

    fits = {}
    for v, pts in by_variant.items():
        budgets = sorted(pts)
        if len(budgets) < 3:
            continue
        params, ok = fit_variant(budgets, [pts[b] for b in budgets])
        fits[v] = {"params": params, "clean_fit": ok, "budgets": budgets,
                   "losses": [pts[b] for b in budgets]}

    t_max = max(max(f["budgets"]) for f in fits.values())
    base = fits.get("baseline")

    # effective time multiplier: baseline time needed to reach variant's loss
    # at the variant's own largest measured budget
    for v, f in fits.items():
        v_tmax = max(f["budgets"])
        f["t_max"] = v_tmax
        f["loss_at_tmax"] = by_variant[v][v_tmax]
        f["speedup"] = None
        if base and v != "baseline":
            target = f["loss_at_tmax"]
            bL, bA, ba = base["params"]

            def g(t):
                return power_law(t, bL, bA, ba) - target
            try:
                if g(v_tmax) > 0:  # baseline hasn't reached target: extrapolate
                    t_eq = brentq(g, v_tmax, 1e7) if g(1e7) < 0 else float("inf")
                else:
                    t_eq = brentq(g, 1e-2, v_tmax)
                f["speedup"] = t_eq / v_tmax
            except Exception:
                pass

    order = sorted(fits, key=lambda v: fits[v]["loss_at_tmax"])
    plot_scaling(fits, order, plot_top, out_dir, plot_include)
    write_report(fits, order, by_variant, meta, t_max, out_dir)
    return fits, order


def plot_scaling(fits, order, plot_top, out_dir, plot_include=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter, NullFormatter

    # choose series: baseline + best performers, capped to palette size
    chosen = list(plot_include) if plot_include else []
    if "baseline" not in chosen:
        chosen.append("baseline")
    for v in order:
        if len(chosen) >= min(plot_top + 1, len(PALETTE)):
            break
        if v not in chosen:
            chosen.append(v)
    # fixed color assignment by name order (stable across re-runs)
    color_of = {v: PALETTE[i] for i, v in enumerate(sorted(chosen))}

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    t_grid = np.geomspace(min(min(f["budgets"]) for f in fits.values()),
                          max(max(f["budgets"]) for f in fits.values()), 200)
    for v in sorted(chosen, key=lambda v: order.index(v)):
        f = fits[v]
        c = color_of[v]
        L_inf, A, alpha = f["params"]
        ax.plot(t_grid, power_law(t_grid, L_inf, A, alpha), color=c, lw=2,
                alpha=0.85, zorder=2)
        ax.plot(f["budgets"], f["losses"], "o", color=c, ms=5.5, mec=SURFACE,
                mew=1.0, zorder=3,
                label="%s  (α=%.2f)" % (v, alpha))

    ax.set_xscale("log")
    ax.set_yscale("log")
    all_budgets = sorted({b for f in fits.values() for b in f["budgets"]})
    ax.set_xticks(all_budgets)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    lo = min(min(f["losses"]) for f in fits.values())
    hi = max(max(f["losses"]) for f in fits.values())
    yticks = [y for y in [1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.2] if lo * 0.97 <= y <= hi * 1.03]
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.grid(True, which="major", color=GRID, lw=0.8)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(MUTED)
    ax.set_xlabel("training-time budget (s, log)", color=INK2, fontsize=10)
    ax.set_ylabel("val loss (nats/char, log)", color=INK2, fontsize=10)
    ax.set_title("GPT Nano: val loss vs training-time budget\n"
                 "points = measured, lines = fitted  L(t) = L∞ + A·t^−α",
                 color=INK, fontsize=11, loc="left")
    leg = ax.legend(frameon=False, fontsize=9, labelcolor=INK2,
                    loc="upper right")
    fig.tight_layout()
    out = os.path.join(out_dir, "scaling.png")
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def write_report(fits, order, by_variant, meta, t_max, out_dir):
    lines = []
    lines.append("# GPT Nano autoresearch: scaling laws for improvements\n")
    lines.append("Each variant trained under wall-clock training-time budgets "
                 "(eval off the clock), char-level tiny Shakespeare, Apple M1 Pro (MPS), 1 seed.\n")
    lines.append("Fitted law: `L(t) = L_inf + A * t^-alpha`. *Effective time "
                 "multiplier* = how much training time the baseline would need "
                 "to match the variant's loss at t=%ds (from the baseline fit; "
                 "extrapolated when beyond the measured range).\n" % t_max)

    all_budgets = sorted({b for f in fits.values() for b in f["budgets"]})
    hdr = "| variant | " + " | ".join("L @ %ds" % b for b in all_budgets) + \
          " | L_inf | A | alpha | eff. time x |"
    sep = "|---" * (len(all_budgets) + 5) + "|"
    lines += [hdr, sep]
    for v in order:
        f = fits[v]
        L_inf, A, alpha = f["params"]
        cells = []
        for b in all_budgets:
            cells.append("%.4f" % by_variant[v][b] if b in by_variant[v] else "—")
        sp = "—" if f["speedup"] is None else (
            "%.2fx" % f["speedup"] if np.isfinite(f["speedup"]) else ">125x")
        star = "" if f["clean_fit"] else "*"
        lines.append("| %s | %s | %.3f%s | %.3f | %.3f | %s |" % (
            v, " | ".join(cells), L_inf, star, A, alpha, sp))
    lines.append("\n`*` = fallback grid fit (curve_fit did not converge).\n")
    lines.append("![scaling](scaling.png)\n")
    lines.append("## Variant descriptions\n")
    for v in order:
        desc = meta.get(v, {}).get("desc", "")
        lines.append("- **%s** — %s" % (v, desc))

    out = os.path.join(out_dir, "report.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=os.path.join(HERE, "results", "results.jsonl"))
    p.add_argument("--plot-top", type=int, default=5)
    args = p.parse_args()
    fits, order = analyze(args.results, args.plot_top)
    for v in order:
        f = fits[v]
        L_inf, A, alpha = f["params"]
        sp = "" if f["speedup"] is None else "  eff.time %.2fx" % f["speedup"]
        print("%-12s L(%d)=%.4f  fit: L_inf=%.3f A=%.2f alpha=%.3f%s" % (
            v, max(f["budgets"]), f["loss_at_tmax"], L_inf, A, alpha, sp))


if __name__ == "__main__":
    main()
