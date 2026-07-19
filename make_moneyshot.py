"""The money shot: best verified loss vs cumulative search compute, 3 arms.

Arms (identical proposer, acceptance rule, start point, and budget scale):
  SF  — single-fidelity control: every candidate straight to replicated 80 s
  v1  — multi-fidelity trust region, uncertified accepts (1 seed, 0.002 margin)
  v2  — multi-fidelity trust region, certified accepts (CI + skip-confirm)

x: cumulative training seconds the search consumed (screens, replicates,
   confirms — everything except post-search polls/sweeps, excluded for all).
y: the *replicated truth* value of the arm's incumbent at that point (never
   the loop's own 1-seed beliefs).
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# replicated truth values (see README verdict table)
TRUTH_COMBO = 1.6615    # n=7
TRUTH_TR_BEST = 1.6327  # n=5 (v1 final)
TRUTH_TR2_BEST = 1.6489  # n=3 (v2 final)

INK, INK2, MUTED, GRID, AXIS, SURFACE = ("#0b0b0b", "#52514e", "#898781",
                                         "#e1e0d9", "#c3c2b7", "#fcfcfb")
COLORS = {"sf": "#2a78d6", "v1": "#eb6834", "v2": "#008300"}


def runs(path, exclude_prefixes=()):
    out = []
    with open(os.path.join(RESULTS, path)) as f:
        for line in f:
            r = json.loads(line)
            if any(r["variant"].startswith(p) for p in exclude_prefixes):
                continue
            out.append((r["variant"], r["budget_s"], r["final_val_loss"]))
    return out


def cum_at_last(rs, name):
    """Cumulative seconds after the last run named `name`."""
    c, at = 0.0, None
    for v, b, _ in rs:
        c += b
        if v == name:
            at = c
    return at


def mean_reps(path, name):
    vals = [l for v, b, l in runs(path) if v == name and b == 80.0]
    return (float(np.mean(vals)), len(vals)) if vals else (None, 0)


def main():
    # ---- v1: accepts at it1 (-> v1_it1 truth) and it2 (-> tr_best truth) ----
    v1 = runs("tr_runs.jsonl")
    v1_total = sum(b for _, b, _ in v1)
    v1_it1_truth, n1 = mean_reps("replicates.jsonl", "v1_it1")
    v1_curve = [(0, TRUTH_COMBO),
                (cum_at_last(v1, "tr_it1_confirm"), v1_it1_truth),
                (cum_at_last(v1, "tr_it2_confirm"), TRUTH_TR_BEST),
                (v1_total, TRUTH_TR_BEST)]

    # ---- v2: single accept at it5 ----
    v2 = runs("tr2_runs.jsonl", exclude_prefixes=("tr2_poll_",))
    v2_total = sum(b for _, b, _ in v2)
    v2_curve = [(0, TRUTH_COMBO),
                (cum_at_last(v2, "tr2_it5_confirm"), TRUTH_TR2_BEST),
                (v2_total, TRUTH_TR2_BEST)]

    # ---- SF: accepts from its state history; truth = each incumbent's own
    # replicate pool (confirm reps + fresh reps while it reigned) ----
    sf = runs("sf_runs.jsonl")
    sf_total = sum(b for _, b, _ in sf)
    state = json.load(open(os.path.join(RESULTS, "sf_state.json")))
    accept_iters = [h["iter"] for h in state["history"] if h["accepted"]]

    # truth pool per accepted incumbent: its confirm pair + every sf_inc_fresh
    # replicate logged while it reigned
    pools, reigning = {}, None
    for v, b, l in sf:
        if v.startswith("sf_it"):
            pools.setdefault(v, []).append(l)
            if int(v[5:]) in accept_iters:
                reigning = v
        elif v == "sf_inc_fresh" and reigning:
            pools[reigning].append(l)

    sf_curve = [(0, TRUTH_COMBO)]
    for it in accept_iters:
        name = "sf_it%d" % it
        sf_curve.append((cum_at_last(sf, name), float(np.mean(pools[name]))))
    sf_curve.append((sf_total, sf_curve[-1][1]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.0, 5.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    series = [
        ("sf", "single-fidelity (every look costs 80 s ×2)", sf_curve, "o"),
        ("v1", "multi-fidelity, uncertified (v1)", v1_curve, "s"),
        ("v2", "multi-fidelity, certified (v2)", v2_curve, "D"),
    ]
    for key, label, curve, marker in series:
        xs = [p[0] for p in curve]
        ys = [p[1] for p in curve]
        ax.step(xs, ys, where="post", color=COLORS[key], lw=2, zorder=3)
        ax.plot(xs[1:-1], ys[1:-1], marker, color=COLORS[key], ms=7,
                mec=SURFACE, mew=1.0, zorder=4)
        ax.annotate("%s   %.4f" % (label, ys[-1]), (xs[-1], ys[-1]),
                    textcoords="offset points", xytext=(6, -3), fontsize=9,
                    color=COLORS[key], fontweight="bold", va="center")

    ax.axhline(TRUTH_COMBO, color=AXIS, lw=1, ls=":")
    ax.text(30, TRUTH_COMBO + 0.0006, "start: greedy stack (combo), replicated 1.6615",
            fontsize=8.5, color=INK2, va="bottom")
    ax.set_xlim(0, 3350)
    ax.set_xlabel("training seconds consumed by the search (screens + confirms + replicates)",
                  color=INK2, fontsize=10)
    ax.set_ylabel("replicated val loss @ 80 s of current incumbent", color=INK2, fontsize=10)
    ax.grid(True, color=GRID, lw=0.7)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title("Search efficiency, judged by replicated truth — same proposer and start.\n"
                 "SF vs v2 isolate the fidelity schedule (same CI bar); v1 is the uncertified reference",
                 loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    out = os.path.join(RESULTS, "moneyshot.png")
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", out)
    for key, label, curve, _ in series:
        print("%-4s total %5ds  final %.4f  (points: %s)" %
              (key, curve[-1][0], curve[-1][1],
               ", ".join("%d:%.4f" % (t, y) for t, y in curve)))
    print("v1_it1 truth: %.4f (n=%d)" % (v1_it1_truth, n1))


if __name__ == "__main__":
    main()
