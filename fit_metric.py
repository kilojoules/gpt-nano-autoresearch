"""fit_metric.py — empirical grounding for the trust-region metric & surrogate.

Reproduces, from results/results.jsonl ONLY (no other inputs, no side effects):

  1. Surrogate calibration: the loop's own 2-point pinned-L_inf power-law
     prediction L(80) from L(10), L(20) for every sweep variant, under three
     pins for L_inf:
       - baseline 3-point fit on t in {10,20,80}  (what the loop's
         refit_L_inf() computes for an incumbent)
       - baseline 4-point fit on t in {10,20,40,80} (= 1.281)
       - 1.407 (combo's round-1 fitted L_inf, the loop's hardcoded prior)
  2. |prediction error| vs behavioral distance |L_v(10) - L_base(10)| (and the
     L2 distance over {10,20}), Spearman rank correlation.
  3. Per-dimension sensitivities of L(80) in the search's transformed
     coordinates -> a proposed diagonal metric D, and boolean-flag
     displacements expressed as 'equivalent log2-lr steps'.

Usage: python fit_metric.py            (expects results/results.jsonl beside it)
Dependencies: stdlib + numpy + scipy.
"""
import json
import math
import os

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "results.jsonl")

BUDGETS = [10.0, 20.0, 40.0, 80.0]
CONFIRM = 80.0
ORDER = ["baseline", "lr_1e3", "lr_3e3", "cosine", "dropout0", "wtie",
         "llama_mlp", "rotary", "bs128", "combo", "tr_best"]


# ---------------------------------------------------------------- surrogate --
def refit_L_inf(points, fallback=1.407):
    """Same fit as trust_region.refit_L_inf: L + A t^-a, L in [0, y.min()]."""
    try:
        t = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        popt, _ = curve_fit(lambda tt, L, A, a: L + A * tt ** -a, t, y,
                            p0=[max(0.1, y.min() - 0.3), 1.0, 0.5],
                            bounds=([0.0, 1e-6, 0.01], [y.min(), 100.0, 3.0]),
                            maxfev=10000)
        return float(popt[0])
    except Exception:
        return fallback


def surrogate_predict(losses, L_inf, T=CONFIRM):
    """Same as trust_region.surrogate_predict: 2-point pinned-L_inf fit."""
    (t1, y1), (t2, y2) = losses
    if y1 - L_inf < 0.02 or y2 - L_inf < 0.02 or y2 >= y1:
        return y2  # degenerate: no headroom or not improving -> flat
    alpha = math.log((y1 - L_inf) / (y2 - L_inf)) / math.log(t2 / t1)
    A = (y1 - L_inf) * (t1 ** alpha)
    return L_inf + A * (T ** -alpha)


# --------------------------------------------------------------------- load --
def load():
    V = {}
    with open(RESULTS) as f:
        for line in f:
            r = json.loads(line)
            V.setdefault(r["variant"], {})[r["budget_s"]] = r["final_val_loss"]
    return {v: d for v, d in V.items() if all(b in d for b in BUDGETS)}


# -------------------------------------------------------------- computation --
def calibration(V, L_inf, tag):
    print("\n--- calibration, pinned L_inf = %.4f  (%s) ---" % (L_inf, tag))
    print("%-10s %8s %8s %8s %8s %8s" % ("variant", "L(10)", "L(20)",
                                         "L(80)", "pred80", "err"))
    errs = {}
    for v in ORDER:
        if v not in V:
            continue
        d = V[v]
        pred = surrogate_predict([(10.0, d[10.0]), (20.0, d[20.0])], L_inf)
        errs[v] = pred - d[80.0]
        print("%-10s %8.4f %8.4f %8.4f %8.4f %+8.4f"
              % (v, d[10.0], d[20.0], d[80.0], pred, errs[v]))
    e = list(errs.values())
    n_neg = sum(1 for x in e if x < 0)
    print("mean bias (pred-actual) = %+.4f   mean|err| = %.4f   "
          "optimistic (err<0) on %d/%d variants"
          % (np.mean(e), np.mean(np.abs(e)), n_neg, len(e)))
    return errs


def distance_test(V, errs, tag):
    base = V["baseline"]
    vs = [v for v in ORDER if v in errs and v != "baseline"]
    d10 = [abs(V[v][10.0] - base[10.0]) for v in vs]
    d2 = [math.hypot(V[v][10.0] - base[10.0], V[v][20.0] - base[20.0])
          for v in vs]
    ae = [abs(errs[v]) for v in vs]
    r1, p1 = spearmanr(d10, ae)
    r2, p2 = spearmanr(d2, ae)
    print("[%s] Spearman |err| vs |dL(10)|: rho=%+.3f (p=%.3f); "
          "vs L2 over {10,20}: rho=%+.3f (p=%.3f); n=%d (small-sample caveat)"
          % (tag, r1, p1, r2, p2, len(vs)))


def sensitivities(V):
    L80 = {v: V[v][80.0] for v in V}
    L10 = {v: V[v][10.0] for v in V}
    print("\n=== 3. per-dimension sensitivities at 80 s "
          "(single-change variants around baseline) ===")
    # lr: baseline 5e-4 -> lr_1e3 1e-3 (Dlog2 = 1) -> lr_3e3 3e-3 (Dlog2 = log2 3)
    s_lr1 = (L80["lr_1e3"] - L80["baseline"]) / 1.0
    s_lr2 = (L80["lr_3e3"] - L80["lr_1e3"]) / math.log2(3.0)
    D_lr = 0.5 * (abs(s_lr1) + abs(s_lr2))
    # batch: 64 -> 128 (Dlog2 = 1)
    s_bs = (L80["bs128"] - L80["baseline"]) / 1.0
    # dropout: 0.1 -> 0.0
    s_do = (L80["dropout0"] - L80["baseline"]) / (-0.1)
    print("d L80 / d log2(lr):    %+.4f (5e-4->1e-3), %+.4f (1e-3->3e-3); "
          "|mean| = %.4f nats/log2-step" % (s_lr1, s_lr2, D_lr))
    print("d L80 / d log2(batch): %+.4f nats/log2-step (64->128)" % s_bs)
    print("d L80 / d dropout:     %+.4f nats/unit  (= %+.4f per 0.05 = one "
          "delta=1 proposal step)" % (s_do, s_do * 0.05))
    print("d L80 / d log2(wd):    NOT IDENTIFIABLE from the sweep "
          "(no single-change weight_decay variant)")
    print("\nproposed diagonal D (nats per unit step, transformed coords):")
    print("  D[log2 lr]    = %.4f" % D_lr)
    print("  D[log2 batch] = %.4f" % abs(s_bs))
    print("  D[dropout/0.05] = %.4f   (dropout in proposal units of 0.05)"
          % abs(s_do * 0.05))
    print("  D[log2 wd]    = n/a (assume <= D[log2 batch] until measured)")
    print("\nboolean-flag displacements and exchange rates:")
    print("%-10s %10s %10s %26s" % ("flag", "dL(10)", "dL(80)",
                                    "equiv log2-lr steps @80s"))
    for f in ["rotary", "wtie", "cosine", "llama_mlp"]:
        d10 = L10[f] - L10["baseline"]
        d80 = L80[f] - L80["baseline"]
        print("%-10s %+10.4f %+10.4f %26.2f" % (f, d10, d80, abs(d80) / D_lr))
    print("(llama_mlp = rmsnorm+swiglu flipped jointly; per-flag split "
          "not identifiable from the sweep)")


def main():
    V = load()
    base_pts = [(t, V["baseline"][t]) for t in BUDGETS]
    L3 = refit_L_inf([p for p in base_pts if p[0] != 40.0])   # loop-style 3-pt
    L4 = refit_L_inf(base_pts)                                # 4-pt
    print("=== 1. surrogate calibration (loop's 2-point pinned-L_inf "
          "extrapolation 10,20 -> 80 s) ===")
    print("baseline L_inf: 3-pt fit {10,20,80} = %.4f ; "
          "4-pt fit {10,20,40,80} = %.4f ; loop prior = 1.407" % (L3, L4))
    runs = [(L3, "baseline 3-pt, what the loop would pin"),
            (L4, "baseline 4-pt"),
            (1.407, "combo round-1 fit, loop's hardcoded prior")]
    all_errs = [(tag, calibration(V, L, tag)) for L, tag in runs]
    print("\n=== 2. |error| vs behavioral distance from baseline ===")
    for tag, errs in all_errs:
        distance_test(V, errs, tag)
    sensitivities(V)


if __name__ == "__main__":
    main()
