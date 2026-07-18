"""Trust-region autoresearch for GPT Nano under training-time budgets.

Classic trust-region structure mapped onto recipe search:

  expensive objective  f(x)  = val loss at the confirm budget (80 s)
  surrogate            m(x)  = runs at cheap budgets (10 s, 20 s) + power-law
                               extrapolation L(t) = L_inf + A t^-alpha with
                               L_inf pinned from the incumbent's fit
  trust region         Delta = max step in config space per iteration:
                               continuous dims move at most x2^Delta (log space,
                               additive for dropout), and at most 1-2 boolean
                               architecture flags may flip
  ratio test           rho   = actual improvement / predicted improvement
                               rho >= 0.75 -> grow Delta; rho < 0.25 -> shrink
                               and reject; accept the step only if the actual
                               80 s loss improves

The loop therefore stays aggressive while the scaling-law surrogate predicts
well, and automatically becomes conservative when extrapolation stops being
trustworthy.

Because the objective is a *function* L(t), not a number, two axes of the
method are pluggable:

  --objective confirm    f = L(confirm budget)            (scalar-at-t*)
  --objective integral   f = mean of L over the measured budget grid
                             (equal weight per log-spaced budget — cares about
                             the whole curve, not one point)
  --behav-delta D        a *function-space* trust region a la TRPO: candidates
                         whose short-budget loss deviates from the incumbent's
                         by more than D (in either direction) are screened out
                         before spending the confirm budget — the region bounds
                         behavior change, not config-space step size. 0 = off.

Usage: python trust_region.py --iters 6 --candidates 5 --seed 0
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from train import run
from variants import register_combo, VARIANTS

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
TR_LOG = os.path.join(RESULTS, "tr_log.md")
TR_STATE = os.path.join(RESULTS, "tr_state.json")
TR_RUNS = os.path.join(RESULTS, "tr_runs.jsonl")

SURROGATE_BUDGETS = [10.0, 20.0]
CONFIRM_BUDGET = 80.0
ACCEPT_MARGIN = 0.002   # must beat incumbent by this much at 80 s

# search space -------------------------------------------------------------
CONT = {
    # name: (kind, low, high)  kind 'log' = multiplicative, 'lin' = additive
    "lr":           ("log", 1e-4, 3e-2),
    "batch_size":   ("log", 16, 512),
    "weight_decay": ("log", 1e-3, 1.0),
    "dropout":      ("lin", 0.0, 0.3),
}
FLAGS = ["rotary", "rmsnorm", "swiglu", "weight_tying", "cosine"]

START = {  # = the `combo` winner from the round-1 sweep
    "lr": 3e-3, "batch_size": 128, "weight_decay": 0.1, "dropout": 0.0,
    "rotary": True, "rmsnorm": True, "swiglu": True,
    "weight_tying": False, "cosine": False,
}


def to_overrides(x):
    model_ov = {"rotary": x["rotary"], "rmsnorm": x["rmsnorm"],
                "swiglu": x["swiglu"], "weight_tying": x["weight_tying"],
                "dropout": round(float(x["dropout"]), 4)}
    train_ov = {"lr": float(x["lr"]), "batch_size": int(x["batch_size"]),
                "weight_decay": float(x["weight_decay"])}
    if x["cosine"]:
        train_ov["lr_schedule"] = "cosine"
    return model_ov, train_ov


def key_of(x):
    return json.dumps({k: (round(math.log10(v), 3) if isinstance(v, float) and k != "dropout"
                           else v if not isinstance(v, float) else round(v, 3))
                       for k, v in sorted(x.items())}, sort_keys=True)


def log(msg):
    print(msg, flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    with open(TR_LOG, "a") as f:
        f.write(msg + "\n")


def propose(rng, x, delta, n, tried):
    """n distinct candidates within trust radius delta of x."""
    out = []
    guard = 0
    while len(out) < n and guard < 200:
        guard += 1
        c = dict(x)
        # continuous dims: each moves with prob 0.6, bounded by delta
        for name, (kind, lo, hi) in CONT.items():
            if rng.random() >= 0.6:  # each continuous dim moves with prob 0.6
                continue
            if kind == "log":
                step = rng.uniform(-delta, delta)
                v = float(np.clip(c[name] * (2.0 ** step), lo, hi))
                c[name] = int(round(v)) if name == "batch_size" else v
            else:
                v = float(np.clip(c[name] + rng.uniform(-0.05 * delta, 0.05 * delta), lo, hi))
                c[name] = v
        # boolean flags: up to max_flips flips, each with small prob
        max_flips = 2 if delta >= 2.0 else (1 if delta >= 0.5 else 0)
        flips = 0
        for name in FLAGS:
            if flips >= max_flips:
                break
            if rng.random() < 0.15:
                c[name] = not c[name]
                flips += 1
        if key_of(c) != key_of(x) and key_of(c) not in tried:
            tried.add(key_of(c))
            out.append(c)
    return out


def objective(pts, kind):
    """Scalarize a measured/predicted curve [(t, L), ...]."""
    if kind == "confirm":
        return pts[-1][1]
    return float(np.mean([l for _, l in pts]))


def surrogate_predict(losses, L_inf):
    """Pinned-L_inf 2-point power-law fit -> predicted loss at CONFIRM_BUDGET."""
    (t1, y1), (t2, y2) = losses
    if y1 - L_inf < 0.02 or y2 - L_inf < 0.02 or y2 >= y1:
        return y2  # degenerate: no headroom or not improving -> flat prediction
    alpha = math.log((y1 - L_inf) / (y2 - L_inf)) / math.log(t2 / t1)
    A = (y1 - L_inf) * (t1 ** alpha)
    return L_inf + A * (CONFIRM_BUDGET ** -alpha)


def refit_L_inf(points, fallback):
    """3-point fit of L_inf from {(t, loss)}; keep it sane else fallback."""
    try:
        from scipy.optimize import curve_fit
        t = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        popt, _ = curve_fit(lambda tt, L, A, a: L + A * tt ** -a, t, y,
                            p0=[max(0.1, y.min() - 0.3), 1.0, 0.5],
                            bounds=([0.0, 1e-6, 0.01], [y.min(), 100.0, 3.0]),
                            maxfev=10000)
        return float(popt[0])
    except Exception:
        return fallback


def eval_config(name, x, budgets, device, seed):
    model_ov, train_ov = to_overrides(x)
    register_combo(name, "TR candidate", model_ov, train_ov)
    pts = []
    for b in budgets:
        r = run(name, b, seed, device,
                os.path.join(HERE, "data", "tinyshakespeare.txt"), TR_RUNS)
        pts.append((b, r["final_val_loss"]))
        log("    %s @ %ss -> %.4f (%d steps)" % (name, int(b), r["final_val_loss"], r["steps"]))
    return pts


def describe(x, ref):
    diffs = []
    for k in sorted(x):
        if key_of({k: x[k]}) != key_of({k: ref[k]}):
            if isinstance(x[k], bool):
                diffs.append("%s->%s" % (k, x[k]))
            elif isinstance(x[k], float):
                diffs.append("%s=%.4g" % (k, x[k]))
            else:
                diffs.append("%s=%s" % (k, x[k]))
    return ", ".join(diffs) if diffs else "(no change)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--candidates", type=int, default=5)
    p.add_argument("--delta0", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--surrogate-budgets", default="10,20")
    p.add_argument("--confirm-budget", type=float, default=80.0)
    p.add_argument("--objective", choices=["confirm", "integral"], default="confirm")
    p.add_argument("--behav-delta", type=float, default=0.0,
                   help="function-space region half-width at the first "
                        "surrogate budget; 0 disables screening")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()

    global SURROGATE_BUDGETS, CONFIRM_BUDGET
    SURROGATE_BUDGETS = [float(b) for b in args.surrogate_budgets.split(",")]
    CONFIRM_BUDGET = args.confirm_budget

    rng = np.random.default_rng(args.seed)
    delta, delta_min, delta_max = args.delta0, 0.1, 4.0

    log("\n# Trust-region autoresearch %s" % time.strftime("%Y-%m-%d %H:%M"))
    log("start = combo winner; surrogate budgets %s, confirm %ss, delta0=%.2f\n"
        % ([int(b) for b in SURROGATE_BUDGETS], int(CONFIRM_BUDGET), delta))

    incumbent = dict(START)
    tried = {key_of(incumbent)}
    L_inf = 1.407  # combo's fitted L_inf from the round-1 sweep

    log("## Iteration 0: score incumbent")
    inc_pts = eval_config("tr_incumbent0", incumbent,
                          SURROGATE_BUDGETS + [CONFIRM_BUDGET], args.device, args.seed)
    f_inc = objective(inc_pts, args.objective)
    L_inf = refit_L_inf(inc_pts, L_inf)
    log("  incumbent objective (%s) = %.4f, L_inf = %.3f"
        % (args.objective, f_inc, L_inf))

    history = []
    for it in range(1, args.iters + 1):
        log("\n## Iteration %d  (delta=%.2f, incumbent %.4f)" % (it, delta, f_inc))
        cands = propose(rng, incumbent, delta, args.candidates, tried)
        if not cands:
            log("  no new candidates in region; stopping")
            break
        preds, cand_pts = [], []
        for j, c in enumerate(cands):
            name = "tr_it%d_c%d" % (it, j)
            log("  candidate %d: %s" % (j, describe(c, incumbent)))
            pts = eval_config(name, c, SURROGATE_BUDGETS[:1], args.device, args.seed)
            # function-space trust region: screen candidates whose short-budget
            # behavior already left the band around the incumbent's curve
            dev = abs(pts[0][1] - inc_pts[0][1])
            if args.behav_delta > 0 and dev > args.behav_delta:
                log("    screened: |dL(%ds)| = %.3f > behav-delta %.3f"
                    % (int(pts[0][0]), dev, args.behav_delta))
                preds.append(float("inf"))
                cand_pts.append(pts)
                continue
            pts = pts + eval_config(name, c, SURROGATE_BUDGETS[1:], args.device, args.seed)
            pred80 = surrogate_predict(pts, L_inf)
            pred_obj = objective(pts + [(CONFIRM_BUDGET, pred80)], args.objective)
            preds.append(pred_obj)
            cand_pts.append(pts)
            log("    surrogate: L(%d) ~= %.4f, objective ~= %.4f"
                % (int(CONFIRM_BUDGET), pred80, pred_obj))

        if not np.isfinite(min(preds)):
            log("  all candidates screened by the function-space region; "
                "shrinking delta")
            delta = max(delta * 0.5, delta_min)
            continue

        j_best = int(np.argmin(preds))
        best, pred_best = cands[j_best], preds[j_best]
        pred_improve = f_inc - pred_best
        log("  best candidate %d (%s): predicted objective %.4f (improve %.4f)"
            % (j_best, describe(best, incumbent), pred_best, pred_improve))

        pts80 = eval_config("tr_it%d_confirm" % it, best, [CONFIRM_BUDGET],
                            args.device, args.seed)
        full_pts = cand_pts[j_best] + pts80
        f_new = objective(full_pts, args.objective)
        actual_improve = f_inc - f_new
        # rho is only meaningful when the model predicted an improvement;
        # a non-positive prediction that we ran anyway counts as model failure
        rho = actual_improve / pred_improve if pred_improve > 1e-4 else 0.0
        log("  confirm: objective = %.4f  (actual improve %.4f, rho = %.2f)"
            % (f_new, actual_improve, rho))

        accepted = f_new < f_inc - ACCEPT_MARGIN
        if accepted:
            incumbent, f_inc, inc_pts = best, f_new, full_pts
            L_inf = refit_L_inf(full_pts, L_inf)
            log("  ACCEPT -> new incumbent %.4f (L_inf %.3f)  [%s]"
                % (f_inc, L_inf, describe(best, START)))
        else:
            log("  reject (did not beat incumbent by %.3f)" % ACCEPT_MARGIN)

        if rho >= 0.75 and accepted:
            delta = min(delta * 2.0, delta_max)
        elif rho < 0.25:
            delta = max(delta * 0.5, delta_min)
        log("  new delta = %.2f" % delta)

        history.append({"iter": it, "delta": delta, "rho": round(rho, 3),
                        "pred": round(pred_best, 4), "actual": round(f_new, 4),
                        "incumbent": round(f_inc, 4), "accepted": accepted,
                        "config": {k: (v if not isinstance(v, float) else round(v, 6))
                                   for k, v in best.items()}})
        with open(TR_STATE, "w") as f:
            json.dump({"incumbent": incumbent, "f_inc": f_inc, "delta": delta,
                       "L_inf": L_inf, "history": history}, f, indent=2)

    log("\n## Final incumbent: %.4f  [%s]" % (f_inc, describe(incumbent, START)))
    model_ov, train_ov = to_overrides(incumbent)
    register_combo("tr_best", "trust-region search result: " + describe(incumbent, START),
                   model_ov, train_ov)
    log("\n## Confirm sweep of tr_best across the budget grid")
    for b in [10.0, 20.0, 40.0, 80.0]:
        r = run("tr_best", b, args.seed, args.device,
                os.path.join(HERE, "data", "tinyshakespeare.txt"),
                os.path.join(RESULTS, "results.jsonl"))
        log("  tr_best @ %ss -> %.4f" % (int(b), r["final_val_loss"]))
    log("\ndone; run `python analyze.py` for the updated scaling plot")


if __name__ == "__main__":
    main()
