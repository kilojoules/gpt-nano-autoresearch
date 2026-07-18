"""Trust-region autoresearch v2 — learned metric + noise-aware acceptance.

Every change vs trust_region.py (v1) is sourced from the research pass over
our own campaign logs (fit_metric.py) and the stochastic-TR literature:

1. LEARNED METRIC (the gbar region). Distance is measured in *behavioral
   nats* at the 80 s horizon, not in raw config units:
       d(x,x')^2 = sum_i w_i^2 (z_i - z'_i)^2  +  sum_j g_j^2 [flag_j differs]
   over z = (log2 lr, log2 batch, log2 wd, dropout/0.05). w comes from the
   measured per-dimension sensitivities, g ("gbar") from the measured flag
   displacements at 80 s — rotary 0.224 nats, weight tying 0.136, cosine
   0.136, rmsnorm/swiglu 0.069 each (llama_mlp joint split evenly). All
   weights are clipped to [0.01, 1.0] so the region stays uniformly
   norm-equivalent (the CGT Thm 6.7.1 condition that keeps scaled-TR
   convergence theory intact). A flip of flag j is admissible iff
   g_j <= Delta, and flips consume metric budget: the continuous move gets
   sqrt(Delta^2 - sum g_j^2). This prices "how many LR-doublings equal a
   rotary flip" empirically instead of by decree.
2. NOISE-AWARE ACCEPTANCE. Confirm runs are replicated (--reps seeds, common
   seed lists for candidate and incumbent = paired comparison); accept only if
       mean improvement >= max(C_SD * Delta^2, Z95 * SE(diff))
   i.e. a Delta^2-scaled sufficient-decrease test plus a one-sided 95%
   lower-confidence-bound test — a high-probability no-degradation guarantee.
   v1's fixed 0.002 margin was 0.16 sigma: a coin flip.
3. FRESH INCUMBENT. Each iteration adds one fresh 80 s replicate of the
   incumbent; its estimate is the running mean (sharpens instead of stale).
4. SKIP-CONFIRM. If no candidate is *predicted* to improve, no confirm budget
   is spent and the region shrinks (v1 spent 5/6 confirms on candidates its
   own surrogate had rejected, then let a hard-coded rho=0 shrink Delta).
5. RHO GATING. rho is computed only when predicted improvement > RHO_GATE;
   the radius grows only on accepted steps whose rho shows calibration.
6. BETTER SURROGATE INPUTS. Surrogate budgets {20, 40} s (10 s runs have
   replicate sd ~0.05 and step-count CV 0.16 — the worst fidelity we log);
   L_inf pinned from the incumbent's {20, 40, 80} fit.

Usage: python trust_region_v2.py --iters 5 --candidates 4 --reps 2
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from train import run
from variants import register_combo
from trust_region import START, to_overrides, refit_L_inf

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
LOG = os.path.join(RESULTS, "tr2_log.md")
STATE = os.path.join(RESULTS, "tr2_state.json")
RUNS = os.path.join(RESULTS, "tr2_runs.jsonl")
DATA = os.path.join(HERE, "data", "tinyshakespeare.txt")

# --- learned metric (fit_metric.py, 80 s horizon), clipped to [0.01, 1.0] ---
W = {"lr": 0.097, "batch_size": 0.060, "weight_decay": 0.060, "dropout": 0.034}
GBAR = {"rotary": 0.224, "weight_tying": 0.136, "cosine": 0.136,
        "rmsnorm": 0.069, "swiglu": 0.069}
CONT_BOUNDS = {"lr": (1e-4, 3e-2), "batch_size": (16, 512),
               "weight_decay": (1e-3, 1.0), "dropout": (0.0, 0.3)}

SIGMA80 = 0.0095   # replicate sd at 80 s (fit_metric.py, same-config pairs)
C_SD = 0.05        # sufficient-decrease coefficient (threshold = C_SD*Delta^2)
Z95 = 1.645        # one-sided 95%
RHO_GATE = 0.01    # rho counts as informative only above this predicted gain


def z_of(x):
    return np.array([math.log2(x["lr"]), math.log2(x["batch_size"]),
                     math.log2(x["weight_decay"]), x["dropout"] / 0.05])


def dist(x, xp):
    wvec = np.array([W["lr"], W["batch_size"], W["weight_decay"], W["dropout"]])
    d2 = float(np.sum((wvec * (z_of(x) - z_of(xp))) ** 2))
    for j in GBAR:
        if bool(x[j]) != bool(xp[j]):
            d2 += GBAR[j] ** 2
    return math.sqrt(d2)


def key_of(x):
    return json.dumps({k: (round(math.log2(v), 3) if k in ("lr", "batch_size", "weight_decay")
                           else round(v, 3) if k == "dropout" else bool(v))
                       for k, v in sorted(x.items())}, sort_keys=True)


def log(msg):
    print(msg, flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def propose(rng, x, delta, n, tried, max_flips=2):
    """n distinct candidates inside the metric ball d(x, .) <= delta."""
    out, guard = [], 0
    wvec = np.array([W["lr"], W["batch_size"], W["weight_decay"], W["dropout"]])
    while len(out) < n and guard < 300:
        guard += 1
        c = dict(x)
        budget2 = delta ** 2
        flips = []
        for j in sorted(GBAR, key=lambda _: rng.random()):
            if len(flips) >= max_flips:
                break
            if GBAR[j] ** 2 <= budget2 and rng.random() < 0.2:
                c[j] = not c[j]
                budget2 -= GBAR[j] ** 2
                flips.append(j)
        rem = math.sqrt(max(budget2, 0.0))
        v = rng.standard_normal(4)
        v /= max(np.linalg.norm(v), 1e-9)
        dz = v * rem * rng.uniform(0.4, 1.0) / wvec  # metric length <= rem
        z = z_of(x) + dz
        c["lr"] = float(np.clip(2.0 ** z[0], *CONT_BOUNDS["lr"]))
        c["batch_size"] = int(np.clip(round(2.0 ** z[1]), *CONT_BOUNDS["batch_size"]))
        c["weight_decay"] = float(np.clip(2.0 ** z[2], *CONT_BOUNDS["weight_decay"]))
        c["dropout"] = float(np.clip(z[3] * 0.05, *CONT_BOUNDS["dropout"]))
        if key_of(c) != key_of(x) and key_of(c) not in tried:
            tried.add(key_of(c))
            out.append((c, flips))
    return out


def run_at(name, x, budget, seed, device):
    model_ov, train_ov = to_overrides(x)
    register_combo(name, "TR v2", model_ov, train_ov)
    r = run(name, budget, seed, device, DATA, RUNS)
    log("    %s @ %ss seed=%d -> %.4f (%d steps)"
        % (name, int(budget), seed, r["final_val_loss"], r["steps"]))
    return r["final_val_loss"]


def pinned_predict(l1, t1, l2, t2, L_inf, T):
    if l1 - L_inf < 0.02 or l2 - L_inf < 0.02 or l2 >= l1:
        return l2
    alpha = math.log((l1 - L_inf) / (l2 - L_inf)) / math.log(t2 / t1)
    A = (l1 - L_inf) * (t1 ** alpha)
    return L_inf + A * (T ** -alpha)


def describe(x, ref):
    parts = []
    for k in sorted(x):
        if key_of({k: x[k]}) != key_of({k: ref[k]}):
            parts.append("%s->%s" % (k, x[k]) if isinstance(x[k], bool)
                         else "%s=%.4g" % (k, x[k]))
    return ", ".join(parts) if parts else "(no change)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--reps", type=int, default=2, help="confirm replicates")
    p.add_argument("--delta0", type=float, default=0.25, help="radius in nats")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--surrogate-budgets", default="20,40")
    p.add_argument("--confirm-budget", type=float, default=80.0)
    p.add_argument("--poll", action="store_true",
                   help="final 1-flip optimality poll of all 5 flags")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()

    t1, t2 = [float(b) for b in args.surrogate_budgets.split(",")]
    T = args.confirm_budget
    dmin, dmax = 0.04, 0.6
    delta = args.delta0
    rng = np.random.default_rng(args.seed)

    log("\n# TR v2 campaign %s" % time.strftime("%Y-%m-%d %H:%M"))
    log("metric: W=%s  GBAR=%s" % (W, GBAR))
    log("surrogate {%d,%d}s, confirm %ds x %d reps, delta0=%.2f nats, "
        "accept >= max(%.2f*delta^2, %.3f*SE)\n"
        % (int(t1), int(t2), int(T), args.reps, delta, C_SD, Z95))

    incumbent = dict(START)
    tried = {key_of(incumbent)}

    log("## Iteration 0: score incumbent (combo)")
    inc_s1 = run_at("tr2_inc0", incumbent, t1, 0, args.device)
    inc_s2 = run_at("tr2_inc0", incumbent, t2, 0, args.device)
    inc_reps = [run_at("tr2_inc0", incumbent, T, s + 1, args.device)
                for s in range(args.reps)]
    F_inc = float(np.mean(inc_reps))
    L_inf = refit_L_inf([(t1, inc_s1), (t2, inc_s2), (T, F_inc)], 1.4)
    log("  F_inc = %.4f (n=%d), L_inf = %.3f" % (F_inc, len(inc_reps), L_inf))

    history = []
    for it in range(1, args.iters + 1):
        # fresh incumbent replicate: the estimate sharpens every iteration
        inc_reps.append(run_at("tr2_inc_fresh", incumbent, T,
                               len(inc_reps) + 1, args.device))
        F_inc = float(np.mean(inc_reps))
        n_inc = len(inc_reps)
        log("\n## Iteration %d  (delta=%.3f nats, F_inc=%.4f n=%d)"
            % (it, delta, F_inc, n_inc))

        cands = propose(rng, incumbent, delta, args.candidates, tried)
        if not cands:
            log("  region exhausted of new candidates; shrinking")
            delta = max(delta * 0.5, dmin)
            continue

        preds, cand_pts = [], []
        for j, (c, flips) in enumerate(cands):
            d = dist(c, incumbent)
            log("  candidate %d (d=%.3f, flips=%s): %s"
                % (j, d, flips or "-", describe(c, incumbent)))
            l1 = run_at("tr2_it%d_c%d" % (it, j), c, t1, 0, args.device)
            l2 = run_at("tr2_it%d_c%d" % (it, j), c, t2, 0, args.device)
            pred = pinned_predict(l1, t1, l2, t2, L_inf, T)
            preds.append(pred)
            cand_pts.append([(t1, l1), (t2, l2)])
            log("    surrogate: L(%d) ~= %.4f" % (int(T), pred))

        j_best = int(np.argmin(preds))
        best, best_flips = cands[j_best]
        pred_improve = F_inc - preds[j_best]
        log("  best candidate %d: predicted improve %.4f" % (j_best, pred_improve))

        if pred_improve <= 0:
            # v1 confirmed model-rejected candidates 5/6 times; v2 does not
            log("  SKIP-CONFIRM (no predicted improvement in region); shrink")
            delta = max(delta * 0.6, dmin)
            history.append({"iter": it, "delta": delta, "action": "skip",
                            "pred_improve": round(pred_improve, 4)})
            continue

        new_reps = [run_at("tr2_it%d_confirm" % it, best, T, s + 1, args.device)
                    for s in range(args.reps)]  # seeds paired with incumbent's
        F_new = float(np.mean(new_reps))
        se = SIGMA80 * math.sqrt(1.0 / len(new_reps) + 1.0 / n_inc)
        improve = F_inc - F_new
        threshold = max(C_SD * delta ** 2, Z95 * se)
        rho = improve / pred_improve if pred_improve >= RHO_GATE else None
        log("  confirm: F_new = %.4f (n=%d)  improve %.4f vs threshold %.4f "
            "(SD %.4f, CI %.4f)  rho=%s"
            % (F_new, len(new_reps), improve, threshold,
               C_SD * delta ** 2, Z95 * se,
               "%.2f" % rho if rho is not None else "gated"))

        accepted = improve >= threshold
        if accepted:
            incumbent = best
            inc_reps = list(new_reps)
            F_inc = F_new
            L_inf = refit_L_inf(cand_pts[j_best] + [(T, F_new)], L_inf)
            log("  ACCEPT -> new incumbent %.4f (L_inf %.3f)  [%s]"
                % (F_inc, L_inf, describe(incumbent, START)))
        else:
            log("  reject")

        if accepted and rho is not None and 0.5 <= rho <= 2.0:
            delta = min(delta * 1.6, dmax)
        elif not accepted:
            delta = max(delta * 0.5, dmin)
        log("  new delta = %.3f" % delta)

        history.append({"iter": it, "delta": delta,
                        "pred_improve": round(pred_improve, 4),
                        "improve": round(improve, 4),
                        "threshold": round(threshold, 4),
                        "rho": None if rho is None else round(rho, 3),
                        "accepted": accepted,
                        "config": {k: (v if isinstance(v, bool) else round(float(v), 6))
                                   for k, v in best.items()}})
        with open(STATE, "w") as f:
            json.dump({"incumbent": incumbent, "F_inc": F_inc, "n_inc": len(inc_reps),
                       "delta": delta, "L_inf": L_inf, "history": history}, f, indent=2)

    log("\n## Final incumbent: %.4f (n=%d)  [%s]"
        % (F_inc, len(inc_reps), describe(incumbent, START)))

    if args.poll:
        log("\n## 1-flip optimality poll (all 5 flags, %ds, 1 seed)" % int(T))
        lcb = Z95 * SIGMA80 * math.sqrt(1.0 + 1.0 / len(inc_reps))
        optimal = True
        for j in sorted(GBAR):
            c = dict(incumbent)
            c[j] = not c[j]
            l = run_at("tr2_poll_%s" % j, c, T, 1, args.device)
            verdict = "improves beyond noise!" if l < F_inc - lcb else "no gain"
            if l < F_inc - lcb:
                optimal = False
            log("  flip %s -> %.4f (%s)" % (j, l, verdict))
        log("  1-flip local optimality: %s (LCB margin %.4f)"
            % ("CERTIFIED" if optimal else "REFUTED", lcb))

    log("\n## Budget-grid sweep of tr2_best")
    model_ov, train_ov = to_overrides(incumbent)
    register_combo("tr2_best", "TR v2 result: " + describe(incumbent, START),
                   model_ov, train_ov)
    for b in [10.0, 20.0, 40.0, 80.0]:
        r = run("tr2_best", b, args.seed, args.device, DATA,
                os.path.join(RESULTS, "results.jsonl"))
        log("  tr2_best @ %ss -> %.4f" % (int(b), r["final_val_loss"]))
    log("\ndone; python analyze.py for the updated scaling plot")


if __name__ == "__main__":
    main()
