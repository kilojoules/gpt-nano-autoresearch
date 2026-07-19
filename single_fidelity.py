"""Single-fidelity control arm for the money-shot comparison.

Identical to trust_region_v2 in every respect EXCEPT the fidelity schedule:
no cheap 20/40 s screening — each iteration proposes ONE candidate from the
same learned-metric ball and pays the full replicated 80 s price to judge it.
Radius adapts on success/failure alone (no surrogate, no rho). Stops at a
total-training-seconds cap matched to v2's search phase, so cumulative-
compute curves are comparable.

Usage: python single_fidelity.py --budget-cap 2300 --reps 2
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
from trust_region import START
from trust_region_v2 import (W, GBAR, SIGMA80, C_SD, Z95, propose, dist,
                             key_of, to_overrides, describe)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
LOG = os.path.join(RESULTS, "sf_log.md")
STATE = os.path.join(RESULTS, "sf_state.json")
RUNS = os.path.join(RESULTS, "sf_runs.jsonl")
DATA = os.path.join(HERE, "data", "tinyshakespeare.txt")


def log(msg):
    print(msg, flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget-cap", type=float, default=2300.0,
                   help="total training seconds for the search (matches v2)")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--delta0", type=float, default=0.25)
    p.add_argument("--confirm-budget", type=float, default=80.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()

    T = args.confirm_budget
    rng = np.random.default_rng(args.seed)
    delta, dmin, dmax = args.delta0, 0.04, 0.6
    spent = 0.0

    def run_at(name, x, budget, seed):
        nonlocal spent
        model_ov, train_ov = to_overrides(x)
        register_combo(name, "SF arm", model_ov, train_ov)
        r = run(name, budget, seed, args.device, DATA, RUNS)
        spent += budget
        log("    %s @ %ss seed=%d -> %.4f (spent %ds)"
            % (name, int(budget), seed, r["final_val_loss"], int(spent)))
        return r["final_val_loss"]

    log("\n# Single-fidelity control arm %s" % time.strftime("%Y-%m-%d %H:%M"))
    log("no screening: every candidate pays %d s x %d reps; cap %d s\n"
        % (int(T), args.reps, int(args.budget_cap)))

    incumbent = dict(START)
    tried = {key_of(incumbent)}
    inc_reps = [run_at("sf_inc0", incumbent, T, s + 1) for s in range(args.reps)]
    F_inc = float(np.mean(inc_reps))
    log("  F_inc = %.4f (n=%d)" % (F_inc, len(inc_reps)))

    history = []
    it = 0
    while spent + T * (args.reps + 1) <= args.budget_cap:
        it += 1
        inc_reps.append(run_at("sf_inc_fresh", incumbent, T, len(inc_reps) + 1))
        F_inc = float(np.mean(inc_reps))
        n_inc = len(inc_reps)
        log("\n## Iteration %d  (delta=%.3f, F_inc=%.4f n=%d, spent %ds)"
            % (it, delta, F_inc, n_inc, int(spent)))

        cands = propose(rng, incumbent, delta, 1, tried)
        if not cands:
            delta = max(delta * 0.5, dmin)
            continue
        c, flips = cands[0]
        log("  candidate (d=%.3f, flips=%s): %s"
            % (dist(c, incumbent), flips or "-", describe(c, incumbent)))

        new_reps = [run_at("sf_it%d" % it, c, T, s + 1) for s in range(args.reps)]
        F_new = float(np.mean(new_reps))
        se = SIGMA80 * math.sqrt(1.0 / len(new_reps) + 1.0 / n_inc)
        improve = F_inc - F_new
        threshold = max(C_SD * delta ** 2, Z95 * se)
        accepted = improve >= threshold
        log("  F_new = %.4f (n=%d)  improve %.4f vs threshold %.4f -> %s"
            % (F_new, len(new_reps), improve, threshold,
               "ACCEPT" if accepted else "reject"))
        if accepted:
            incumbent, inc_reps, F_inc = c, list(new_reps), F_new
            delta = min(delta * 1.6, dmax)
        else:
            delta = max(delta * 0.5, dmin)
        history.append({"iter": it, "spent_s": round(spent, 1), "delta": delta,
                        "improve": round(improve, 4), "accepted": accepted,
                        "config": {k: (v if isinstance(v, bool) else round(float(v), 6))
                                   for k, v in c.items()}})
        with open(STATE, "w") as f:
            json.dump({"incumbent": incumbent, "F_inc": F_inc,
                       "n_inc": len(inc_reps), "spent_s": spent,
                       "history": history}, f, indent=2)

    log("\n## Final: %.4f (n=%d) after %ds  [%s]"
        % (F_inc, len(inc_reps), int(spent), describe(incumbent, START)))


if __name__ == "__main__":
    main()
