"""Automated research loop for GPT Nano under training-time budgets.

Greedy stacking search:
  round r: apply each untried candidate ON TOP of the current best config,
  train at the (cheap) search budgets, adopt the best candidate if it beats
  the incumbent at the largest search budget by > --eps. Stop when no
  candidate helps or --rounds is exhausted.
Afterwards, the final adopted stack is registered as `auto_combo` and swept
across the (full) confirm budgets alongside `baseline`, and analyze.py fits
the scaling laws.

With --claude, each round also asks Claude (via the `claude` CLI) to propose
novel candidates as config overrides, given the research log so far.

Usage:
  python autoresearch.py --rounds 3 --search-budgets 10,40 --confirm-budgets 10,20,40,80
"""
import argparse
import json
import os
import subprocess
import time

import torch

from train import run, TrainConfig
from model import GPTConfig
from variants import VARIANTS, register_combo

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
LOG_PATH = os.path.join(RESULTS, "research_log.md")

MODEL_KEYS = set(GPTConfig.__dataclass_fields__)
TRAIN_KEYS = set(TrainConfig.__dataclass_fields__)


def log(msg):
    print(msg, flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")


def valid_overrides(model_ov, train_ov):
    return set(model_ov) <= MODEL_KEYS and set(train_ov) <= TRAIN_KEYS


def claude_ideas(history_text, n=3):
    """Ask Claude (CLI) for novel candidate improvements as JSON overrides."""
    prompt = (
        "You are proposing the next experiments in an automated ML research "
        "loop for a tiny char-level GPT (3 layers, 3 heads, 48 dim) trained "
        "on tiny Shakespeare under a wall-clock training-time budget on Apple "
        "MPS. Valid model config keys: %s. Valid train config keys: %s.\n"
        "Research log so far:\n%s\n\n"
        "Propose %d NEW single-change candidates not already tried. Reply with "
        "ONLY a JSON list, each item: {\"name\": str, \"desc\": str, "
        "\"model\": {..overrides..}, \"train\": {..overrides..}}."
        % (sorted(MODEL_KEYS), sorted(TRAIN_KEYS), history_text, n)
    )
    try:
        out = subprocess.run(["claude", "-p", prompt], capture_output=True,
                             text=True, timeout=180).stdout.strip()
        start, end = out.find("["), out.rfind("]")
        ideas = json.loads(out[start:end + 1])
        good = []
        for it in ideas:
            if it.get("name") and valid_overrides(it.get("model", {}), it.get("train", {})):
                good.append(it)
        return good
    except Exception as e:
        log("  (claude idea generation failed: %s)" % e)
        return []


def sweep_config(name, budgets, seeds, device, data, out):
    """Run `name` at each budget/seed; return mean loss at the largest budget."""
    finals = []
    for b in budgets:
        for s in seeds:
            r = run(name, b, s, device, data, out)
            log("    %s budget=%ss seed=%d -> %.4f (%d steps)" %
                (name, b, s, r["final_val_loss"], r["steps"]))
            if b == max(budgets):
                finals.append(r["final_val_loss"])
    return sum(finals) / len(finals)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--search-budgets", default="10,40")
    p.add_argument("--confirm-budgets", default="10,20,40,80")
    p.add_argument("--seeds", default="0")
    p.add_argument("--eps", type=float, default=0.005)
    p.add_argument("--claude", action="store_true",
                   help="ask Claude CLI for extra candidate ideas each round")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--data", default=os.path.join(HERE, "data", "tinyshakespeare.txt"))
    args = p.parse_args()

    search_budgets = [float(b) for b in args.search_budgets.split(",")]
    confirm_budgets = [float(b) for b in args.confirm_budgets.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    search_out = os.path.join(RESULTS, "search.jsonl")
    confirm_out = os.path.join(RESULTS, "results_auto.jsonl")

    log("\n# Autoresearch run %s\n" % time.strftime("%Y-%m-%d %H:%M"))
    log("search budgets %s, confirm budgets %s, eps %.3f" %
        (search_budgets, confirm_budgets, args.eps))

    adopted = []          # list of (name, desc)
    cur_model, cur_train = {}, {}
    log("\n## Round 0: baseline")
    incumbent = sweep_config("baseline", search_budgets, seeds, args.device,
                             args.data, search_out)
    log("  incumbent loss @ %ss: %.4f" % (max(search_budgets), incumbent))

    candidates = {k: v for k, v in VARIANTS.items() if k != "baseline"}

    for rnd in range(1, args.rounds + 1):
        log("\n## Round %d (incumbent %.4f, stack: %s)" %
            (rnd, incumbent, [n for n, _ in adopted] or "baseline"))
        if args.claude:
            hist = open(LOG_PATH).read()[-4000:]
            for it in claude_ideas(hist):
                if it["name"] not in candidates and it["name"] not in [n for n, _ in adopted]:
                    candidates[it["name"]] = {"desc": it.get("desc", ""),
                                              "model": it.get("model", {}),
                                              "train": it.get("train", {})}
                    log("  + claude proposed: %s — %s" % (it["name"], it.get("desc", "")))

        scores = {}
        for name, spec in list(candidates.items()):
            m_ov, t_ov = spec.get("model", {}), spec.get("train", {})
            # skip candidates that clash with an already-adopted override key
            if set(m_ov) & set(cur_model) or set(t_ov) & set(cur_train):
                continue
            trial = "r%d_%s" % (rnd, name)
            register_combo(trial, "%s + %s" % ([n for n, _ in adopted] or "baseline", name),
                           {**cur_model, **m_ov}, {**cur_train, **t_ov})
            log("  testing %s (%s)" % (trial, spec.get("desc", "")))
            scores[name] = sweep_config(trial, search_budgets, seeds,
                                        args.device, args.data, search_out)

        if not scores:
            log("  no untried non-conflicting candidates left; stopping")
            break
        best = min(scores, key=scores.get)
        log("  round %d scores: %s" %
            (rnd, {k: round(v, 4) for k, v in sorted(scores.items(), key=lambda kv: kv[1])}))
        if scores[best] < incumbent - args.eps:
            spec = candidates.pop(best)
            cur_model.update(spec.get("model", {}))
            cur_train.update(spec.get("train", {}))
            adopted.append((best, spec.get("desc", "")))
            incumbent = scores[best]
            log("  ADOPTED %s -> incumbent %.4f" % (best, incumbent))
        else:
            log("  no candidate beat incumbent by eps; stopping")
            break

    log("\n## Final stack: %s" % ([n for n, _ in adopted] or "baseline (nothing adopted)"))
    register_combo("auto_combo", " + ".join(n for n, _ in adopted) or "baseline",
                   cur_model, cur_train)

    log("\n## Confirm sweep across %s" % confirm_budgets)
    for name in ["baseline", "auto_combo"]:
        for b in confirm_budgets:
            for s in seeds:
                r = run(name, b, s, args.device, args.data, confirm_out)
                log("  %s budget=%ss seed=%d -> %.4f" % (name, b, s, r["final_val_loss"]))

    from analyze import analyze
    analyze(confirm_out)
    log("\ndone; see results/report.md")


if __name__ == "__main__":
    main()
