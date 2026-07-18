"""Run a grid of variants x time budgets, each as a subprocess for isolation.

Usage: python sweep.py --variants baseline,lr_3e3 --budgets 10,20,40,80 --seeds 0
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", required=True, help="comma-separated variant names")
    p.add_argument("--budgets", default="10,20,40,80")
    p.add_argument("--seeds", default="0")
    p.add_argument("--out", default=os.path.join(HERE, "results", "results.jsonl"))
    args = p.parse_args()

    variants = args.variants.split(",")
    budgets = [float(b) for b in args.budgets.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    jobs = [(v, b, s) for v in variants for b in budgets for s in seeds]
    est = sum(b for _, b, _ in jobs)
    print("%d runs, ~%.0f s of training time (plus eval/startup overhead)" % (len(jobs), est))

    t_start = time.time()
    failures = 0
    for i, (v, b, s) in enumerate(jobs, 1):
        cmd = [sys.executable, os.path.join(HERE, "train.py"),
               "--variant", v, "--budget", str(b), "--seed", str(s), "--out", args.out]
        print("[%d/%d, %.0fs elapsed] %s budget=%s seed=%s" %
              (i, len(jobs), time.time() - t_start, v, b, s), flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            failures += 1
            print("  FAILED (exit %d)" % r.returncode, flush=True)
    print("done in %.0f s, %d failures" % (time.time() - t_start, failures))


if __name__ == "__main__":
    main()
