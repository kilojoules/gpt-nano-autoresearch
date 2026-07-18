# GPT Nano autoresearch

Automated research harness for finding **scaling laws of improvements** to GPT
Nano (minGPT's smallest config: 3 layers, 3 heads, 48-dim, char-level tiny
Shakespeare) under **wall-clock training-time budgets** (e.g. 20 s, 40 s).

The core question: when you change the model or training recipe, does the
improvement (a) shift the loss-vs-time curve down, (b) change its slope, and
(c) how much *effective extra training time* is it worth?

## Method

- Each run trains under a fixed training-time budget; eval is off the clock
  (speedrun convention), so eval frequency can't distort comparisons.
- Time-budgeted comparison automatically charges slower architectures for
  their extra per-step cost — improvements must pay for themselves in
  wall-clock terms.
- All runs are evaluated on the same fixed, seeded set of validation batches.
- Per variant, we fit `L(t) = L_inf + A * t^-alpha` to (budget, final val
  loss) points and report each improvement as an **effective time multiplier**
  vs the baseline curve.

## Files

| file | what it does |
|---|---|
| `model.py` | GPT Nano with improvement flags (RMSNorm, SwiGLU, rotary, weight tying) |
| `variants.py` | registry of candidate improvements (single change each) |
| `train.py` | one time-budgeted run → JSON line (loss curve, throughput, config) |
| `sweep.py` | grid of variants × budgets (subprocess-isolated) |
| `analyze.py` | power-law fits, effective-speedup table, log-log plot, report.md |
| `autoresearch.py` | the automated loop: propose → test cheap → adopt best → repeat |

## Quickstart

```bash
# one run
python train.py --variant baseline --budget 20

# manual sweep + analysis
python sweep.py --variants baseline,lr_3e3,cosine,dropout0 --budgets 10,20,40,80
python analyze.py

# the full automated research loop (greedy stacking search, then a
# confirmation sweep of the winning stack across the budget grid)
python autoresearch.py --rounds 3 --search-budgets 10,40 --confirm-budgets 10,20,40,80

# let Claude propose novel candidates each round (uses the `claude` CLI)
python autoresearch.py --rounds 3 --claude
```

Outputs land in `results/`: `results.jsonl` (raw runs), `report.md`,
`scaling.png`, `research_log.md` (autoresearch decisions).

## Notes

- Budgets below ~10 s are noisy (few hundred steps); prefer ≥10 s points for
  fitting, and add seeds (`--seeds 0,1,2`) for publication-grade curves.
- The power-law fit has 3 parameters; with a 4-point budget grid it is
  descriptive, not gospel. Extend the grid (160 s, 320 s…) to pin down alpha.
- Hardware: tuned on Apple M1 Pro via MPS; tiny models are dispatch-bound, so
  batch size 128 gives ~1.6× token throughput over 64 at equal wall-clock.
