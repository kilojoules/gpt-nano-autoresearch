# GPT Nano autoresearch

Automated research harness for finding **scaling laws of improvements** to GPT
Nano (minGPT's smallest config: 3 layers, 3 heads, 48-dim, char-level tiny
Shakespeare) under **wall-clock training-time budgets** (e.g. 20 s, 40 s).

The core question: when you change the model or training recipe, does the
improvement (a) shift the loss-vs-time curve down, (b) change its slope, and
(c) how much *effective extra training time* is it worth?

## Flagship result

**The winning stack (`combo` = lr 3e-3 + rotary + RMSNorm/SwiGLU + dropout 0 +
batch 128) reaches the baseline's 80-second loss in under 10 seconds — a
measured >8× effective-training-time speedup (~60× extrapolated from the
baseline's fitted power law) — and steepens the scaling exponent from
α ≈ 0.21 to α ≈ 0.50.** Val loss at 80 s: **1.665** vs baseline **2.191**.

![scaling laws](results/scaling.png)

Effective time multiplier = training time the baseline would need to match the
variant's loss at 80 s, from the baseline fit (1 seed, M1 Pro / MPS):

| variant | L @ 80s | fitted α | eff. time × |
|---|---|---|---|
| **combo (stack of winners)** | **1.665** | 0.50 | **~62× (extrap.)** |
| lr 3e-3 | 1.951 | 0.11 | 4.5× |
| rotary | 1.967 | 0.56 | 4.0× |
| lr 1e-3 | 2.076 | 0.09 | 2.0× |
| RMSNorm + SwiGLU | 2.122 | 0.25 | 1.6× |
| dropout 0 | 2.123 | 0.11 | 1.5× |
| batch 128 | 2.130 | 0.11 | 1.5× |
| baseline | 2.191 | 0.21 | 1× |
| cosine LR decay | 2.327 | 0.88 | 0.56× |
| weight tying | 2.327 | 0.62 | 0.55× |

Two negative results worth knowing: budget-aware **cosine decay** and **weight
tying** both *hurt* in this regime — decaying a 5e-4 LR makes undertraining
worse, and tying costs capacity at 48-dim. Full numbers in
[`results/report.md`](results/report.md).

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

## Trust-region autoresearch

`trust_region.py` recasts the research loop as a trust-region method:

| TR concept | Here |
|---|---|
| expensive objective f(x) | val loss of config x at the confirm budget (80 s) |
| surrogate model m(x) | 10 s + 20 s runs → power-law fit (L∞ pinned from incumbent) → extrapolate to 80 s |
| trust radius Δ | max config step per iter: LR/batch/wd move ≤ ×2^Δ (log-space), dropout ±0.05Δ, ≤ 1–2 flag flips |
| ratio test ρ | actual 80 s improvement ÷ surrogate-predicted improvement |
| radius update | ρ ≥ 0.75 → Δ×2; ρ < 0.25 → Δ/2 and reject |

So the search stays aggressive while short-budget results extrapolate reliably,
and turns cautious exactly when the scaling-law surrogate stops predicting.

**Open question — we're optimizing functions, not numbers.** Each config
produces a whole curve L(t), so the "right" trust region is genuinely
underdetermined, and several axes are pluggable:

- **Objective functional** (`--objective`): `confirm` scalarizes at t\* = 80 s;
  `integral` averages the curve over the measured budget grid. Other options
  worth trying: maximize the fitted exponent α subject to no short-budget
  regression, or minimize *predicted* loss at a budget beyond the grid.
- **Where the region lives**: the default bounds steps in *config space*, but
  config distance is a poor proxy for behavior change (one boolean flip can
  move the curve more than any LR nudge — the same reason TRPO bounds KL
  between policies rather than parameter distance). `--behav-delta D` adds a
  *function-space* region: candidates whose 10 s loss deviates from the
  incumbent's by more than D (in either direction) are screened out before any
  confirm budget is spent.
- **Surrogate**: ours is a physics-informed power law per candidate. The
  Bayesian-optimization version would be a joint GP over (config, budget) —
  multi-fidelity BO / freeze-thaw; with a maintained trust radius that is
  essentially TuRBO with a scaling-law kernel.
- **ρ for curves**: a scalar ratio at t\* now; comparing the predicted vs
  measured *curve* (fit residuals at the confirm points) would grade the
  surrogate as a function approximator instead.

```bash
python trust_region.py --iters 6 --candidates 5 --behav-delta 0.3
```

Outputs: `results/tr_log.md` (full decision log), `results/tr_state.json`
(resume state), and a `tr_best` sweep appended to `results/results.jsonl` so
`analyze.py` places it on the scaling plot.

## Files

| file | what it does |
|---|---|
| `model.py` | GPT Nano with improvement flags (RMSNorm, SwiGLU, rotary, weight tying) |
| `variants.py` | registry of candidate improvements (single change each) |
| `train.py` | one time-budgeted run → JSON line (loss curve, throughput, config) |
| `sweep.py` | grid of variants × budgets (subprocess-isolated) |
| `analyze.py` | power-law fits, effective-speedup table, log-log plot, report.md |
| `autoresearch.py` | the automated loop: propose → test cheap → adopt best → repeat |
| `trust_region.py` | trust-region variant of the loop (surrogate + radius + ρ test) |

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
