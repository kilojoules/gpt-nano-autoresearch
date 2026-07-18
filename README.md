# GPT Nano autoresearch: when can cheap experiments steer expensive ones?

## The question

Autoresearch loops (à la [karpathy/autoresearch](https://github.com/karpathy/autoresearch))
work like this: an agent proposes a change to the training recipe, trains for a
fixed budget, keeps the change if the loss improved, and repeats. Compute cost
forces the inner loop to run at **cheap fidelity** — say, GPT Nano with a max
training time of 20 s — while the model you actually care about is
**expensive** — GPT Nano at 10 minutes, or a real LLM.

That workflow embeds an assumption: *a verdict rendered at 20 s transfers to
10 min.* Today's loops assume we CAN transfer — there is no logic anywhere for
**not trusting the low-fidelity surrogate**. This repo makes that assumption
measurable, shows it failing, and borrows the formal machinery from nonlinear
optimization that fixes it: **trust regions**.

## Cheap verdicts lie: fidelity inversions in our own data

We trained GPT Nano (minGPT's 3-layer/3-head/48-dim config, char-level tiny
Shakespeare) under wall-clock training budgets {10, 20, 40, 80} s for the
baseline plus 8 single-change candidates. The ranking a 10 s judge produces
disagrees with the 80 s judge: Spearman rank correlation **0.70**, with the
disagreements concentrated exactly where it hurts:

| candidate | verdict @ 10 s | verdict @ 80 s |
|---|---|---|
| rotary embeddings | **worse than baseline** (rank 7/10) | **2nd-best single change** (rank 3, worth 4.0× training time) |
| RMSNorm + SwiGLU | worse than baseline (rank 9/10) | better than baseline (1.6× training time) |
| LR 3e-3 | best single change | best single change (transfer ✓) |

A 10 s-fidelity autoresearch loop would have *discarded rotary* — the
second-most-valuable improvement found in the whole sweep.

![counterexample](results/counterexample.png)

And the failure runs in **both directions**. At real campaign step sizes the
cheap signal also produces false positives — in v2's campaign, iteration 4's
surrogate predicted its candidate was **+0.008 better** and the replicated
80 s truth measured it **−0.004 worse** (the CI acceptance test rejected it):

| change / decision | cheap verdict | replicated 80 s truth | outcome |
|---|---|---|---|
| rotary | −0.006 (looks worse) | **+0.224 better** | false negative — a winner nearly discarded |
| RMSNorm+SwiGLU | −0.083 (looks worse) | +0.069 better | false negative |
| v1 iteration 2 | −0.023 (predicted worse) | +0.014 better | false negative — v1 confirmed anyway and got lucky |
| **v2 iteration 4** | **+0.008 (predicted better)** | **−0.004 worse** | **false positive — caught by the CI test** |
| v1 iteration 1 | +0.081 predicted | +0.009 | 9× over-promise, accepted |

(Full quadrant table: `python make_counterexample.py`.) Architecture
changes systematically look like regressions at low fidelity (they slow down
early optimization) and pay off at high fidelity. Hyperparameter changes
transfer better. The cheap judge is not wrong *uniformly*; it is wrong
*structurally* — which is what makes blind transfer dangerous.

## The fix, formally: trust-region optimization

Nonlinear optimization solved this problem decades ago. To minimize an
expensive function f(x) you build a cheap local model m(x), but you only ever
trust it inside a region whose size the model must continuously *earn*:

```
minimize f(x)                     f  = val loss at the target budget T   (expensive)
model    m_k ≈ f near x_k         m  = short-budget runs + power-law
                                       extrapolation L(t) = L∞ + A·t^−α  (cheap)
step     s_k = argmin m_k(x_k+s)  over steps ‖s‖ ≤ Δ_k                  (trust radius)
ratio    ρ_k = f(x_k) − f(x_k+s_k)     actual improvement
              ─────────────────────  = ──────────────────
              m_k(x_k) − m_k(x_k+s_k)  predicted improvement
update   ρ_k ≥ 0.75 → Δ_{k+1} = 2Δ_k       (model earned trust: bigger steps)
         ρ_k < 0.25 → Δ_{k+1} = Δ_k/2      (model over-promised: shrink, reject)
```

The ratio ρ is the missing logic. It *measures* whether the low-fidelity
surrogate predicted the high-fidelity outcome, and adapts how much the search
is allowed to lean on cheap experiments. `trust_region.py` instantiates this
for recipe search:

| TR concept | Here |
|---|---|
| expensive objective f(x) | val loss of recipe x at the confirm budget (80 s) |
| surrogate model m(x) | 10 s + 20 s runs → power-law fit (L∞ pinned from incumbent) → extrapolate |
| trust radius Δ | max recipe step per iteration: LR/batch/wd move ≤ ×2^Δ (log-space), dropout ±0.05Δ, ≤ 1–2 architecture-flag flips |
| ratio test ρ | actual 80 s improvement ÷ surrogate-predicted improvement |

It works in practice. Over a 6-iteration campaign (~35 min on a laptop,
starting from `combo`): iteration 1's surrogate predicted an improvement of
0.081 and the confirm run delivered 0.009 — ρ = 0.11 — so the loop kept the
(real but small) win and **halved the trust radius**. It accepted again at
iteration 2, then rejected four in a row as ρ stayed near zero, walking the
radius down 1.0 → 0.1: the search became conservative at exactly the rate the
20 s judge was shown to over-promise. That is the behavior today's
autoresearch loops are missing. Net result: `tr_best` (batch 176, lr 5.4e-3,
**RMSNorm removed**, wd 0.165, dropout 0.013) reached **1.6155** at 80 s vs
combo's 1.665 — and notably *reverted* one of the greedy round's adopted
changes, something a forward-only greedy loop can never do.

## The one-figure summary

![headline](results/headline_transfer.png)

Each point is one recipe change, judged twice: by a cheap 10-second
training run (x-axis) and by the real 80-second target (y-axis). If cheap
verdicts transferred, every point would hug the dashed diagonal. Instead the
two most valuable architecture changes — **rotary embeddings** (worth 4×
training time) and RMSNorm+SwiGLU — sit in the false-negative quadrant: the
cheap judge would have discarded them. That is the assumption every
low-fidelity autoresearch loop makes silently, made measurable — and why
this repo wraps the loop in trust-region distrust logic (§ below).
(Regenerate: `python make_counterexample.py`.)

## Did the trust region work? The replicated verdict

**Decision-level** (one ~50-min v2 campaign with the learned gbar metric,
replicated CI acceptance, fresh incumbent estimates):

- 2 of 5 iterations **skipped the confirm entirely** (surrogate predicted no
  in-region gain) — v1 had wasted 5/6 confirm budgets exactly there.
- Iteration 3: a +0.0027 "improvement" correctly **held below the CI bar**
  (threshold 0.0131) — v1's margin would have accepted it.
- Iteration 4: a **false positive caught** — surrogate predicted +0.008, the
  replicated truth measured −0.004, rejected.
- Iteration 5: the first **certified accept** — +0.0158 ≥ threshold 0.0125,
  n=2 paired seeds (batch 154, lr 2.9e-3, wd 0.0995).
- Final poll: **1-flip local optimality certified** (LCB margin 0.019); the
  rmsnorm-off flip measured +0.012 — promising but below the 1-seed bar, and
  it is the same move v1 had adopted.

**Outcome-level** (val loss @ 80 s, fresh replicate seeds):

| recipe | mean ± sd (n) | vs greedy combo |
|---|---|---|
| greedy combo | 1.6615 ± 0.0114 (7) | — |
| v1 trust region (uncertified accepts) | **1.6327** ± 0.0169 (5) | −0.029, z ≈ 3.3 — real |
| v2 trust region (certified accepts) | 1.6489 ± 0.0060 (3) | −0.013, z ≈ 2.3 — real |

Both trust-region campaigns beat the greedy stack under replication. The
honest trade-off: v1's risk-taking gained more ground than v2's certification
(Δ ≈ 0.016, z ≈ 2.0) in these short campaigns — v1's *individual* accepts
were noise-sized bets that happened to compound well, and only replication
after the fact could tell; v2 paid part of its budget for the guarantee that
every accepted step is real and finished with a certificate instead of a
hope. At larger fidelity gaps and candidate volumes the false-positive rate
v1 tolerates compounds against it (see the counterexample figure).

## Flagship result

![time to match quality](results/headline.png)

**The metric: time-to-match-quality.** For each recipe, solve its fitted
scaling law for the training time that reaches the baseline's 80-second
loss: autoresearch compressed 80 s of training into ~8 s (`python
make_headline.py`).

**The winning stack (`combo` = lr 3e-3 + rotary + RMSNorm/SwiGLU + dropout 0 +
batch 128) reaches the baseline's 80-second loss in under 10 seconds — a
measured >8× effective-training-time speedup — and the trust-region campaign
improved on it further: `tr_best` hits **1.6155** at 80 s vs baseline
**2.191**, steepening the scaling exponent from α ≈ 0.21 to α ≈ 0.75.** Full
decision log in [`results/tr_log.md`](results/tr_log.md). (Single-seed
numbers; the replicated three-way verdict is in the section above.)

![scaling laws](results/scaling.png)

Effective time multiplier = training time the baseline would need to match the
variant's loss at 80 s, from the baseline fit (1 seed, M1 Pro / MPS):

| variant | L @ 80s | fitted α | eff. time × |
|---|---|---|---|
| **tr_best (trust-region result)** | **1.616** | 0.75 | **~118× (extrap.)** |
| combo (greedy stack of winners) | 1.665 | 0.50 | ~62× (extrap.) |
| lr 3e-3 | 1.951 | 0.11 | 4.5× |
| rotary | 1.967 | 0.56 | 4.0× |
| lr 1e-3 | 2.076 | 0.09 | 2.0× |
| RMSNorm + SwiGLU | 2.122 | 0.25 | 1.6× |
| dropout 0 | 2.123 | 0.11 | 1.5× |
| batch 128 | 2.130 | 0.11 | 1.5× |
| baseline | 2.191 | 0.21 | 1× |
| cosine LR decay | 2.327 | 0.88 | 0.56× |
| weight tying | 2.327 | 0.62 | 0.55× |

Negative results worth knowing: budget-aware **cosine decay** and **weight
tying** both *hurt* in this regime — decaying a 5e-4 LR makes undertraining
worse, and tying costs capacity at 48-dim. Full numbers in
[`results/report.md`](results/report.md).

## Open question: we are optimizing functions, not numbers

Each recipe produces a whole curve L(t), not a scalar — so the "right" trust
region is genuinely underdetermined. The pluggable axes in `trust_region.py`:

- **Objective functional** (`--objective`): `confirm` scalarizes at t\* = 80 s;
  `integral` averages the curve over the measured budget grid. Also natural:
  maximize the fitted exponent α subject to no short-budget regression, or
  minimize *predicted* loss at a budget beyond the grid.
- **Where the region lives**: the default bounds steps in *recipe space*, but
  recipe distance is a poor proxy for behavior change (one boolean flip can
  move the curve more than any LR nudge — the same reason TRPO bounds KL
  between policies rather than parameter distance). `--behav-delta D` adds a
  *function-space* region: candidates whose 10 s loss deviates from the
  incumbent's by more than D, in either direction, are screened out before any
  confirm budget is spent.
- **Surrogate**: ours is a physics-informed power law per candidate. The
  Bayesian-optimization version is a joint GP over (recipe, budget) —
  multi-fidelity BO / freeze-thaw; with a maintained radius it is essentially
  TuRBO with a scaling-law kernel.
- **ρ for curves**: a scalar ratio at t\* now; grading the surrogate on its
  whole predicted *curve* (residuals at the confirm points) is the functional
  generalization.

Campaigns cost ~35 min on a laptop, so these are empirical questions here, not
philosophy: A/B the trust-region method itself across seeds and compare
trajectories.

## What's in the repo

| file | what it does |
|---|---|
| `model.py` | GPT Nano with improvement flags (RMSNorm, SwiGLU, rotary, weight tying) |
| `variants.py` | registry of candidate improvements (single change each) |
| `train.py` | one time-budgeted run → JSON line (loss curve, throughput, config) |
| `sweep.py` | grid of variants × budgets (subprocess-isolated) |
| `analyze.py` | power-law fits, effective-speedup table, log-log plot, report.md |
| `autoresearch.py` | greedy loop: propose → test cheap → adopt best → repeat (the *trusting* baseline) |
| `trust_region.py` | the trust-region loop (surrogate + radius + ρ test) |

## Method notes

- The budget counts training time only; eval is off the clock (speedrun
  convention), so eval frequency can't distort comparisons.
- Time-budgeted comparison automatically charges slower architectures for
  their extra per-step cost — improvements must pay for themselves in
  wall-clock terms.
- All runs are evaluated on the same fixed, seeded set of validation batches.
- Budgets below ~10 s are noisy; prefer ≥10 s points for fitting and add
  seeds (`--seeds 0,1,2`) for publication-grade curves. The 3-parameter fit on
  a 4-point grid is descriptive, not gospel — extend the grid to pin down α.

## Quickstart

```bash
# one run
python train.py --variant baseline --budget 20

# sweep + scaling-law analysis
python sweep.py --variants baseline,lr_3e3,cosine,dropout0 --budgets 10,20,40,80
python analyze.py

# the trusting greedy loop (optionally with Claude proposing candidates)
python autoresearch.py --rounds 3 --claude

# the trust-region loop
python trust_region.py --iters 6 --candidates 5 --behav-delta 0.3
```

Outputs land in `results/`: raw runs (`results.jsonl`), `report.md`,
`scaling.png`, `research_log.md` / `tr_log.md` (decision logs),
`tr_state.json` (resume state).

## Related work

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — agent
  iterates on a real LLM at a single fixed 5-min budget on an H100; verdicts
  are trusted as rendered.
- [The Automated LLM Speedrunning Benchmark](https://arxiv.org/abs/2506.22419)
  — can agents reproduce the 19 nanoGPT-speedrun records?
- [Prime Intellect auto-nanogpt](https://www.primeintellect.ai/auto-nanogpt) —
  two weeks of autonomous speedrun iteration (~10k runs).
- Trust-region methods: Conn, Gould & Toint (2000); TuRBO (trust-region BO);
  multi-fidelity BO and freeze-thaw BO are the GP-flavored cousins of our
  power-law surrogate.
