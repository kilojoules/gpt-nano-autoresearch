# Can autoresearch be provably convergent?

Distance measures and convergence conditions for trust-region training-recipe
search, grounded in this repo's own campaign data. Sources were verified
against primary texts where possible; every claim whose source could not be
re-verified online is marked. The empirical numbers come from
`fit_metric.py` over `results/*.jsonl` and are reproducible.

**Short answers.**

- *Is there a better distance measure?* Yes, and it is measurable from our
  logs: a diagonal metric in behavioral-nat units over the continuous
  coordinates plus per-flag "exchange rates" (implemented in
  `trust_region_v2.py`). But the metric was **not** the binding problem —
  the surrogate's error is distance-independent, so no metric alone fixes
  the loop (§3).
- *Can the algorithm be provably convergent?* Yes, in a precise, tiered
  sense — and the guarantees attach to the **acceptance test**, not to the
  surrogate. A per-decision approximation of the cheap tier
  (high-probability monotone descent) is implemented in v2; the fully
  δ-budgeted version is v3 machinery (§7). The expensive tier (almost-sure stationarity in the
  continuous variables) has exact published conditions whose price on this
  hardware is ~Δ⁻⁴ replications; it is affordable down to Δ ≈ 0.25–0.5 in
  metric units and rapidly unaffordable below (§6). For the boolean flags,
  nothing stronger than certified local minimality w.r.t. the flip
  neighborhood exists — for anyone (§7).

---

## 1. What the v1 campaign data actually shows

The forensic pass (`fit_metric.py`) corrected several beliefs we formed
while watching the campaign live:

1. **The surrogate was not chronically over-promising.** In-campaign signed
   prediction errors at 80 s were {−0.072, +0.037, −0.007, −0.003, −0.004,
   +0.017}: mean bias −0.005, mean |error| 0.023. The notorious ρ ≈ 0.1 came
   from *one* large over-promise (iteration 1, a flag flip at Δ=1). The
   logged ρ=0.00 of iterations 2–6 was a **code artifact**: ρ was hard-coded
   to 0 whenever predicted improvement ≤ 1e-4.
2. **The loop confirmed candidates its own surrogate had rejected.** In 5 of
   6 iterations the best candidate's *predicted* objective was worse than
   the incumbent's — the loop had no skip branch, spent the 80 s confirm
   anyway, then let the artifact ρ=0 shrink the radius. The radius collapse
   1.0 → 0.1 was driven by noise plus this artifact, not by measured
   geometry.
3. **The L∞ pin dominated surrogate error.** The loop pinned L∞ from the
   incumbent's {10, 20, 80} fit; on baseline that gives L∞ = 1.861 vs 1.281
   when the 40 s point is included. Under the loop-realistic 3-pt pin
   (1.861) the surrogate is *pessimistic* (mean bias +0.083 over the 10
   non-baseline variants); under the round-1 prior pin (1.407) it is
   near-unbiased (−0.015, baseline included) with error independent of
   behavioral distance (Spearman −0.02/+0.04); the 4-pt pin (1.281) sits
   between (−0.029, Spearman 0.18–0.27). |Error| correlates with behavioral
   distance only under the bad pin (Spearman 0.72–0.79). Error *sign* under
   the good pin tracks change type, with a caveat: curve-reshaping changes
   are optimistic (cosine −0.200, weight tying −0.123), small
   curve-sliding changes are near-zero (batch +0.014, dropout +0.009, and
   the in-campaign continuous-only steps: −0.007…+0.017), but *large* lr
   moves at sweep scale (1–2.6 log₂ steps, far outside any campaign radius)
   are strongly pessimistic (+0.145/+0.168) — big continuous moves reshape
   the curve too.
4. **The noise floor.** Same-config 80 s replicate pairs give per-run
   sd ≈ 0.0095 (0.008–0.0095 across pair sets; n = 2–3 pairs, so a rough
   95% band of [0.5×, 2.5×] applies). Convention used throughout: a 1-seed
   *comparison* of two configs has σ_diff = σ√2 ≈ 0.013. The v1 accept
   margin of 0.002 is ≈ 0.15 σ_diff: the accept test's false-accept rate on
   a true-zero improvement is ≈ 44% — a coin flip. The one-sided 95%
   resolvable difference at 1 seed is ≈ 0.022 nats. v1's accepted
   improvements (0.009, 0.014) are individually 0.7 σ_diff and 1.0 σ_diff
   events. 10 s runs are ~5× noisier (sd ≈ 0.05, step-count CV
   0.16) — the worst fidelity we log, and v1's behavioral screen ran on it
   (it once screened a candidate because the run got 30 optimizer steps
   instead of ~105 — a throughput hiccup, not behavior).
5. **The 10 s window mis-prices architecture flags.** Flag displacement at
   10 s vs 80 s: rotary +0.006 → −0.224 (invisible early, biggest effect
   late); cosine +0.294 → +0.136 (2.2× overstated early). Rank correlation
   between |ΔL(10)| and |ΔL(80)| across the four flips is *negative*. Any
   metric or screen built on the 10 s window prices the discrete dimensions
   wrong.

## 2. The structural diagnosis: a fidelity lift is not a spatial model

Every convergence theorem for trust-region methods — classical,
derivative-free, probabilistic, stochastic — requires a **spatial model**
m(x) over the region whose error contracts as the region shrinks:

    |f(y) − m(y)| ≤ κ_ef·Δ²  and  ‖∇f(y) − ∇m(y)‖ ≤ κ_eg·Δ   on B(x_k, Δ)

("fully linear", Conn–Scheinberg–Vicente 2009, Def 3.1 — verified). Under
fully-linearity, **ρ → 1 as Δ → 0 while Δ_k ≲ ‖g_k‖ is a theorem** (the
acceptance lemma; the qualifier matters — near stationarity ρ is 0/0). Our
v1 data shows ρ never approaching 1 while Δ shrank 10× (0.11 once, then
artifact zeros — §1.1) with predicted decreases that did *not* vanish with
Δ, and — decisively — surrogate error independent of distance under a good
pin (§1.3): together these *empirically falsify* fully-linearity in any
config metric for the v1 surrogate. The
reason is structural: the power-law lift L̂(80 | x) = L∞ + A(x)·80^(−α(x))
is a **fidelity model in t**, fit per-candidate from its own cheap runs. It
defines no function over the ball, carries zero ∇ₓ information, and its
error (extrapolation bias + pin-transfer error + 2× 1-seed noise) is
**independent of Δ**. Shrinking the region therefore never improves the
model — the self-correction mechanism every trust-region proof runs on is
absent by construction.

Taxonomically (STORM's axes): randomness/bias can live in the *model* or in
the *estimates*. Our lift belongs on the **estimate axis** — it is a cheap,
biased estimator of f at the confirm fidelity — not the model axis. The
provable home for it is variance/cost reduction inside the estimator
(predict-then-correct with 80 s anchors, as in bi-fidelity stochastic TR),
never as m(x).

## 3. The distance-measure question (RQ1)

**What theory requires of a metric.** Convergence proofs tolerate any
iteration-dependent norm that stays *uniformly equivalent* to the Euclidean
one — (1/c)‖s‖₂ ≤ ‖s‖_k ≤ c‖s‖₂ for one c across all k (Conn–Gould–Toint
2000 §6.7; verified via secondary restatements). For a diagonal D_k this
means uniformly bounded condition number and scale. The metric never decides
*whether* a TR method converges; it decides the constants (κ_ef, κ_eg,
Lipschitz moduli measured in that metric) and hence the iteration
complexity. CMA-ES is the metric-*learning* analogue and deliberately
violates the boundedness safeguard — which is exactly why full CMA-ES has no
TR-style convergence proof. Lesson: **learn the metric from data, then clip
it** to fixed bounds; the clipped estimator keeps provability and most of
the conditioning benefit.

**The learned metric** (implemented in v2, fitted at the 80 s horizon):

    d(x,x')² = Σᵢ wᵢ² (zᵢ−z'ᵢ)²  +  Σⱼ gⱼ² · 1[flag_j differs]
    z = (log₂ lr, log₂ batch, log₂ wd, dropout/0.05)
    w = (0.097, 0.060, ~0.060*, 0.034) nats per unit        (*wd unidentified, assumed)
    g = (rotary 0.224, wtie 0.136, cosine 0.136, rmsnorm 0.069, swiglu 0.069)

The gⱼ are the **exchange rates**: a rotary flip "costs" 2.31 equivalent
log₂-lr steps. Flips are admissible iff gⱼ ≤ Δ and consume metric budget
(continuous part gets √(Δ² − Σg²)). All weights lie in [0.01, 1.0] — the
clip that keeps norm-equivalence; enforcement is currently by construction
of the constants, not by code, so a future refit must re-apply it.

**Do function-space (behavioral) metrics make sense?** Partly, with three
structural caveats established by the research pass:

- *Membership costs an evaluation* (≥ t₁/T of a run), so a behavioral radius
  can never shape proposal generation — only post-hoc filtering. In the
  constraint taxonomy it is a QRSK constraint; progressive-barrier theory
  (Audet–Dennis 2009) gives Clarke-stationarity for such constraints **only
  with a fixed feasible set** — an incumbent-centered band moves every
  accepted step and has no off-the-shelf theorem.
- *The TRPO template does not transfer.* TRPO's monotone-improvement bound
  η(π') ≥ L(π') − C·D_KL^max with computable C = 4εγ/(1−γ)² (Theorem 1 in
  D_TV^max form and its KL corollary, eq. 9; verified from the primary
  text) rests on the performance-difference lemma
  over a *shared MDP*: the surrogate is computable for all candidate
  policies from one incumbent batch, and distribution shift is bounded by a
  computable policy divergence. Recipe search has no shared probability
  space, no advantage decomposition, and no computable divergence that
  bounds |f(y) − f(x)|. The only TRPO-shaped statement available would
  *assume* an error modulus κ(d(x,y)) — an assumption our own ρ data already
  falsified by a factor of 3–10.
- *A screen must never veto geometry.* If a behavioral screen can block the
  model-improvement/poisedness evaluations, the fully-linear certification
  breaks and with it the convergence engine. Screens may reallocate budget
  among candidates, must scale with Δ, and "all screened" must not be a
  shrink signal (v1 did exactly this).

The resolution, and what v2 does: learn a **config-space proxy** of the
behavioral metric from logs (computable *before* evaluating). The
validation suite is specified but not yet implemented (`fit_metric.py`
currently provides only the Spearman diagnostic for the v1 surrogate):
Spearman/Kendall of (distance, surrogate-error) pairs, a log-log slope test
(slope ≈ 2 is fully-linear-like; slope ≈ 0 reproduces the v1 pathology),
and a 0.9-quantile κ̂ calibration test — which provides an empirical
consistency check of the value-error (κ_ef) half of STORM's α-probabilistic
fully-linearity at α = 0.9, marginally over sampled configs; the
gradient-error condition and the conditional-on-the-past structure of
STORM's Def 3.4 remain unverified assumptions. The metric question thereby
becomes a *checkable-in-part assumption* of the theorem instead of a
belief.

**One honest deviation in v2:** MADS-lineage theory says discrete
neighborhoods must be polled at *every* scale (integer steps floor at 1 and
never shrink with Δ; G-MADS, CatMADS Def 13). v2 still gates flips by
gⱼ ≤ Δ, so below Δ = 0.069 flips stop being proposed — same criticism as
v1's max_flips rule, softened (when the opt-in `--poll` is run) by the
final flip poll (§7). A fully theory-conform version would poll the
Hamming-1 neighborhood on a schedule at all radii.

## 4. Do we need gradients? (No — but noise is the real constraint)

Derivative-free methods never evaluate ∇f yet still earn gradient-flavored
guarantees; what every provable method supplies is a mechanism ensuring that
*if descent exists at scale Δ, it will eventually be found*: positive
spanning sets (direct search / MADS), implicit gradients of interpolation
models on Λ-poised sets (model-based DFO), or random directions (descent
probability ~½ for smooth f; Nesterov–Spokoiny-style methods converge at
~n× gradient-descent cost — trivial at n = 4). The binding constraint in our
setting is not gradients but **noise**: as Δ → 0, true improvements scale
like ‖∇f‖·Δ and drop below the 0.0095-nat replicate floor, at which point
*every* method must average. That is exactly where the Δ⁻⁴ replication laws
below come from.

## 5. The theory landscape (verified conditions)

| Framework | Randomness | Key conditions | Conclusion |
|---|---|---|---|
| Classical TR (Nocedal–Wright Thm 4.5/4.6) | none | Cauchy decrease, bounded model Hessians, Lipschitz ∇f, f bounded below | liminf ‖∇f‖=0 at η=0; **lim** ‖∇f‖=0 for η ∈ (0, ¼) |
| CSV 2009 (DFO-TR) | none | fully-linear class + finite model-improvement + criticality step (Δ = Θ(‖g‖) lock-step) | lim ‖∇f(x_k)‖ = 0 (Thm 5.9); also Δ_k → 0 (Lemma 5.5) |
| BSV 2014 | model | models fully linear with prob ≥ ½ conditioned on the past; f exact; accept needs ρ ≥ η₁ AND ‖g_k‖ ≥ η₂Δ_k | lim ‖∇f(X_k)‖ = 0 a.s. |
| STORM 2018 | model + estimates | α-prob fully-linear models; β-prob ε_F-accurate estimates with error ≤ ε_F·Δ²; ε_F ≤ min{κ_ef, η₁η₂κ_fcd/8}; probability conditions incl. αβ ≥ ½; **fresh estimates of both points every iteration** | lim ‖∇f(X_k)‖ = 0 a.s.; tolerates *biased* noise while bias ≤ ε_F·Δ² |
| ASTRO-DF 2018 | estimates (adaptive) | N_k = max{λ_k, λ_k σ̂²/(κ²Δ⁴)} with λ_k ≳ k^(1+ε); Δ locked to ‖∇M_k‖; **unbiased** iid noise, bounded 4v-th moments; authors: any lower power of Δ threatens consistency | Δ_k → 0 and lim ‖∇f(X_k)‖ = 0 w.p.1; O(ε⁻²) iterations (Ha–Shashaani) |
| AMMO 1998–2001 (multifidelity) | none | corrected low-fi model matches f **and ∇f** at the center (first-order consistency); value-only ("zeroth-order") explicitly loses the small-step guarantee | inherits classical TR convergence |
| March–Willcox 2012 | none | m = f_lo + RBF interpolant of the error f_hi − f_lo on poised sets ⊇ center → fully linear; **no high-fidelity gradients needed** | lim ‖∇f_hi‖ = 0 via CSV |
| StoMADS 2021 | estimates | β-prob ε_f-accurate (β > ½ with an explicit inequality), variance ≤ κ_F²Δ_p⁴, γ-sufficient-decrease with *certified-unsuccessful* branch, p_k ~ Δ_p⁻⁴ replications | a.s. Clarke stationarity — **continuous variables only** |
| MADS / MV-MADS / CatMADS | none | mesh refinement; positive spanning / dense refining directions; user-defined or learned categorical neighborhoods | Clarke stationarity in continuous dims + f(x*) ≤ f(neighbor) over the discrete neighborhood; CatMADS's analysis **does not cover** the full-product [cat-int-cont] grade (its guarantees top out at [cat-cont, cat-int] with the extended poll, ξ = ∞) |
| TRPO / CPI | sampling | shared MDP, performance-difference lemma, computable penalty C | monotone improvement — **does not transfer** to recipe search (§3) |
| Hyperband / Successive Halving | none (envelope) | per-arm limits ν_i exist; deviation envelope γ(j); budget > z_SH | returns an ε-best arm **of the sampled pool** — the only assumption-light guarantee for truncated-training fidelity, and it uses no extrapolation model at all |
| ASTRO-BFDF 2024 | estimates, bi-fidelity | adaptive sampling + bi-fidelity Monte Carlo | a.s. first-order stationarity of the high-fidelity objective — closest published architecture to our setting |

Verified gap (scoped): **no published stationarity-type convergence theorem
(TR/MADS lineage) covers stochastic objectives over a mixed
categorical-continuous space** (as of mid-2026). Pool-based stochastic
guarantees over finite/categorical sets do exist (best-arm identification,
e.g. Karnin et al. 2013; Andradóttir-style a.s. random search over finite
sets) but are of the ε-best-of-pool kind already tabled under Hyperband,
not local-convergence statements. The strongest honest statement for our
mixed space must be assembled from parts (§7).

## 6. The five violations of v1, and what v2/v3 do about them

| # | v1 violation (quantified) | v2 status | v3 (provable tier) |
|---|---|---|---|
| 1 | Fixed accept margin 0.002 ≈ 0.15 σ_diff → false-accept ≈ 44%; also permanently too strict as Δ → 0 (improvements scale O(Δ²) < 0.002) | accept ≥ max(0.05·Δ², 1.645·SE) — CI + sufficient decrease, fixed δ = 0.05 per test | additive θ + z·SE form with per-test δ_k budget Σδ_k ≤ δ |
| 2 | 1-seed estimates never sharpen; incumbent estimate stale (observed 0.013 same-config gap) | n=2 paired-seed confirms; fresh incumbent replicate every iteration (pool sharpens) | n(Δ) = max{λ_k, σ̂²λ_k/(κ²Δ⁴)} (ASTRO-DF), CRN pairing |
| 3 | Best-of-5 random candidates: no model decrease, no g_k, no Δ-progress coupling; confirms spent on model-rejected candidates | skip-confirm branch; ρ gated (only counts when predicted gain ≥ 0.01) | spatial linear model on ≥5 poised points → g_k; growth gated on ‖g_k‖ ≥ η₂Δ |
| 4 | Fidelity surrogate ≠ spatial model: error independent of Δ; ρ never recovers; L∞ pin badly biased | surrogate demoted out of the *accept test* (still used for candidate ranking, the skip-confirm decision, and ρ-based radius gating); better pin ({20,40,80}); surrogate budgets moved off the noisy 10 s fidelity | March–Willcox: m = lift + residual interpolation on poised 80 s points, recycling all past evals (~1–2 extra confirms/iter steady-state) |
| 5 | Δ floor 0.1 + noise → endgame is a random walk; theory needs Δ → 0 | floor 0.04 in metric nats, chosen as ≈ 3√2·σ̂ (hardcoded, not recomputed from σ̂) | remove floor *or* state the honest guarantee: convergence to the noise-floor neighborhood |

## 7. What is provable, exactly, and at what cost (RQ2)

**Tier 1 — high-probability monotone descent (approximated in v2; fully
realized only in v3).** The full rule: accept y_k iff
mean_n f(x_k) − mean_n f(y_k) ≥ θ + z_(1−δ_k)·σ̂·√(2/n), with fresh
incumbent replicates and a summable per-test budget δ_k = δ·2^(−(k+1)).
Then, with no assumption on the surrogate whatsoever:

- (G1) P(every accepted step truly improves by ≥ θ) ≥ 1 − δ — uniform
  no-degradation;
- (G2) on that event, #accepts ≤ (f(x₀) − f_min)/θ;
- (G3) with R < ∞ a uniform bound on the worst per-step degradation a false
  accept can cause (available here since val loss is bounded; or via f
  Lipschitz with steps confined to the region), V_k = f(x_k) + R·Σ_{j≥k} δ_j
  is a supermartingale bounded below ⇒ f(x_k) converges almost surely.

**What v2 actually implements** is the per-decision approximation: accept
iff improvement ≥ max(0.05·Δ², 1.645·σ̂·√(1/n + 1/n_inc)) with fixed
δ = 0.05 per test, a hardcoded σ̂ = 0.0095, and a max- rather than
additive-θ threshold. That gives per-iteration 95% no-degradation; G1–G3
as stated need the δ_k schedule and additive form (v3, §6 row 1).

This recovers exactly TRPO's *practical* property (monotone improvement
w.h.p.) without any surrogate-validity assumption — the guarantee moved from
the surrogate to the acceptance test. Costs below use a planning value
σ̂ = 0.0125 (measured 0.008–0.0095 from n = 2–3 pairs, inflated ~30% for
small-sample conservatism): certifying θ ≈ 0.02 needs n ≈ 10–20
replications per arm per accept decision (27–53 min at 80 s/run); θ ≈ 0.01
needs n ≈ 40. Common random numbers would cut n by the seed-correlation
factor (v2 already runs candidate confirms on the incumbent's first seed
list, but its SE formula is unpaired and gains nothing from this yet);
sequential/always-valid CIs allow early stopping. What Tier 1 never
provides: stationarity.

**Tier 2 — certified flip-local minimality.** Test each of the 5 Hamming-1
neighbors with a two-sample CI at level δ/5. Conclusion: "with probability
≥ 1 − δ, f(x*) ≤ f(y) + θ for every 1-flip neighbor y" — the *categorical
half* of the stochastic analogue of CatMADS's weakest grade (the continuous
half is the spanning-set certificate of §8 step 5). v2's `--poll` is the
skeleton only: opt-in, 1 seed per flip, per-test 95% with no multiplicity
correction, and it certifies just "no neighbor significantly better"; the
proper certificate needs replication, the δ/5 correction, and a
non-inferiority margin θ. Upgrading toward the [cat-cont] grade needs an
extended-poll rule: a flip whose CI overlaps the incumbent earns continuous
re-optimization before being judged.

**Tier 3 — almost-sure stationarity in the continuous variables.** With
flags frozen: either the StoMADS route (Δ_p²-scaled acceptance with the
certain/uncertain-unsuccessful distinction — shrink aggressively (τ²) only
when the failure is certified, mildly (τ) when uncertain, never freezing Δ
to await certification; β > ½ accuracy, p_k ~ Δ_p⁻⁴ replications,
positive-spanning/densifying directions) or the STORM/ASTRO-DF route
(probabilistically fully-linear spatial model from poised confirm-level
points + ε_F·Δ² estimates + η₂ growth gate). The STORM/ASTRO-DF route
yields lim ‖∇f(X_k)‖ = 0 a.s.; the StoMADS route yields a.s. Clarke
stationarity of refined points (⇒ ∇f(x*) = 0 there when f is C¹) — both
for f = E[val loss @ 80 s] over the 4 continuous dims: first-order
stationarity of the expected cheap-budget objective, not a global optimum,
not the T→∞ objective. The price on this hardware (planning σ̂ = 0.0125,
80 s/run, κ ≈ 0.02): n ≈ σ²/(κ²Δ⁴) ≈ 6 per evaluated point at Δ = 0.5,
≈ 100 at Δ = 0.25, ≈ 1600 at Δ = 0.125 — every halving of Δ costs 16× —
*before* ASTRO-DF's λ_k ≳ k^(1+ε) inflation and the ≥ 5-point poisedness
multiplier. Provability is
affordable down to Δ ≈ 0.25–0.5 and rapidly unaffordable below. Two further
caveats: ASTRO-DF requires *unbiased* noise, and our wall-clock budgets make
step counts config-correlated (batch size changes steps/s) — a fixed bias b
caps certifiable resolution at Δ ~ √(b/ε_F) regardless of replication;
measure b (fixed-step vs fixed-time runs) before buying replications. And
with a Δ floor retained, the honest statement everywhere is convergence to
the **noise-floor neighborhood**: the set of configs whose best in-ball
improvement is below the noise-scaled threshold.

**Closed routes.** TRPO-style computable penalties (no shared measure);
global optimality (nonconvex); regret-style MF-BO guarantees (need known bias
bounds ζ or a GP over (t, x) with known kernel — neither checkable for a
parametric power-law lift; and no published framework couples a
parametric-in-t extrapolation with any convergence or regret proof).
Distinct from closed: CatMADS Def-9 full mixed local minimality is an
*open gap*, not an impossibility — no published method certifies it, and
CatMADS explicitly does not cover it.

## 8. The v3 sketch (the provable loop, if we want to pay for it)

1. Freeze flags per TR cycle; flags handled by scheduled Hamming-1 polls +
   extended-poll re-optimization (Tier 2 certificates).
2. Maintain a cache of (x, L₂₀, L₄₀, L₈₀-replicates); each iteration ensure
   ≥ 5 affinely independent cached points inside B(x_k, Δ_k) (top up with
   CRN-seeded confirms — steady-state ~1–2 new confirms/iter).
3. Model m_k = power-law lift + linear/RBF interpolant of lift residuals on
   the poised set (March–Willcox): convergence never depends on the power
   law being right; the lift's only job is to shrink the residual's κ's.
4. Minimize m_k in the clipped learned metric ball s.t. fraction-of-Cauchy
   decrease; ρ-test with threshold max(κΔ², CI); on failure, shrink with the
   certain/uncertain distinction (τ² when certified unsuccessful, τ when
   uncertain — never freeze Δ awaiting certification); growth gated on
   ‖g_k‖ ≥ η₂Δ_k.
5. Adaptive replication n(Δ) per ASTRO-DF; terminate with a certificate, not
   a budget: 5 flip tests + a positive spanning set of continuous directions
   at the final Δ, Bonferroni-corrected (~13–16 configs × n ≈ 10 → 3–4 h on
   this laptop — the full certificate is a deliberate splurge).

## 9. References (verification status)

Primary-text verified: Conn–Scheinberg–Vicente SIOPT 2009; Bandeira–
Scheinberg–Vicente SIOPT 2014; Chen–Menickelly–Scheinberg (STORM) Math.
Prog. 2018; Shashaani–Hashemi–Pasupathy (ASTRO-DF) SIOPT 2018; Audet–
Dzahini–Kokkolaras–Le Digabel (StoMADS) COAP 2021; Schulman et al. (TRPO)
ICML 2015; Alexandrov–Dennis–Lewis–Torczon 1998 + Alexandrov–Lewis 2000
(AMMO); March–Willcox AIAA J 2012; Li et al. (Hyperband) JMLR 2018;
Kandasamy et al. (MF-GP-UCB JAIR 2019, BOCA ICML 2017); CatMADS
arXiv:2506.06937; Giovannelli–Liuzzi–Lucidi–Rinaldi arXiv:2107.00601;
Larson–Menickelly–Wild Acta Numerica 2019; Hansen CMA-ES tutorial;
Audet–Dennis MADS SIOPT 2006 (via restatements), progressive barrier SIOPT
2009; Blanchet et al. IJOO 2019 and Ha–Shashaani 2023 (abstracts).

Partially verified / lower confidence: Conn–Gould–Toint 2000 exact theorem
numbering (§6.7 norm-equivalence condition verified via secondary sources);
STORM's exact numeric α, β thresholds (definitions and the ε_F inequality
verified; Cor 4.12's inequalities transcribed from the text but not
independently re-derived); Kakade–Langford CPI original phrasing (scanned
PDF; cross-verified via two restatements); Rinaldi et al. SIOPT 2024
(reduced sample sizes under weak tail bounds — title/venue only); MV-MADS
theorem numbering (paywalled; via CatMADS's reproduction).

Empirical claims: all from this repo's `results/*.jsonl` via
`fit_metric.py`; n is small everywhere (2–3 replicate pairs for σ̂, 10–11
variants for calibration/sensitivities, 6 in-campaign prediction pairs) —
treat the constants as first estimates, refit as campaigns accumulate.
