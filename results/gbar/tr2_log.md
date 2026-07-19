
# TR v2 campaign 2026-07-19 02:42
metric: W={'lr': 0.097, 'batch_size': 0.06, 'weight_decay': 0.06, 'dropout': 0.034}  GBAR={'rotary': 0.224, 'weight_tying': 0.136, 'cosine': 0.136, 'rmsnorm': 0.069, 'swiglu': 0.069}
surrogate {4,8}s, confirm 16s x 2 reps, delta0=0.25 nats, accept >= max(0.05*delta^2, 1.645*SE)

## Iteration 0: score incumbent (combo)
    tr2_inc0 @ 4s seed=0 -> 2.8914 (20 steps)
    tr2_inc0 @ 8s seed=0 -> 2.6701 (30 steps)
    tr2_inc0 @ 16s seed=1 -> 2.2981 (70 steps)
    tr2_inc0 @ 16s seed=2 -> 2.2960 (70 steps)
  F_inc = 2.2970 (n=2), L_inf = 0.000
    tr2_inc_fresh @ 16s seed=3 -> 2.2820 (70 steps)

## Iteration 1  (delta=0.250 nats, F_inc=2.2920 n=3)
  candidate 0 (d=0.209, flips=-): batch_size=133, lr=0.00204, weight_decay=0.009757
    tr2_it1_c0 @ 4s seed=0 -> 3.0636 (20 steps)
    tr2_it1_c0 @ 8s seed=0 -> 2.8304 (30 steps)
    surrogate: L(16) ~= 2.6151
  candidate 1 (d=0.250, flips=['rotary', 'rmsnorm']): batch_size=173, dropout=0.01018, lr=0.002114, rmsnorm->False, rotary->False, weight_decay=0.2156
    tr2_it1_c1 @ 4s seed=0 -> 3.4656 (10 steps)
    tr2_it1_c1 @ 8s seed=0 -> 3.0814 (20 steps)
    surrogate: L(16) ~= 2.7399
  best candidate 0: predicted improve -0.3230
  SKIP-CONFIRM (no predicted improvement in region); shrink
    tr2_inc_fresh @ 16s seed=4 -> 2.2906 (70 steps)

## Iteration 2  (delta=0.150 nats, F_inc=2.2917 n=4)
  candidate 0 (d=0.144, flips=['swiglu']): batch_size=250, lr=0.003852, swiglu->False, weight_decay=0.02922
    tr2_it2_c0 @ 4s seed=0 -> 3.1738 (10 steps)
    tr2_it2_c0 @ 8s seed=0 -> 2.8025 (20 steps)
    surrogate: L(16) ~= 2.4746
  candidate 1 (d=0.123, flips=-): batch_size=292, lr=0.006029, weight_decay=0.1251
    tr2_it2_c1 @ 4s seed=0 -> 3.0633 (10 steps)
    tr2_it2_c1 @ 8s seed=0 -> 2.6667 (20 steps)
    surrogate: L(16) ~= 2.3214
  best candidate 1: predicted improve -0.0297
  SKIP-CONFIRM (no predicted improvement in region); shrink

## Final incumbent: 2.2917 (n=4)  [(no change)]

## Budget-grid sweep of tr2_best
  tr2_best @ 10s -> 2.5305
  tr2_best @ 20s -> 2.2460
  tr2_best @ 40s -> 1.9922
  tr2_best @ 80s -> 1.8205

done; python analyze.py for the updated scaling plot
