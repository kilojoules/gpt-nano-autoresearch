
# TR v2 campaign 2026-07-18 15:59
metric: W={'lr': 0.097, 'batch_size': 0.06, 'weight_decay': 0.06, 'dropout': 0.034}  GBAR={'rotary': 0.224, 'weight_tying': 0.136, 'cosine': 0.136, 'rmsnorm': 0.069, 'swiglu': 0.069}
surrogate {20,40}s, confirm 80s x 2 reps, delta0=0.25 nats, accept >= max(0.05*delta^2, 1.645*SE)

## Iteration 0: score incumbent (combo)
    tr2_inc0 @ 20s seed=0 -> 1.8955 (230 steps)
    tr2_inc0 @ 40s seed=0 -> 1.7400 (460 steps)
    tr2_inc0 @ 80s seed=1 -> 1.6553 (920 steps)
    tr2_inc0 @ 80s seed=2 -> 1.6596 (910 steps)
  F_inc = 1.6574 (n=2), L_inf = 1.564
    tr2_inc_fresh @ 80s seed=3 -> 1.6739 (930 steps)

## Iteration 1  (delta=0.250 nats, F_inc=1.6629 n=3)
  candidate 0 (d=0.209, flips=-): batch_size=133, lr=0.00204, weight_decay=0.009757
    tr2_it1_c0 @ 20s seed=0 -> 1.9494 (230 steps)
    tr2_it1_c0 @ 40s seed=0 -> 1.7628 (460 steps)
    surrogate: L(80) ~= 1.6665
  candidate 1 (d=0.250, flips=['rotary', 'rmsnorm']): batch_size=173, dropout=0.01018, lr=0.002114, rmsnorm->False, rotary->False, weight_decay=0.2156
    tr2_it1_c1 @ 20s seed=0 -> 2.1218 (240 steps)
    tr2_it1_c1 @ 40s seed=0 -> 1.8754 (480 steps)
    surrogate: L(80) ~= 1.7379
  candidate 2 (d=0.134, flips=['swiglu']): batch_size=236, lr=0.002818, swiglu->False, weight_decay=0.3209
    tr2_it1_c2 @ 20s seed=0 -> 2.1046 (170 steps)
    tr2_it1_c2 @ 40s seed=0 -> 1.9258 (320 steps)
    surrogate: L(80) ~= 1.8060
  candidate 3 (d=0.155, flips=['weight_tying']): lr=0.001873, weight_decay=0.1513, weight_tying->True
    tr2_it1_c3 @ 20s seed=0 -> 2.0426 (220 steps)
    tr2_it1_c3 @ 40s seed=0 -> 1.8458 (450 steps)
    surrogate: L(80) ~= 1.7299
  best candidate 0: predicted improve -0.0036
  SKIP-CONFIRM (no predicted improvement in region); shrink
    tr2_inc_fresh @ 80s seed=4 -> 1.6461 (870 steps)

## Iteration 2  (delta=0.150 nats, F_inc=1.6587 n=4)
  candidate 0 (d=0.147, flips=['rmsnorm', 'swiglu']): batch_size=163, dropout=0.1496, lr=0.002389, rmsnorm->False, swiglu->False, weight_decay=0.08253
    tr2_it2_c0 @ 20s seed=0 -> 2.0701 (220 steps)
    tr2_it2_c0 @ 40s seed=0 -> 1.9020 (450 steps)
    surrogate: L(80) ~= 1.7897
  candidate 1 (d=0.105, flips=-): batch_size=81, dropout=0.004322, lr=0.001705, weight_decay=0.191
    tr2_it2_c1 @ 20s seed=0 -> 2.0315 (200 steps)
    tr2_it2_c1 @ 40s seed=0 -> 1.8683 (400 steps)
    surrogate: L(80) ~= 1.7620
  candidate 2 (d=0.149, flips=['swiglu']): batch_size=366, dropout=0.1246, lr=0.002244, swiglu->False, weight_decay=0.08363
    tr2_it2_c2 @ 20s seed=0 -> 2.3183 (90 steps)
    tr2_it2_c2 @ 40s seed=0 -> 2.0701 (200 steps)
    surrogate: L(80) ~= 1.9036
  candidate 3 (d=0.047, flips=-): batch_size=84, lr=0.002884, weight_decay=0.1404
    tr2_it2_c3 @ 20s seed=0 -> 1.9291 (230 steps)
    tr2_it2_c3 @ 40s seed=0 -> 1.7883 (450 steps)
    surrogate: L(80) ~= 1.7019
  best candidate 3: predicted improve -0.0431
  SKIP-CONFIRM (no predicted improvement in region); shrink
    tr2_inc_fresh @ 80s seed=5 -> 1.6654 (870 steps)

## Iteration 3  (delta=0.090 nats, F_inc=1.6601 n=5)
  candidate 0 (d=0.055, flips=-): batch_size=99, dropout=0.03071, lr=0.00346, weight_decay=0.1603
    tr2_it3_c0 @ 20s seed=0 -> 1.9299 (210 steps)
    tr2_it3_c0 @ 40s seed=0 -> 1.7887 (400 steps)
    surrogate: L(80) ~= 1.7020
  candidate 1 (d=0.055, flips=-): batch_size=142, lr=0.00209, weight_decay=0.07918
    tr2_it3_c1 @ 20s seed=0 -> 1.9420 (230 steps)
    tr2_it3_c1 @ 40s seed=0 -> 1.7713 (450 steps)
    surrogate: L(80) ~= 1.6777
  candidate 2 (d=0.085, flips=-): batch_size=200, dropout=0.01951, lr=0.001768, weight_decay=0.1131
    tr2_it3_c2 @ 20s seed=0 -> 2.0049 (180 steps)
    tr2_it3_c2 @ 40s seed=0 -> 1.8231 (350 steps)
    surrogate: L(80) ~= 1.7162
  candidate 3 (d=0.073, flips=['rmsnorm']): batch_size=131, lr=0.00284, rmsnorm->False, weight_decay=0.07705
    tr2_it3_c3 @ 20s seed=0 -> 1.9070 (230 steps)
    tr2_it3_c3 @ 40s seed=0 -> 1.7450 (480 steps)
    surrogate: L(80) ~= 1.6594
  best candidate 3: predicted improve 0.0006
    tr2_it3_confirm @ 80s seed=1 -> 1.6544 (930 steps)
    tr2_it3_confirm @ 80s seed=2 -> 1.6603 (910 steps)
  confirm: F_new = 1.6573 (n=2)  improve 0.0027 vs threshold 0.0131 (SD 0.0004, CI 0.0131)  rho=gated
  reject
  new delta = 0.045
    tr2_inc_fresh @ 80s seed=6 -> 1.6777 (830 steps)

## Iteration 4  (delta=0.045 nats, F_inc=1.6630 n=6)
  candidate 0 (d=0.040, flips=-): batch_size=108, lr=0.003601, weight_decay=0.07373
    tr2_it4_c0 @ 20s seed=0 -> 1.8950 (220 steps)
    tr2_it4_c0 @ 40s seed=0 -> 1.7514 (430 steps)
    surrogate: L(80) ~= 1.6701
  candidate 1 (d=0.044, flips=-): batch_size=130, dropout=0.01644, lr=0.003775, weight_decay=0.1376
    tr2_it4_c1 @ 20s seed=0 -> 1.8928 (200 steps)
    tr2_it4_c1 @ 40s seed=0 -> 1.7371 (400 steps)
    surrogate: L(80) ~= 1.6551
  candidate 2 (d=0.042, flips=-): batch_size=85, lr=0.002854, weight_decay=0.1275
    tr2_it4_c2 @ 20s seed=0 -> 1.9442 (220 steps)
    tr2_it4_c2 @ 40s seed=0 -> 1.7967 (430 steps)
    surrogate: L(80) ~= 1.7064
  candidate 3 (d=0.039, flips=-): batch_size=102, lr=0.002434, weight_decay=0.1208
    tr2_it4_c3 @ 20s seed=0 -> 1.9515 (220 steps)
    tr2_it4_c3 @ 40s seed=0 -> 1.8051 (440 steps)
    surrogate: L(80) ~= 1.7140
  best candidate 1: predicted improve 0.0079
    tr2_it4_confirm @ 80s seed=1 -> 1.6611 (750 steps)
    tr2_it4_confirm @ 80s seed=2 -> 1.6720 (770 steps)
  confirm: F_new = 1.6666 (n=2)  improve -0.0035 vs threshold 0.0128 (SD 0.0001, CI 0.0128)  rho=gated
  reject
  new delta = 0.040
    tr2_inc_fresh @ 80s seed=7 -> 1.6528 (840 steps)

## Iteration 5  (delta=0.040 nats, F_inc=1.6615 n=7)
  candidate 0 (d=0.018, flips=-): batch_size=120, dropout=0.009601, lr=0.002875, weight_decay=0.08471
    tr2_it5_c0 @ 20s seed=0 -> 1.9257 (200 steps)
    tr2_it5_c0 @ 40s seed=0 -> 1.7637 (410 steps)
    surrogate: L(80) ~= 1.6743
  candidate 1 (d=0.027, flips=-): batch_size=150, dropout=0.03396, lr=0.002901, weight_decay=0.1016
    tr2_it5_c1 @ 20s seed=0 -> 1.9417 (200 steps)
    tr2_it5_c1 @ 40s seed=0 -> 1.7858 (370 steps)
    surrogate: L(80) ~= 1.6943
  candidate 2 (d=0.017, flips=-): batch_size=154, lr=0.002903, weight_decay=0.09953
    tr2_it5_c2 @ 20s seed=0 -> 1.9534 (180 steps)
    tr2_it5_c2 @ 40s seed=0 -> 1.7576 (430 steps)
    surrogate: L(80) ~= 1.6602
  candidate 3 (d=0.024, flips=-): batch_size=104, lr=0.002824, weight_decay=0.0865
    tr2_it5_c3 @ 20s seed=0 -> 1.9032 (220 steps)
    tr2_it5_c3 @ 40s seed=0 -> 1.7448 (450 steps)
    surrogate: L(80) ~= 1.6604
  best candidate 2: predicted improve 0.0013
    tr2_it5_confirm @ 80s seed=1 -> 1.6433 (880 steps)
    tr2_it5_confirm @ 80s seed=2 -> 1.6483 (840 steps)
  confirm: F_new = 1.6458 (n=2)  improve 0.0158 vs threshold 0.0125 (SD 0.0001, CI 0.0125)  rho=gated
  ACCEPT -> new incumbent 1.6458 (L_inf 1.497)  [batch_size=154, lr=0.002903, weight_decay=0.09953]
  new delta = 0.040

## Final incumbent: 1.6458 (n=2)  [batch_size=154, lr=0.002903, weight_decay=0.09953]

## 1-flip optimality poll (all 5 flags, 80s, 1 seed)
    tr2_poll_cosine @ 80s seed=1 -> 1.6679 (820 steps)
  flip cosine -> 1.6679 (no gain)
    tr2_poll_rmsnorm @ 80s seed=1 -> 1.6340 (910 steps)
  flip rmsnorm -> 1.6340 (no gain)
    tr2_poll_rotary @ 80s seed=1 -> 1.7156 (960 steps)
  flip rotary -> 1.7156 (no gain)
    tr2_poll_swiglu @ 80s seed=1 -> 1.6909 (940 steps)
  flip swiglu -> 1.6909 (no gain)
    tr2_poll_weight_tying @ 80s seed=1 -> 1.6634 (850 steps)
  flip weight_tying -> 1.6634 (no gain)
  1-flip local optimality: CERTIFIED (LCB margin 0.0191)

## Budget-grid sweep of tr2_best
  tr2_best @ 10s -> 2.0902
  tr2_best @ 20s -> 1.9027
  tr2_best @ 40s -> 1.7470
  tr2_best @ 80s -> 1.6552

done; python analyze.py for the updated scaling plot
