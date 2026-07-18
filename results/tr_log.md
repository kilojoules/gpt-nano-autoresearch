
# Trust-region autoresearch 2026-07-18 14:47
start = combo winner; surrogate budgets [10, 20], confirm 80s, delta0=1.00

## Iteration 0: score incumbent
    tr_incumbent0 @ 10s -> 2.1044 (110 steps)
    tr_incumbent0 @ 20s -> 1.8931 (230 steps)
    tr_incumbent0 @ 80s -> 1.6509 (920 steps)
  incumbent objective (confirm) = 1.6509, L_inf = 1.441

## Iteration 1  (delta=1.00, incumbent 1.6509)
  candidate 0: batch_size=68, weight_decay=0.1544
    tr_it1_c0 @ 10s -> 2.1093 (120 steps)
    tr_it1_c0 @ 20s -> 1.9145 (240 steps)
    surrogate: L(80) ~= 1.6787, objective ~= 1.6787
  candidate 1: batch_size=176, lr=0.004924, rmsnorm->False, weight_decay=0.1654
    tr_it1_c1 @ 10s -> 2.0316 (120 steps)
    tr_it1_c1 @ 20s -> 1.7969 (240 steps)
    surrogate: L(80) ~= 1.5703, objective ~= 1.5703
  candidate 2: dropout=0.04972, lr=0.003801
    tr_it1_c2 @ 10s -> 2.0930 (110 steps)
    tr_it1_c2 @ 20s -> 1.8968 (220 steps)
    surrogate: L(80) ~= 1.6637, objective ~= 1.6637
  candidate 3: batch_size=98, lr=0.004078, weight_decay=0.1716
    tr_it1_c3 @ 10s -> 2.0850 (120 steps)
    tr_it1_c3 @ 20s -> 1.8811 (230 steps)
    surrogate: L(80) ~= 1.6465, objective ~= 1.6465
  candidate 4: batch_size=152, lr=0.005153, swiglu->False, weight_decay=0.1586
    tr_it1_c4 @ 10s -> 2.1210 (130 steps)
    tr_it1_c4 @ 20s -> 1.9562 (250 steps)
    surrogate: L(80) ~= 1.7368, objective ~= 1.7368
  best candidate 1 (batch_size=176, lr=0.004924, rmsnorm->False, weight_decay=0.1654): predicted objective 1.5703 (improve 0.0807)
    tr_it1_confirm @ 80s -> 1.6421 (890 steps)
  confirm: objective = 1.6421  (actual improve 0.0089, rho = 0.11)
  ACCEPT -> new incumbent 1.6421 (L_inf 1.602)  [batch_size=176, lr=0.004924, rmsnorm->False, weight_decay=0.1654]
  new delta = 0.50

## Iteration 2  (delta=0.50, incumbent 1.6421)
  candidate 0: batch_size=216, lr=0.003864, rotary->False, weight_decay=0.1213
    tr_it2_c0 @ 10s -> 2.3249 (120 steps)
    tr_it2_c0 @ 20s -> 2.0297 (240 steps)
    surrogate: L(80) ~= 1.7517, objective ~= 1.7517
  candidate 1: lr=0.004283, weight_decay=0.2248
    tr_it2_c1 @ 10s -> 2.1341 (100 steps)
    tr_it2_c1 @ 20s -> 1.8884 (190 steps)
    surrogate: L(80) ~= 1.6850, objective ~= 1.6850
  candidate 2: dropout=0.01289, lr=0.005352
    tr_it2_c2 @ 10s -> 2.1920 (80 steps)
    tr_it2_c2 @ 20s -> 1.8820 (190 steps)
    surrogate: L(80) ~= 1.6650, objective ~= 1.6650
  candidate 3: rmsnorm->True, weight_decay=0.1939
    tr_it2_c3 @ 10s -> 2.1263 (100 steps)
    tr_it2_c3 @ 20s -> 1.8806 (200 steps)
    surrogate: L(80) ~= 1.6806, objective ~= 1.6806
  candidate 4: dropout=0.02363
    tr_it2_c4 @ 10s -> 2.0787 (110 steps)
    tr_it2_c4 @ 20s -> 1.8539 (200 steps)
    surrogate: L(80) ~= 1.6723, objective ~= 1.6723
  best candidate 2 (dropout=0.01289, lr=0.005352): predicted objective 1.6650 (improve -0.0229)
    tr_it2_confirm @ 80s -> 1.6282 (800 steps)
  confirm: objective = 1.6282  (actual improve 0.0139, rho = 0.00)
  ACCEPT -> new incumbent 1.6282 (L_inf 1.527)  [batch_size=176, dropout=0.01289, lr=0.005352, rmsnorm->False, weight_decay=0.1654]
  new delta = 0.25

## Iteration 3  (delta=0.25, incumbent 1.6282)
  candidate 0: batch_size=178, dropout=0.01869, weight_decay=0.1921
    tr_it3_c0 @ 10s -> 2.0872 (110 steps)
    tr_it3_c0 @ 20s -> 1.9027 (190 steps)
    surrogate: L(80) ~= 1.6960, objective ~= 1.6960
  candidate 1: batch_size=190, dropout=0.02361, weight_decay=0.1809
    tr_it3_c1 @ 10s -> 2.1622 (80 steps)
    tr_it3_c1 @ 20s -> 1.8785 (190 steps)
    surrogate: L(80) ~= 1.6347, objective ~= 1.6347
  candidate 2: batch_size=167, dropout=0.006858, lr=0.006024, weight_decay=0.1944
    tr_it3_c2 @ 10s -> 2.0701 (110 steps)
    tr_it3_c2 @ 20s -> 1.8536 (210 steps)
    surrogate: L(80) ~= 1.6452, objective ~= 1.6452
  candidate 3: batch_size=155, dropout=0.02063, lr=0.006122, weight_decay=0.1705
    tr_it3_c3 @ 10s -> 2.0862 (110 steps)
    tr_it3_c3 @ 20s -> 1.8948 (200 steps)
    surrogate: L(80) ~= 1.6862, objective ~= 1.6862
  candidate 4: batch_size=197, lr=0.004973
    tr_it3_c4 @ 10s -> 2.0762 (100 steps)
    tr_it3_c4 @ 20s -> 1.8812 (190 steps)
    surrogate: L(80) ~= 1.6743, objective ~= 1.6743
  best candidate 1 (batch_size=190, dropout=0.02361, weight_decay=0.1809): predicted objective 1.6347 (improve -0.0065)
    tr_it3_confirm @ 80s -> 1.6414 (730 steps)
  confirm: objective = 1.6414  (actual improve -0.0132, rho = 0.00)
  reject (did not beat incumbent by 0.002)
  new delta = 0.12

## Iteration 4  (delta=0.12, incumbent 1.6282)
  candidate 0: batch_size=187, lr=0.005401, weight_decay=0.1628
    tr_it4_c0 @ 10s -> 2.0930 (100 steps)
    tr_it4_c0 @ 20s -> 1.8770 (190 steps)
    surrogate: L(80) ~= 1.6609, objective ~= 1.6609
  candidate 1: batch_size=186, dropout=0.01479, lr=0.005659, weight_decay=0.1616
    tr_it4_c1 @ 10s -> 2.5099 (30 steps)
    screened: |dL(10s)| = 0.318 > behav-delta 0.300
  candidate 2: dropout=0.0114, lr=0.005543, weight_decay=0.1762
    tr_it4_c2 @ 10s -> 2.0957 (110 steps)
    tr_it4_c2 @ 20s -> 1.8752 (210 steps)
    surrogate: L(80) ~= 1.6576, objective ~= 1.6576
  candidate 3: dropout=0.01001
    tr_it4_c3 @ 10s -> 2.1164 (100 steps)
    tr_it4_c3 @ 20s -> 1.8676 (210 steps)
    surrogate: L(80) ~= 1.6408, objective ~= 1.6408
  candidate 4: weight_decay=0.161
    tr_it4_c4 @ 10s -> 2.0931 (100 steps)
    tr_it4_c4 @ 20s -> 1.8543 (210 steps)
    surrogate: L(80) ~= 1.6364, objective ~= 1.6364
  best candidate 4 (weight_decay=0.161): predicted objective 1.6364 (improve -0.0082)
    tr_it4_confirm @ 80s -> 1.6396 (800 steps)
  confirm: objective = 1.6396  (actual improve -0.0114, rho = 0.00)
  reject (did not beat incumbent by 0.002)
  new delta = 0.10

## Iteration 5  (delta=0.10, incumbent 1.6282)
  candidate 0: lr=0.005121
    tr_it5_c0 @ 10s -> 2.0914 (100 steps)
    tr_it5_c0 @ 20s -> 1.8539 (200 steps)
    surrogate: L(80) ~= 1.6367, objective ~= 1.6367
  candidate 1: dropout=0.01184, lr=0.005092
    tr_it5_c1 @ 10s -> 2.0952 (100 steps)
    tr_it5_c1 @ 20s -> 1.8498 (200 steps)
    surrogate: L(80) ~= 1.6313, objective ~= 1.6313
  candidate 2: batch_size=178, dropout=0.008776, weight_decay=0.166
    tr_it5_c2 @ 10s -> 2.0975 (100 steps)
    tr_it5_c2 @ 20s -> 1.8632 (200 steps)
    surrogate: L(80) ~= 1.6438, objective ~= 1.6438
  candidate 3: batch_size=164
    tr_it5_c3 @ 10s -> 2.0883 (100 steps)
    tr_it5_c3 @ 20s -> 1.8957 (200 steps)
    surrogate: L(80) ~= 1.6861, objective ~= 1.6861
  candidate 4: batch_size=180, lr=0.00522, weight_decay=0.1672
    tr_it5_c4 @ 10s -> 2.0558 (110 steps)
    tr_it5_c4 @ 20s -> 1.8529 (210 steps)
    surrogate: L(80) ~= 1.6508, objective ~= 1.6508
  best candidate 1 (dropout=0.01184, lr=0.005092): predicted objective 1.6313 (improve -0.0031)
    tr_it5_confirm @ 80s -> 1.6355 (790 steps)
  confirm: objective = 1.6355  (actual improve -0.0073, rho = 0.00)
  reject (did not beat incumbent by 0.002)
  new delta = 0.10

## Iteration 6  (delta=0.10, incumbent 1.6282)
  candidate 0: dropout=0.01177
    tr_it6_c0 @ 10s -> 2.0938 (100 steps)
    tr_it6_c0 @ 20s -> 1.8633 (200 steps)
    surrogate: L(80) ~= 1.6454, objective ~= 1.6454
  candidate 1: batch_size=173, dropout=0.01701, lr=0.005522
    tr_it6_c1 @ 10s -> 2.0596 (110 steps)
    tr_it6_c1 @ 20s -> 1.8720 (210 steps)
    surrogate: L(80) ~= 1.6719, objective ~= 1.6719
  candidate 2: dropout=0.01657, weight_decay=0.1559
    tr_it6_c2 @ 10s -> 2.0603 (110 steps)
    tr_it6_c2 @ 20s -> 1.8556 (210 steps)
    surrogate: L(80) ~= 1.6518, objective ~= 1.6518
  candidate 3: batch_size=168, dropout=0.015
    tr_it6_c3 @ 10s -> 2.0901 (110 steps)
    tr_it6_c3 @ 20s -> 1.8726 (210 steps)
    surrogate: L(80) ~= 1.6572, objective ~= 1.6572
  candidate 4: dropout=0.01597, lr=0.005357, weight_decay=0.1673
    tr_it6_c4 @ 10s -> 2.0777 (110 steps)
    tr_it6_c4 @ 20s -> 1.8628 (210 steps)
    surrogate: L(80) ~= 1.6519, objective ~= 1.6519
  best candidate 0 (dropout=0.01177): predicted objective 1.6454 (improve -0.0172)
    tr_it6_confirm @ 80s -> 1.6281 (810 steps)
  confirm: objective = 1.6281  (actual improve 0.0001, rho = 0.00)
  reject (did not beat incumbent by 0.002)
  new delta = 0.10

## Final incumbent: 1.6282  [batch_size=176, dropout=0.01289, lr=0.005352, rmsnorm->False, weight_decay=0.1654]

## Confirm sweep of tr_best across the budget grid
  tr_best @ 10s -> 2.1065
  tr_best @ 20s -> 1.8539
  tr_best @ 40s -> 1.7044
  tr_best @ 80s -> 1.6155

done; run `python analyze.py` for the updated scaling plot
