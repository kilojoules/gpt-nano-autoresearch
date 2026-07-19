
# Single-fidelity control arm 2026-07-18 17:12
no screening: every candidate pays 80 s x 2 reps; cap 2300 s

    sf_inc0 @ 80s seed=1 -> 1.6563 (spent 80s)
    sf_inc0 @ 80s seed=2 -> 1.6697 (spent 160s)
  F_inc = 1.6630 (n=2)
    sf_inc_fresh @ 80s seed=3 -> 1.6728 (spent 240s)

## Iteration 1  (delta=0.250, F_inc=1.6663 n=3, spent 240s)
  candidate (d=0.209, flips=-): batch_size=133, lr=0.00204, weight_decay=0.009757
    sf_it1 @ 80s seed=1 -> 1.6673 (spent 320s)
    sf_it1 @ 80s seed=2 -> 1.6689 (spent 400s)
  F_new = 1.6681 (n=2)  improve -0.0018 vs threshold 0.0143 -> reject
    sf_inc_fresh @ 80s seed=4 -> 1.6401 (spent 480s)

## Iteration 2  (delta=0.125, F_inc=1.6598 n=4, spent 480s)
  candidate (d=0.125, flips=['rmsnorm', 'swiglu']): batch_size=167, dropout=0.009142, lr=0.002191, rmsnorm->False, swiglu->False, weight_decay=0.1994
    sf_it2 @ 80s seed=1 -> 1.7238 (spent 560s)
    sf_it2 @ 80s seed=2 -> 1.7049 (spent 640s)
  F_new = 1.7144 (n=2)  improve -0.0546 vs threshold 0.0135 -> reject
    sf_inc_fresh @ 80s seed=5 -> 1.6685 (spent 720s)

## Iteration 3  (delta=0.062, F_inc=1.6615 n=5, spent 720s)
  candidate (d=0.043, flips=-): batch_size=188, dropout=0.03226, lr=0.002797, weight_decay=0.1165
    sf_it3 @ 80s seed=1 -> 1.6653 (spent 800s)
    sf_it3 @ 80s seed=2 -> 1.6610 (spent 880s)
  F_new = 1.6632 (n=2)  improve -0.0017 vs threshold 0.0131 -> reject
    sf_inc_fresh @ 80s seed=6 -> 1.6576 (spent 960s)

## Iteration 4  (delta=0.040, F_inc=1.6608 n=6, spent 960s)
  candidate (d=0.021, flips=-): batch_size=144, lr=0.00341, weight_decay=0.1042
    sf_it4 @ 80s seed=1 -> 1.6471 (spent 1040s)
    sf_it4 @ 80s seed=2 -> 1.6534 (spent 1120s)
  F_new = 1.6503 (n=2)  improve 0.0106 vs threshold 0.0128 -> reject
    sf_inc_fresh @ 80s seed=7 -> 1.6433 (spent 1200s)

## Iteration 5  (delta=0.040, F_inc=1.6583 n=7, spent 1200s)
  candidate (d=0.027, flips=-): batch_size=147, dropout=0.01039, lr=0.002998, weight_decay=0.07663
    sf_it5 @ 80s seed=1 -> 1.6279 (spent 1280s)
    sf_it5 @ 80s seed=2 -> 1.6504 (spent 1360s)
  F_new = 1.6392 (n=2)  improve 0.0192 vs threshold 0.0125 -> ACCEPT
    sf_inc_fresh @ 80s seed=3 -> 1.6606 (spent 1440s)

## Iteration 6  (delta=0.064, F_inc=1.6463 n=3, spent 1440s)
  candidate (d=0.033, flips=-): batch_size=133, dropout=0.003579, lr=0.003746, weight_decay=0.08203
    sf_it6 @ 80s seed=1 -> 1.6416 (spent 1520s)
    sf_it6 @ 80s seed=2 -> 1.6518 (spent 1600s)
  F_new = 1.6467 (n=2)  improve -0.0004 vs threshold 0.0143 -> reject
    sf_inc_fresh @ 80s seed=4 -> 1.6341 (spent 1680s)

## Iteration 7  (delta=0.040, F_inc=1.6433 n=4, spent 1680s)
  candidate (d=0.034, flips=-): batch_size=132, dropout=0, lr=0.003345, weight_decay=0.1057
    sf_it7 @ 80s seed=1 -> 1.6590 (spent 1760s)
    sf_it7 @ 80s seed=2 -> 1.6637 (spent 1840s)
  F_new = 1.6613 (n=2)  improve -0.0181 vs threshold 0.0135 -> reject
    sf_inc_fresh @ 80s seed=5 -> 1.6533 (spent 1920s)

## Iteration 8  (delta=0.040, F_inc=1.6453 n=5, spent 1920s)
  candidate (d=0.019, flips=-): batch_size=129, dropout=0.01987, lr=0.002917, weight_decay=0.06589
    sf_it8 @ 80s seed=1 -> 1.6478 (spent 2000s)
    sf_it8 @ 80s seed=2 -> 1.6718 (spent 2080s)
  F_new = 1.6598 (n=2)  improve -0.0145 vs threshold 0.0131 -> reject

## Final: 1.6453 (n=5) after 2080s  [batch_size=147, dropout=0.01039, lr=0.002998, weight_decay=0.07663]
