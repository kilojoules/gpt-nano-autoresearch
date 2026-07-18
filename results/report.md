# GPT Nano autoresearch: scaling laws for improvements

Each variant trained under wall-clock training-time budgets (eval off the clock), char-level tiny Shakespeare, Apple M1 Pro (MPS), 1 seed.

Fitted law: `L(t) = L_inf + A * t^-alpha`. *Effective time multiplier* = how much training time the baseline would need to match the variant's loss at t=80s (from the baseline fit; extrapolated when beyond the measured range).

| variant | L @ 10s | L @ 20s | L @ 40s | L @ 80s | L_inf | A | alpha | eff. time x |
|---|---|---|---|---|---|---|---|---|
| tr_best | 2.1065 | 1.8539 | 1.7044 | 1.6155 | 1.486 | 3.523 | 0.754 | 117.92x |
| combo | 2.1380 | 1.9217 | 1.7763 | 1.6650 | 1.407 | 2.298 | 0.498 | 61.60x |
| lr_3e3 | 2.4397 | 2.3094 | 2.1366 | 1.9507 | 0.000 | 3.143 | 0.106 | 4.51x |
| rotary | 2.7317 | 2.3696 | 2.1390 | 1.9665 | 1.625 | 4.023 | 0.561 | 4.04x |
| lr_1e3 | 2.5169 | 2.4171 | 2.2612 | 2.0758 | 0.000 | 3.138 | 0.091 | 2.01x |
| llama_mlp | 2.8092 | 2.5034 | 2.3585 | 2.1217 | 1.188 | 2.897 | 0.255 | 1.55x |
| dropout0 | 2.6617 | 2.4520 | 2.3229 | 2.1227 | 0.000 | 3.388 | 0.105 | 1.54x |
| bs128 | 2.6816 | 2.4689 | 2.3202 | 2.1304 | 0.000 | 3.437 | 0.108 | 1.47x |
| baseline | 2.7260 | 2.4882 | 2.3848 | 2.1907 | 1.281 | 2.342 | 0.213 | — |
| cosine | 3.0200 | 2.6394 | 2.4501 | 2.3265 | 2.198 | 6.171 | 0.876 | 0.56x |
| wtie | 2.7340 | 2.5263 | 2.4330 | 2.3266 | 2.183 | 2.280 | 0.620 | 0.55x |

`*` = fallback grid fit (curve_fit did not converge).

![scaling](scaling.png)

## Variant descriptions

- **tr_best** — trust-region search result: batch_size=176, dropout=0.01289, lr=0.005352, rmsnorm->False, weight_decay=0.1654
- **combo** — stack of the round-1 winners: lr 3e-3 + rotary + RMSNorm/SwiGLU + dropout 0 + batch 128 (cosine and weight tying excluded — both hurt at these budgets)
- **lr_3e3** — learning rate 5e-4 -> 3e-3
- **rotary** — rotary position embeddings instead of learned absolute
- **lr_1e3** — learning rate 5e-4 -> 1e-3
- **llama_mlp** — RMSNorm + SwiGLU MLP (param-matched), llama-style
- **dropout0** — dropout 0.1 -> 0.0 (regularization hurts in the short-budget regime)
- **bs128** — batch size 64 -> 128 (better accelerator utilization per wall-clock second; fewer optimizer steps)
- **baseline** — minGPT gpt-nano: 3L/3H/48d, constant LR 5e-4, dropout 0.1, LayerNorm+GELU, learned pos emb, no weight tying
- **cosine** — budget-aware cosine LR decay with 5% warmup (schedule driven by fraction of time budget used, not step count)
- **wtie** — tie token embedding and lm_head weights
