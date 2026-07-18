"""Registry of candidate improvements for the autoresearch loop.

Each variant is a *single* change vs baseline (clean attribution), expressed as
overrides on GPTConfig ("model") and/or TrainConfig ("train"). `combo` variants
stack adopted winners and are added programmatically or by hand after a round.
"""

VARIANTS = {
    "baseline": {
        "desc": "minGPT gpt-nano: 3L/3H/48d, constant LR 5e-4, dropout 0.1, "
                "LayerNorm+GELU, learned pos emb, no weight tying",
        "model": {},
        "train": {},
    },
    "lr_1e3": {
        "desc": "learning rate 5e-4 -> 1e-3",
        "train": {"lr": 1e-3},
    },
    "lr_3e3": {
        "desc": "learning rate 5e-4 -> 3e-3",
        "train": {"lr": 3e-3},
    },
    "cosine": {
        "desc": "budget-aware cosine LR decay with 5% warmup (schedule driven "
                "by fraction of time budget used, not step count)",
        "train": {"lr_schedule": "cosine"},
    },
    "dropout0": {
        "desc": "dropout 0.1 -> 0.0 (regularization hurts in the short-budget regime)",
        "model": {"dropout": 0.0},
    },
    "wtie": {
        "desc": "tie token embedding and lm_head weights",
        "model": {"weight_tying": True},
    },
    "llama_mlp": {
        "desc": "RMSNorm + SwiGLU MLP (param-matched), llama-style",
        "model": {"rmsnorm": True, "swiglu": True},
    },
    "rotary": {
        "desc": "rotary position embeddings instead of learned absolute",
        "model": {"rotary": True},
    },
    "bs128": {
        "desc": "batch size 64 -> 128 (better accelerator utilization per "
                "wall-clock second; fewer optimizer steps)",
        "train": {"batch_size": 128},
    },
}


def resolve(name):
    if name not in VARIANTS:
        raise KeyError("unknown variant %r; known: %s" % (name, sorted(VARIANTS)))
    v = VARIANTS[name]
    return v.get("model", {}), v.get("train", {}), v.get("desc", "")


def register_combo(name, desc, model_overrides, train_overrides):
    VARIANTS[name] = {"desc": desc, "model": model_overrides, "train": train_overrides}
