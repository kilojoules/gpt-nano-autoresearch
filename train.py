"""Train GPT Nano under a wall-clock *training-time* budget.

The budget counts only time spent in training steps; evaluation is off the
clock (speedrun convention), so eval frequency doesn't distort comparisons.
Each run appends one JSON line to the results file with the final val loss and
the full (train_seconds, val_loss) curve.

Usage: python train.py --variant baseline --budget 20 --seed 0
"""
import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict, replace

import numpy as np
import torch

from model import GPT, GPTConfig
from variants import resolve

HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class TrainConfig:
    batch_size: int = 64
    lr: float = 5e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    lr_schedule: str = "constant"  # or "cosine"
    warmup_frac: float = 0.05      # fraction of the time budget
    min_lr_ratio: float = 0.1
    eval_batches: int = 16


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    n = int(0.9 * len(data))
    return data[:n], data[n:], len(chars)


def get_batch(data_t, batch_size, block_size, device, generator=None):
    ix = torch.randint(len(data_t) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data_t[i:i + block_size] for i in ix])
    y = torch.stack([data_t[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


def lr_at(frac_of_budget, tc):
    if tc.lr_schedule == "constant":
        return tc.lr
    if frac_of_budget < tc.warmup_frac:
        return tc.lr * frac_of_budget / tc.warmup_frac
    p = min((frac_of_budget - tc.warmup_frac) / (1.0 - tc.warmup_frac), 1.0)
    min_lr = tc.min_lr_ratio * tc.lr
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * p)) * (tc.lr - min_lr)


@torch.no_grad()
def evaluate(model, eval_batches, device):
    model.eval()
    losses = []
    for x, y in eval_batches:
        _, loss = model(x.to(device), y.to(device))
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def run(variant, budget, seed, device, data_path, out_path, block_size=128):
    model_ov, train_ov, desc = resolve(variant)
    mc = GPTConfig(block_size=block_size)
    tc = TrainConfig()
    mc = replace(mc, **model_ov)
    tc = replace(tc, **train_ov)

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_np, val_np, vocab_size = load_data(data_path)
    mc = replace(mc, vocab_size=vocab_size)
    train_t = torch.from_numpy(train_np)
    val_t = torch.from_numpy(val_np)

    # fixed eval set, identical across all runs/variants (batch size pinned to
    # 64 so variants that change tc.batch_size still see the same eval data)
    g = torch.Generator().manual_seed(1234)
    eval_set = [get_batch(val_t, 64, mc.block_size, "cpu", generator=g)
                for _ in range(tc.eval_batches)]

    model = GPT(mc).to(device)
    opt = model.configure_optimizer(tc.lr, tc.weight_decay, (tc.beta1, tc.beta2))

    # one warmup step off the clock (kernel compilation / allocator warmup)
    x, y = get_batch(train_t, tc.batch_size, mc.block_size, device)
    _, loss = model(x, y)
    loss.backward()
    opt.zero_grad(set_to_none=True)

    curve = []
    train_time = 0.0
    eval_every = max(2.0, budget / 8.0)
    next_eval = eval_every
    steps = 0
    sync_every = 10  # sync device clock every N steps so MPS can pipeline
    train_loss = float("nan")
    while train_time < budget:
        lr = lr_at(train_time / budget, tc)
        for group in opt.param_groups:
            group["lr"] = lr
        t0 = time.perf_counter()
        for _ in range(sync_every):
            x, y = get_batch(train_t, tc.batch_size, mc.block_size, device)
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if tc.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            opt.step()
            steps += 1
        train_loss = loss.item()  # forces device sync -> honest chunk timing
        train_time += time.perf_counter() - t0
        if train_time >= next_eval or train_time >= budget:
            val = evaluate(model, eval_set, device)
            curve.append([round(train_time, 3), round(val, 5)])
            next_eval += eval_every

    result = {
        "variant": variant,
        "desc": desc,
        "budget_s": budget,
        "seed": seed,
        "device": device,
        "final_val_loss": curve[-1][1],
        "steps": steps,
        "steps_per_s": round(steps / train_time, 2),
        "tokens_seen": steps * tc.batch_size * mc.block_size,
        "n_params": model.num_params(),
        "final_train_loss": round(train_loss, 5),
        "curve": curve,
        "model_config": asdict(mc),
        "train_config": asdict(tc),
    }
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "a") as f:
            f.write(json.dumps(result) + "\n")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="baseline")
    p.add_argument("--budget", type=float, required=True, help="training-time budget, seconds")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--data", default=os.path.join(HERE, "data", "tinyshakespeare.txt"))
    p.add_argument("--out", default=os.path.join(HERE, "results", "results.jsonl"))
    args = p.parse_args()

    r = run(args.variant, args.budget, args.seed, args.device, args.data, args.out)
    print("%s budget=%ss seed=%s -> val %.4f (%d steps, %.1f steps/s, %.2fM tokens)" % (
        r["variant"], r["budget_s"], r["seed"], r["final_val_loss"],
        r["steps"], r["steps_per_s"], r["tokens_seen"] / 1e6))


if __name__ == "__main__":
    main()
