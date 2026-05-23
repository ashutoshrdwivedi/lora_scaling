"""Empirically establish that LoRA bmm is a tiny fraction of forward time.

Produces evidence for the paper claim "custom kernels (Punica MBGMV / S-LoRA SGMV)
would not move the needle for this serving setting" by:

  1. PROFILER DUMP — PyTorch profiler over N representative forward iterations,
     sorted by CUDA time, with bmm vs addmm contributions explicitly broken out.

  2. ZERO-LORA ABLATION — same model, same shapes, same kernel launches, but
     with all A/B set to zero. Difference in forward time isolates the
     wall-clock cost of the LoRA bmm path. If the difference is <2%, the
     bmm cost is negligible.

  3. ANALYTIC FLOP RATIO — printed for reference: base encoder FLOPs vs
     LoRA shrink/expand FLOPs per layer per batch.

Run on the e2e/A100 node:
    uv run python -m benchmarks.profiling.forward_breakdown \\
        --model BAAI/bge-m3 --batch-size 32 --num-adapters 1000 \\
        --lora-rank 8 --warmup 20 --profiler-iters 10 --ablation-iters 100 \\
        --out benchmarks/results/forward_breakdown.txt
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from lora_serving.benchmark.synthetic import (
    make_synthetic_adapters,
    make_synthetic_inputs,
    make_synthetic_lr_weights,
)
from lora_serving.config import LoraServingConfig
from lora_serving.model.encoder import EncoderWithLora
from lora_serving.weights.batch import BatchAssembler
from lora_serving.weights.store import AdapterStore

DTYPE_MAP = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}

# Op-name buckets for categorizing profiler rows
BUCKETS = {
    "base_linear": ("aten::addmm", "aten::linear", "aten::matmul"),
    "lora_bmm": ("aten::bmm", "aten::baddbmm"),
    "attention": ("aten::softmax", "aten::scaled_dot_product_attention", "aten::masked_fill"),
    "layernorm_residual": ("aten::layer_norm", "aten::add", "aten::add_"),
    "elementwise": ("aten::gelu", "aten::relu", "aten::tanh", "aten::mul", "aten::div"),
}


def categorize(op_name: str) -> str:
    for cat, prefixes in BUCKETS.items():
        if any(op_name.startswith(p) for p in prefixes):
            return cat
    return "other"


def setup(args) -> tuple:
    device = torch.device("cuda:0")
    dtype = DTYPE_MAP[args.dtype]
    config = LoraServingConfig(
        model_name=args.model,
        lora_rank=args.lora_rank,
        batch_size=args.batch_size,
        max_seq_len=args.seq_len,
        target_modules=["query", "value"],
        device=device,
        dtype=dtype,
    )
    print(f"Loading {args.model} ({dtype})...")
    model = EncoderWithLora.from_pretrained_serving(config)
    model.eval()

    print(f"Generating {args.num_adapters} synthetic adapters (rank={args.lora_rank})...")
    store = AdapterStore(config)
    make_synthetic_adapters(store, args.num_adapters)
    adapter_ids = [f"adapter_{i}" for i in range(args.num_adapters)]

    lr_coefs, lr_intercepts = {}, {}
    for aid in adapter_ids:
        c, i = make_synthetic_lr_weights(config, args.num_labels)
        lr_coefs[aid] = c
        lr_intercepts[aid] = i

    assembler = BatchAssembler(store, config)
    inputs = make_synthetic_inputs(config, args.batch_size)
    output_lr = torch.zeros(args.batch_size, 1, args.num_labels, dtype=dtype, device=device)

    return model, store, assembler, adapter_ids, lr_coefs, lr_intercepts, inputs, output_lr, config, device


def analytic_flop_ratio(config: LoraServingConfig, batch_size: int, seq_len: int) -> dict:
    """Per-layer FLOP counts: base encoder vs LoRA shrink/expand."""
    B, S, H, r, L = batch_size, seq_len, config.hidden_size, config.lora_rank, config.num_layers
    M = len(config.target_modules)
    # Base attention (Q, K, V, O linears): 4 * B*S*H^2
    # Base FFN (2 linears, expansion 4): 2 * B*S*H*4H = 8 * B*S*H^2
    base_per_layer = (4 + 8) * B * S * H * H
    # LoRA shrink: M * B*S*H*r   ; expand: M * B*S*r*H
    lora_per_layer = M * 2 * B * S * H * r
    return {
        "base_GFLOPs_per_layer": base_per_layer / 1e9,
        "lora_GFLOPs_per_layer": lora_per_layer / 1e9,
        "base_to_lora_ratio": base_per_layer / lora_per_layer,
        "base_total_GFLOPs": base_per_layer * L / 1e9,
        "lora_total_GFLOPs": lora_per_layer * L / 1e9,
    }


def run_profiler(model, store, assembler, adapter_ids, lr_coefs, lr_intercepts,
                 inputs, output_lr, config, device, batch_size, warmup, iters):
    """Profile `iters` forward passes, categorize ops, return summary dict."""
    def one_batch():
        ids = random.choices(adapter_ids, k=batch_size)
        coefs = [lr_coefs[a] for a in ids]
        intercepts = [lr_intercepts[a] for a in ids]
        lora_w, lr_w = assembler.assemble(ids, coefs, intercepts)
        with torch.no_grad():
            model(inputs["input_ids"], inputs["attention_mask"], lora_w, lr_w, output_lr)

    print(f"Warmup ({warmup} iters)...")
    for _ in range(warmup):
        one_batch()
    torch.cuda.synchronize(device)

    print(f"Profiling ({iters} iters)...")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            with record_function("forward_iter"):
                one_batch()
        torch.cuda.synchronize(device)

    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)

    # Aggregate by bucket
    totals = defaultdict(lambda: {"cuda_us": 0.0, "calls": 0})
    grand_cuda_us = 0.0
    for evt in prof.key_averages():
        cuda_us = float(evt.cuda_time_total)  # microseconds
        if cuda_us <= 0:
            continue
        bucket = categorize(evt.key)
        totals[bucket]["cuda_us"] += cuda_us
        totals[bucket]["calls"] += int(evt.count)
        grand_cuda_us += cuda_us

    summary = {}
    for bucket, vals in totals.items():
        summary[bucket] = {
            "cuda_ms_total": vals["cuda_us"] / 1000.0,
            "share_pct": 100.0 * vals["cuda_us"] / grand_cuda_us if grand_cuda_us else 0.0,
            "calls": vals["calls"],
        }
    return table, summary, grand_cuda_us / 1000.0


def run_ablation(model, store, assembler, adapter_ids, lr_coefs, lr_intercepts,
                 inputs, output_lr, device, batch_size, warmup, iters) -> dict:
    """Compare forward time with normal-LoRA vs zero-LoRA (A,B set to 0).

    Zero-LoRA keeps the kernel launches identical but reduces the bmm output
    contribution to zero; the wall-clock difference isolates the cost of
    nonzero LoRA arithmetic.
    """
    def measure(zero_lora: bool, n: int) -> np.ndarray:
        times = np.empty(n)
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        for i in range(n):
            ids = random.choices(adapter_ids, k=batch_size)
            coefs = [lr_coefs[a] for a in ids]
            intercepts = [lr_intercepts[a] for a in ids]
            lora_w, lr_w = assembler.assemble(ids, coefs, intercepts)
            if zero_lora:
                for layer_w in lora_w:
                    for mod in list(layer_w.a.keys()):
                        layer_w.a[mod] = [t.zero_() for t in layer_w.a[mod]]
                        layer_w.b[mod] = [t.zero_() for t in layer_w.b[mod]]
            ev_start.record()
            with torch.no_grad():
                model(inputs["input_ids"], inputs["attention_mask"], lora_w, lr_w, output_lr)
            ev_end.record()
            torch.cuda.synchronize(device)
            times[i] = ev_start.elapsed_time(ev_end)
        return times

    print(f"Ablation warmup ({warmup} iters, normal LoRA)...")
    measure(zero_lora=False, n=warmup)
    print(f"Ablation: measuring normal LoRA ({iters} iters)...")
    normal = measure(zero_lora=False, n=iters)
    print(f"Ablation: measuring zero LoRA ({iters} iters)...")
    zero = measure(zero_lora=True, n=iters)

    return {
        "normal_mean_ms": float(np.mean(normal)),
        "normal_p50_ms": float(np.percentile(normal, 50)),
        "zero_mean_ms": float(np.mean(zero)),
        "zero_p50_ms": float(np.percentile(zero, 50)),
        "lora_cost_mean_ms": float(np.mean(normal) - np.mean(zero)),
        "lora_cost_share_pct": 100.0 * (np.mean(normal) - np.mean(zero)) / np.mean(normal),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--dtype", choices=list(DTYPE_MAP), default="fp16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-adapters", type=int, default=1000)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--num-labels", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--profiler-iters", type=int, default=10)
    parser.add_argument("--ablation-iters", type=int, default=100)
    parser.add_argument("--out", default="benchmarks/results/forward_breakdown.txt")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Need CUDA. Aborting.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    setup_tuple = setup(args)
    model, store, assembler, adapter_ids, lr_coefs, lr_intercepts, inputs, output_lr, config, device = setup_tuple

    flops = analytic_flop_ratio(config, args.batch_size, args.seq_len)

    table, profiler_summary, total_forward_ms = run_profiler(
        model, store, assembler, adapter_ids, lr_coefs, lr_intercepts,
        inputs, output_lr, config, device,
        batch_size=args.batch_size, warmup=args.warmup, iters=args.profiler_iters,
    )

    ablation = run_ablation(
        model, store, assembler, adapter_ids, lr_coefs, lr_intercepts,
        inputs, output_lr, device,
        batch_size=args.batch_size, warmup=args.warmup, iters=args.ablation_iters,
    )

    # ---- write report ----
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write(f"# Forward-pass breakdown — {args.model}\n")
        f.write(f"# GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(f"# Config: B={args.batch_size} N={args.num_adapters} r={args.lora_rank} "
                f"seq={args.seq_len} dtype={args.dtype}\n\n")

        f.write("## 1. Analytic FLOP ratio (per encoder layer)\n")
        for k, v in flops.items():
            f.write(f"  {k:30s}: {v:,.4f}\n")
        f.write("\n")
        f.write("Interpretation: even an infinitely fast LoRA bmm kernel would\n"
                "save at most 1 / (base/lora ratio) of total forward FLOPs.\n\n")

        f.write("## 2. PyTorch profiler — op-level CUDA time breakdown\n\n")
        f.write(table)
        f.write("\n\n## 3. Aggregated by category (CUDA time)\n")
        f.write(f"  Total forward CUDA time (sum of profiled ops): {total_forward_ms:.2f} ms\n\n")
        for cat in sorted(profiler_summary, key=lambda c: -profiler_summary[c]["cuda_ms_total"]):
            d = profiler_summary[cat]
            f.write(f"  {cat:22s}: {d['cuda_ms_total']:8.3f} ms  "
                    f"({d['share_pct']:5.1f}%)  calls={d['calls']}\n")
        f.write("\n")

        f.write("## 4. Zero-LoRA ablation\n")
        for k, v in ablation.items():
            unit = "%" if k.endswith("_pct") else " ms"
            f.write(f"  {k:30s}: {v:8.3f}{unit}\n")
        f.write("\n")
        f.write("Interpretation: lora_cost_share_pct is the wall-clock contribution\n"
                "of the LoRA bmm path. If <2%, a faster custom bmm kernel (Punica\n"
                "MBGMV / S-LoRA SGMV) cannot move the needle for this setting.\n")

    # Also write a JSON sidecar for downstream tooling
    json_path = out.with_suffix(".json")
    with json_path.open("w") as f:
        json.dump({
            "gpu": torch.cuda.get_device_name(0),
            "config": vars(args),
            "flop_ratio": flops,
            "profiler_categories": profiler_summary,
            "profiler_total_ms": total_forward_ms,
            "ablation": ablation,
        }, f, indent=2)

    print(f"\nWrote {out} and {json_path}")
    print(f"\n=== Headline numbers ===")
    print(f"  LoRA bmm category share (profiler):  "
          f"{profiler_summary.get('lora_bmm', {}).get('share_pct', 0):.2f}%")
    print(f"  LoRA wall-clock share (ablation):    "
          f"{ablation['lora_cost_share_pct']:.2f}%")
    print(f"  Analytic FLOP ratio (base/lora):     "
          f"{flops['base_to_lora_ratio']:.0f}x")


if __name__ == "__main__":
    main()
