"""PEFT per-request swap baseline.

Measures latency / throughput for a mixed-tenant batch served via HuggingFace PEFT.
This is the worst-case-but-realistic baseline that our multi-tenant impl is compared against.

Two modes:
  - sequential: each sample uses set_adapter() + forward(B=1). Worst case (no batching).
  - grouped:    bucket batch by adapter_id, one batched forward per bucket. Best case PEFT.

Output CSV is schema-compatible with `lora_serving.benchmark.run` so the two can be
merged for plotting. Extra columns: `harness=peft_<mode>`.

Run on TIR:
    uv run python -m benchmarks.baselines.peft_swap \\
        --adapters 100 1000 5000 --batch-sizes 8 32 128 --mode sequential
"""

from __future__ import annotations

import argparse
import csv
import gc
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModel

DTYPE_MAP = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}


def build_peft_model(model_name: str, lora_rank: int, dtype: torch.dtype, device: torch.device):
    """Load base + wrap with PEFT, returning model with adapter named 'adapter_0'."""
    base = AutoModel.from_pretrained(model_name, torch_dtype=dtype)
    cfg = LoraConfig(
        r=lora_rank, lora_alpha=lora_rank,
        target_modules=["query", "value"],
        lora_dropout=0.0, bias="none",
    )
    model = get_peft_model(base, cfg, adapter_name="adapter_0")
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, cfg


def add_n_adapters(model, lora_cfg: LoraConfig, n: int, device: torch.device, dtype: torch.dtype):
    """Add adapter_1..adapter_{n-1} to the model (adapter_0 already exists)."""
    for i in range(1, n):
        model.add_adapter(f"adapter_{i}", lora_cfg)
    model.to(device=device, dtype=dtype)


def make_inputs(batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(1, 30000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    return input_ids, attention_mask


def run_sequential(model, adapter_ids: list[str], inputs: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> float:
    """One forward per sample with set_adapter() between each. Returns total ms (event-timed)."""
    input_ids, attention_mask = inputs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        for i, aid in enumerate(adapter_ids):
            model.set_adapter(aid)
            model(input_ids[i:i+1], attention_mask=attention_mask[i:i+1])
    end.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(end)


def run_grouped(model, adapter_ids: list[str], inputs: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> float:
    """Bucket samples by adapter_id, one batched forward per bucket. Returns total ms."""
    input_ids, attention_mask = inputs
    buckets: dict[str, list[int]] = {}
    for i, aid in enumerate(adapter_ids):
        buckets.setdefault(aid, []).append(i)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        for aid, idxs in buckets.items():
            model.set_adapter(aid)
            idx_t = torch.tensor(idxs, device=device)
            model(input_ids[idx_t], attention_mask=attention_mask[idx_t])
    end.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(end)


def run_single_config(
    model_name: str,
    num_adapters: int,
    batch_size: int,
    lora_rank: int,
    seq_len: int,
    warmup: int,
    iters: int,
    dtype: torch.dtype,
    mode: str,
) -> dict:
    device = torch.device("cuda:0")
    print(f"  Building PEFT model + adding {num_adapters} adapters...")
    model, lora_cfg = build_peft_model(model_name, lora_rank, dtype, device)
    t0 = time.perf_counter()
    add_n_adapters(model, lora_cfg, num_adapters, device, dtype)
    add_time_s = time.perf_counter() - t0
    print(f"  add_adapter() x{num_adapters - 1} took {add_time_s:.1f}s")

    adapter_ids = [f"adapter_{i}" for i in range(num_adapters)]
    inputs = make_inputs(batch_size, seq_len, device)
    runner = run_sequential if mode == "sequential" else run_grouped

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    print(f"  Warming up ({warmup} iters)...")
    for _ in range(warmup):
        batch_ids = random.choices(adapter_ids, k=batch_size)
        runner(model, batch_ids, inputs, device)

    print(f"  Measuring ({iters} iters)...")
    latencies = np.empty(iters)
    for i in range(iters):
        batch_ids = random.choices(adapter_ids, k=batch_size)
        latencies[i] = runner(model, batch_ids, inputs, device)

    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
    throughput = batch_size / (np.mean(latencies) / 1000)

    return {
        "harness": f"peft_{mode}",
        "model": model_name,
        "dtype": str(dtype).replace("torch.", ""),
        "num_adapters": num_adapters,
        "batch_size": batch_size,
        "lora_rank": lora_rank,
        "seq_len": seq_len,
        "p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "p90_ms": round(float(np.percentile(latencies, 90)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies, 99)), 2),
        "mean_ms": round(float(np.mean(latencies)), 2),
        "throughput_samples_sec": round(throughput, 1),
        "peak_gpu_mem_gb": round(peak_mem_gb, 3),
        "add_adapter_total_s": round(add_time_s, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--dtype", choices=list(DTYPE_MAP.keys()), default="fp16")
    parser.add_argument("--adapters", nargs="+", type=int, default=[100, 1000])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 32, 128])
    parser.add_argument("--lora-ranks", nargs="+", type=int, default=[8])
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--mode", choices=["sequential", "grouped"], default="sequential")
    parser.add_argument("--out", default="peft_baseline.csv")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA. Aborting.")
        return

    dtype = DTYPE_MAP[args.dtype]
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Mode: {args.mode}\n")

    total = len(args.adapters) * len(args.batch_sizes) * len(args.lora_ranks)
    done = 0
    results = []
    for num_adapters in args.adapters:
        for batch_size in args.batch_sizes:
            for lora_rank in args.lora_ranks:
                done += 1
                print(f"[{done}/{total}] adapters={num_adapters} batch={batch_size} rank={lora_rank}")
                try:
                    row = run_single_config(
                        model_name=args.model,
                        num_adapters=num_adapters,
                        batch_size=batch_size,
                        lora_rank=lora_rank,
                        seq_len=args.seq_len,
                        warmup=args.warmup,
                        iters=args.iters,
                        dtype=dtype,
                        mode=args.mode,
                    )
                    results.append(row)
                    print(f"  p50={row['p50_ms']}ms  p99={row['p99_ms']}ms  "
                          f"tput={row['throughput_samples_sec']} samples/s  "
                          f"peak={row['peak_gpu_mem_gb']}GB\n")
                except torch.cuda.OutOfMemoryError as e:
                    print(f"  OOM: {e}")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                except Exception as e:
                    print(f"  ERROR ({type(e).__name__}): {e}")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                finally:
                    torch.cuda.empty_cache()
                    gc.collect()

    if not results:
        print("No results collected.")
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
