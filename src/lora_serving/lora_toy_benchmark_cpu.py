from __future__ import annotations

import csv
import random
import time

import numpy as np
import torch

from lora_serving.benchmark.synthetic import (
    make_synthetic_adapters,
    make_synthetic_inputs,
    make_synthetic_lr_weights,
)
from lora_serving.benchmark.run import BERT_COMPATIBLE_MODEL_TYPES, ENCODER_REGISTRY, print_table
from lora_serving.config import LoraServingConfig
from lora_serving.ops.head import LRHeadOps
from lora_serving.ops.lora import LoraOps
from lora_serving.weights.batch import BatchAssembler
from lora_serving.weights.store import AdapterStore


def cpu_shrink(self: LoraOps, x: torch.Tensor, a_weights: torch.Tensor) -> None:
    torch.bmm(x, a_weights, out=self._out_A)


def cpu_expand(self: LoraOps, b_weights: torch.Tensor) -> None:
    torch.bmm(self._out_A, b_weights, out=self._out_B)


LoraOps.shrink = cpu_shrink
LoraOps.expand = cpu_expand


def cpu_predict_proba(
    pooled: torch.Tensor,
    coef: torch.Tensor,
    intercept: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    torch.bmm(pooled, coef.transpose(1, 2), out=out)
    torch.add(out, intercept.unsqueeze(1), out=out)
    return out


LRHeadOps.predict_proba = staticmethod(cpu_predict_proba)


def run_single_config_cpu(
    encoder_family: str,
    model_name: str,
    num_adapters: int,
    batch_size: int,
    lora_rank: int,
    seq_len: int,
    warmup: int,
    iters: int,
    dtype: torch.dtype,
    num_labels: int = 10,
    seed: int | None = None,
) -> dict:
    device = torch.device("cpu")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    config = LoraServingConfig(
        model_name=model_name,
        lora_rank=lora_rank,
        batch_size=batch_size,
        max_seq_len=seq_len,
        target_modules=["query", "value"],
        device=device,
        dtype=dtype,
    )
    model_cls = ENCODER_REGISTRY[encoder_family]
    if encoder_family in {"bert", "sbert"} and config.model_type not in BERT_COMPATIBLE_MODEL_TYPES:
        raise ValueError(f"{model_name} model_type={config.model_type!r} is not compatible.")
    if encoder_family == "t5" and config.model_type != "t5":
        raise ValueError(f"{model_name} model_type={config.model_type!r}; expected T5.")

    print(f"  Loading {encoder_family} base model ({model_name}, {dtype})...")
    model = model_cls.from_pretrained_serving(config)
    model.eval()

    print(f"  Generating {num_adapters} synthetic adapters (rank={lora_rank})...")
    store = AdapterStore(config)
    t0 = time.perf_counter()
    make_synthetic_adapters(store, num_adapters)
    adapter_load_total_s = time.perf_counter() - t0
    adapter_ids = [f"adapter_{i}" for i in range(num_adapters)]
    print(f"  AdapterStore preload x{num_adapters} took {adapter_load_total_s:.2f}s", flush=True)

    lr_coefs = {}
    lr_intercepts = {}
    for aid in adapter_ids:
        coef, intercept = make_synthetic_lr_weights(config, num_labels)
        lr_coefs[aid] = coef
        lr_intercepts[aid] = intercept

    assembler = BatchAssembler(store, config)
    inputs = make_synthetic_inputs(config, batch_size)
    if encoder_family == "t5":
        inputs["input_ids"].remainder_(model.shared.num_embeddings)
    output_lr = torch.zeros(batch_size, 1, num_labels, dtype=dtype, device=device)

    def run_batch_timed() -> tuple[float, float]:
        batch_ids = random.choices(adapter_ids, k=batch_size)
        coefs = [lr_coefs[aid] for aid in batch_ids]
        intercepts = [lr_intercepts[aid] for aid in batch_ids]

        t0 = time.perf_counter()
        lora_w, lr_w = assembler.assemble(batch_ids, coefs, intercepts)
        assemble_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        with torch.no_grad():
            model(
                inputs["input_ids"],
                inputs["attention_mask"],
                lora_w,
                lr_w,
                output_lr,
            )
        forward_ms = (time.perf_counter() - t0) * 1000
        return assemble_ms, forward_ms

    print(f"  Warming up ({warmup} iters)...")
    for _ in range(warmup):
        run_batch_timed()

    print(f"  Measuring ({iters} iters)...")
    assemble_latencies = np.empty(iters)
    forward_latencies = np.empty(iters)
    for i in range(iters):
        assemble_latencies[i], forward_latencies[i] = run_batch_timed()

    total_latencies = assemble_latencies + forward_latencies
    throughput = batch_size / (np.mean(total_latencies) / 1000)

    return {
        "model": model_name,
        "encoder_family": encoder_family,
        "dtype": str(dtype).replace("torch.", ""),
        "num_adapters": num_adapters,
        "batch_size": batch_size,
        "lora_rank": lora_rank,
        "seq_len": seq_len,
        "p50_ms": round(float(np.percentile(total_latencies, 50)), 2),
        "p90_ms": round(float(np.percentile(total_latencies, 90)), 2),
        "p95_ms": round(float(np.percentile(total_latencies, 95)), 2),
        "p99_ms": round(float(np.percentile(total_latencies, 99)), 2),
        "mean_ms": round(float(np.mean(total_latencies)), 2),
        "assemble_p50_ms": round(float(np.percentile(assemble_latencies, 50)), 3),
        "assemble_mean_ms": round(float(np.mean(assemble_latencies)), 3),
        "forward_p50_ms": round(float(np.percentile(forward_latencies, 50)), 3),
        "forward_mean_ms": round(float(np.mean(forward_latencies)), 3),
        "assemble_share_pct": round(100 * np.mean(assemble_latencies) / np.mean(total_latencies), 1),
        "throughput_samples_sec": round(throughput, 1),
        "peak_gpu_mem_gb": 0.0,
        "adapter_cache_gb": round(store.memory_gb(), 3),
        "adapter_load_total_s": round(adapter_load_total_s, 2),
        "warmup": warmup,
        "iters": iters,
        "seed": seed if seed is not None else "",
    }


jobs = [
    ("bert", "bert-base-uncased", 1, 1, 2),
    ("bert", "bert-base-uncased", 2, 1, 2),
    ("sbert", "bert-base-uncased", 1, 1, 2),
    ("sbert", "bert-base-uncased", 2, 1, 2),
    ("t5", "hf-internal-testing/tiny-random-t5", 1, 1, 2),
]

rows = []
for family, model, adapters, batch, rank in jobs:
    print(f"[toy] family={family} model={model} adapters={adapters} batch={batch} rank={rank}")
    row = run_single_config_cpu(
        encoder_family=family,
        model_name=model,
        num_adapters=adapters,
        batch_size=batch,
        lora_rank=rank,
        seq_len=8,
        warmup=1,
        iters=2,
        dtype=torch.float32,
        num_labels=2,
        seed=123,
    )
    rows.append(row)
    print(
        f"  p50={row['p50_ms']}ms p95={row['p95_ms']}ms p99={row['p99_ms']}ms "
        f"asm={row['assemble_mean_ms']}ms ({row['assemble_share_pct']}%) "
        f"fwd={row['forward_mean_ms']}ms tput={row['throughput_samples_sec']} samples/s\n"
    )

print("\n=== Toy Results ===")
print_table(rows)

out = "/tmp/lora_cpu_new_run_toy.csv"
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"\nSaved to {out}")
