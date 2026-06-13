#!/bin/bash
# Driver: regenerate the full paper dataset on ONE A100-SXM4-80GB machine so every
# number comes from the same hardware. Order = cheapest+essential first so an early
# stop (e.g. balance) still leaves the critical pieces on disk.
set -u
export HOME=/root
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
cd /root/lora_scaling
R=benchmarks/results
mkdir -p "$R"
M="BAAI/bge-m3"

echo "=== [0/10] model + env metadata ==="
uv run python -m benchmarks.profiling.model_metadata \
  > "$R/model_metadata.log" 2>&1
echo "  metadata rc=$?"

# (Removed: a separate single-run LateFuse reference. Table 3 and the
# throughput figure now read LateFuse from sweep_main below, filtered to r=8,
# so they share one run with Table 2 and cannot drift.)

# The two sweeps that back the paper's statistical claims (Finding 1: p50 vs
# adapter count; Finding 2: throughput vs batch; Finding 3: p50/p99 vs rank)
# run 5 seeded repeats each (--seeds), all written to one CSV with a 'seed'
# column; paper/build_numbers.py aggregates to mean +/- s.d. per config.
# The PEFT baselines below stay single-run: grouped reaches ~80 s/iter at the
# largest configuration, and the speedup claim is order-of-magnitude, so it
# does not need variance estimates. Multi-seed wall-clock: sweep_main ~2 h,
# sweep_ranks ~50 min (5x the 2026-06-08 single-run times).

echo "=== [2/10] headline LateFuse sweep (Fig 2 + scaling table), 5 seeds ==="
# Includes N=40000 and 47000 at every batch size (overlapping with sweep_capacity
# at B=32) so paper/build_numbers.py finds the high-N rows it needs in
# sweep_main without cross-file merging. Extra wall-clock is ~2 min.
uv run python -m lora_serving.benchmark.run \
  --model "$M" --dtype fp16 --adapters 100 1000 5000 10000 20000 40000 47000 \
  --batch-sizes 8 16 32 64 128 --lora-ranks 8 --seq-len 128 --warmup 50 --iters 200 \
  --seeds 1 2 3 4 5 \
  --out "$R/sweep_main.csv" > "$R/sweep_main.log" 2>&1
echo "  sweep_main rc=$?"

echo "=== [3/10] LateFuse rank sweep, 5 seeds ==="
uv run python -m lora_serving.benchmark.run \
  --model "$M" --dtype fp16 --adapters 100 1000 10000 --batch-sizes 32 \
  --lora-ranks 4 8 16 32 --seq-len 128 --warmup 50 --iters 200 \
  --seeds 1 2 3 4 5 \
  --out "$R/sweep_ranks.csv" > "$R/sweep_ranks.log" 2>&1
echo "  sweep_ranks rc=$?"

echo "=== [4/10] capacity probe (OOM ceiling on 80GB) ==="
uv run python -m lora_serving.benchmark.run \
  --model "$M" --dtype fp16 --adapters 20000 40000 47000 50000 --batch-sizes 32 \
  --lora-ranks 8 --seq-len 128 --warmup 50 --iters 200 \
  --out "$R/sweep_capacity.csv" > "$R/sweep_capacity.log" 2>&1
echo "  sweep_capacity rc=$?"

# PEFT runs use --warmup 10 --iters 50 across base/grouped/homogeneous/mixed
# so the protocol is symmetric across the four modes; grouped at B=128 N=1000
# hits ~80s/iter, so 200 iters there would dominate wall-clock without adding
# meaningful precision for an order-of-magnitude speedup claim.

echo "=== [5/10] PEFT base ceiling ==="
uv run python -m benchmarks.baselines.peft_swap \
  --model "$M" --dtype fp16 --lora-ranks 8 --seq-len 128 --batch-sizes 8 32 128 \
  --adapters 1 --mode base --warmup 10 --iters 50 \
  --out "$R/peft_base_sxm80.csv" > "$R/peft_base_sxm80.log" 2>&1
echo "  base rc=$?"

echo "=== [6/10] PEFT grouped ==="
uv run python -m benchmarks.baselines.peft_swap \
  --model "$M" --dtype fp16 --lora-ranks 8 --seq-len 128 --batch-sizes 8 32 128 \
  --adapters 100 1000 --mode grouped --warmup 10 --iters 50 \
  --out "$R/peft_grouped_sxm80.csv" > "$R/peft_grouped_sxm80.log" 2>&1
echo "  grouped rc=$?"

echo "=== [7/10] PEFT homogeneous (single-tenant batch) ==="
uv run python -m benchmarks.baselines.peft_swap \
  --model "$M" --dtype fp16 --lora-ranks 8 --seq-len 128 --batch-sizes 8 32 128 \
  --adapters 100 1000 --mode homogeneous --warmup 10 --iters 50 \
  --out "$R/peft_homogeneous_sxm80.csv" > "$R/peft_homogeneous_sxm80.log" 2>&1
echo "  homogeneous rc=$?"

echo "=== [8/10] PEFT mixed (native adapter_names mixed-batch API) ==="
# Bare encoder forward only: PEFT's mixed-batch path cannot route per-tenant
# classification heads (huggingface/peft#1960), so like the other PEFT modes
# this times the encoder without a head -- conservative in PEFT's favor.
uv run python -m benchmarks.baselines.peft_swap \
  --model "$M" --dtype fp16 --lora-ranks 8 --seq-len 128 --batch-sizes 8 32 128 \
  --adapters 100 1000 --mode mixed --warmup 10 --iters 50 \
  --out "$R/peft_mixed_sxm80.csv" > "$R/peft_mixed_sxm80.log" 2>&1
echo "  mixed rc=$?"

echo "=== [9/10] forward breakdown ==="
uv run python -m benchmarks.profiling.forward_breakdown \
  --model "$M" --dtype fp16 --batch-size 32 --num-adapters 1000 --lora-rank 8 \
  --seq-len 128 --out "$R/forward_breakdown.txt" > "$R/forward_breakdown.log" 2>&1
echo "  forward_breakdown rc=$?"

echo "ALL DONE"
