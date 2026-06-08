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

echo "=== [0/9] model + env metadata ==="
uv run python -m benchmarks.profiling.model_metadata \
  > "$R/model_metadata.log" 2>&1
echo "  metadata rc=$?"

echo "=== [1/9] sysname Table-4 reference ==="
uv run python -m lora_serving.benchmark.run \
  --model "$M" --dtype fp16 --adapters 100 1000 --batch-sizes 8 32 128 \
  --lora-ranks 8 --seq-len 128 --warmup 50 --iters 200 \
  --out "$R/sysname_sxm80_ref.csv" > "$R/sysname_sxm80_ref.log" 2>&1
echo "  sysname ref rc=$?"

echo "=== [2/9] headline LateFuse sweep (Fig 2 + scaling table) ==="
# Includes N=40000 and 47000 at every batch size (overlapping with sweep_capacity
# at B=32) so paper/build_numbers.py finds the high-N rows it needs in
# sweep_main without cross-file merging. Extra wall-clock is ~2 min.
uv run python -m lora_serving.benchmark.run \
  --model "$M" --dtype fp16 --adapters 100 1000 5000 10000 20000 40000 47000 \
  --batch-sizes 8 16 32 64 128 --lora-ranks 8 --seq-len 128 --warmup 50 --iters 200 \
  --out "$R/sweep_main.csv" > "$R/sweep_main.log" 2>&1
echo "  sweep_main rc=$?"

echo "=== [3/9] LateFuse rank sweep ==="
uv run python -m lora_serving.benchmark.run \
  --model "$M" --dtype fp16 --adapters 100 1000 10000 --batch-sizes 32 \
  --lora-ranks 4 8 16 32 --seq-len 128 --warmup 50 --iters 200 \
  --out "$R/sweep_ranks.csv" > "$R/sweep_ranks.log" 2>&1
echo "  sweep_ranks rc=$?"

echo "=== [4/9] capacity probe (OOM ceiling on 80GB) ==="
uv run python -m lora_serving.benchmark.run \
  --model "$M" --dtype fp16 --adapters 20000 40000 47000 50000 --batch-sizes 32 \
  --lora-ranks 8 --seq-len 128 --warmup 50 --iters 200 \
  --out "$R/sweep_capacity.csv" > "$R/sweep_capacity.log" 2>&1
echo "  sweep_capacity rc=$?"

# PEFT runs use --warmup 10 --iters 50 across base/grouped/homogeneous so the
# protocol is symmetric across the three modes; grouped at B=128 N=1000 hits
# ~80s/iter, so 200 iters there would dominate wall-clock without adding
# meaningful precision for an order-of-magnitude speedup claim.

echo "=== [5/9] PEFT base ceiling ==="
uv run python -m benchmarks.baselines.peft_swap \
  --model "$M" --dtype fp16 --lora-ranks 8 --seq-len 128 --batch-sizes 8 32 128 \
  --adapters 1 --mode base --warmup 10 --iters 50 \
  --out "$R/peft_base_sxm80.csv" > "$R/peft_base_sxm80.log" 2>&1
echo "  base rc=$?"

echo "=== [6/9] PEFT grouped ==="
uv run python -m benchmarks.baselines.peft_swap \
  --model "$M" --dtype fp16 --lora-ranks 8 --seq-len 128 --batch-sizes 8 32 128 \
  --adapters 100 1000 --mode grouped --warmup 10 --iters 50 \
  --out "$R/peft_grouped_sxm80.csv" > "$R/peft_grouped_sxm80.log" 2>&1
echo "  grouped rc=$?"

echo "=== [7/9] PEFT homogeneous (single-tenant batch) ==="
uv run python -m benchmarks.baselines.peft_swap \
  --model "$M" --dtype fp16 --lora-ranks 8 --seq-len 128 --batch-sizes 8 32 128 \
  --adapters 100 1000 --mode homogeneous --warmup 10 --iters 50 \
  --out "$R/peft_homogeneous_sxm80.csv" > "$R/peft_homogeneous_sxm80.log" 2>&1
echo "  homogeneous rc=$?"

echo "=== [8/9] forward breakdown ==="
uv run python -m benchmarks.profiling.forward_breakdown \
  --model "$M" --dtype fp16 --batch-size 32 --num-adapters 1000 --lora-rank 8 \
  --seq-len 128 --out "$R/forward_breakdown.txt" > "$R/forward_breakdown.log" 2>&1
echo "  forward_breakdown rc=$?"

echo "ALL DONE"
