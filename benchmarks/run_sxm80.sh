#!/bin/bash
# Driver: regenerate all Table-4 baselines on ONE A100-SXM4-80GB machine for a
# self-consistent comparison. Order = cheapest+essential first so an early stop
# (e.g. balance) still leaves the critical pieces on disk.
set -u
export HOME=/root
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
cd /root/lora_scaling
R=benchmarks/results
mkdir -p "$R"
M="BAAI/bge-m3"

echo "=== [1/4] sysname reference ==="
uv run python -m lora_serving.benchmark.run \
  --model "$M" --dtype fp16 --adapters 100 1000 --batch-sizes 8 32 128 \
  --lora-ranks 8 --seq-len 128 --warmup 50 --iters 200 \
  --out "$R/sysname_sxm80_ref.csv" > "$R/sysname_sxm80_ref.log" 2>&1
echo "  sysname rc=$?"

echo "=== [2/4] PEFT base ceiling ==="
uv run python -m benchmarks.baselines.peft_swap \
  --model "$M" --dtype fp16 --lora-ranks 8 --seq-len 128 --batch-sizes 8 32 128 \
  --adapters 1 --mode base --warmup 20 --iters 100 \
  --out "$R/peft_base_sxm80.csv" > "$R/peft_base_sxm80.log" 2>&1
echo "  base rc=$?"

echo "=== [3/4] PEFT grouped ==="
uv run python -m benchmarks.baselines.peft_swap \
  --model "$M" --dtype fp16 --lora-ranks 8 --seq-len 128 --batch-sizes 8 32 128 \
  --adapters 100 1000 --mode grouped --warmup 2 --iters 6 \
  --out "$R/peft_grouped_sxm80.csv" > "$R/peft_grouped_sxm80.log" 2>&1
echo "  grouped rc=$?"

echo "=== [4/4] PEFT sequential ==="
uv run python -m benchmarks.baselines.peft_swap \
  --model "$M" --dtype fp16 --lora-ranks 8 --seq-len 128 --batch-sizes 8 32 128 \
  --adapters 100 1000 --mode sequential --warmup 2 --iters 6 \
  --out "$R/peft_seq_sxm80.csv" > "$R/peft_seq_sxm80.log" 2>&1
echo "  sequential rc=$?"

echo "ALL DONE"
