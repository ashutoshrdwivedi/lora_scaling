#!/bin/bash
set -u
export HOME=/workspace
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
cd /workspace/lora_scaling

R=benchmarks/results
M="BAAI/bge-m3"
mkdir -p "$R"


echo "=== [1/2] churn sweep: Zipf alpha 1.0 ==="
uv run python -m lora_serving.benchmark.run \
  --engine hf --model "BAAI/bge-m3" --dtype fp16 \
  --adapters 1000 \
  --batch-sizes 32 --lora-ranks 4 8 16 32\
  --extra-configs 1000:16:8 1000:64:8 1000:128:8 1000:8:8 \
  --seq-len 128 --warmup 50 --iters 1000 --seeds 1 2\
  --churn --churn-num-tenants 1500 --churn-zipf-alpha 1.1 \
  --out "benchmarks/results/churn_alpha_1p1_1k_1p5k.csv" > "benchmarks/results/churn_alpha_1p1_1k_1p5k.log" 2>&1

echo "=== [1/4] churn sweep: Zipf alpha 1.1 ==="
uv run python -m lora_serving.benchmark.run \
  --engine hf --model "BAAI/bge-m3" --dtype fp16 \
  --adapters 100 \
  --batch-sizes 32 --lora-ranks 8\
  --seq-len 128 --warmup 50 --iters 1000 --seeds 1 2\
  --churn --churn-num-tenants 150 --churn-zipf-alpha 1.1 \
  --out "benchmarks/results/churn_alpha_1p1_0p1k_150.csv" > "benchmarks/results/churn_alpha_1p1_0p1k_150.log" 2>&1

uv run python -m lora_serving.benchmark.run \
  --engine hf --model "BAAI/bge-m3" --dtype fp16 \
  --adapters 5000 \
  --batch-sizes 32 --lora-ranks 8\
  --seq-len 128 --warmup 50 --iters 1000 --seeds 1 2\
  --churn --churn-num-tenants 5500 --churn-zipf-alpha 1.1 \
  --out "benchmarks/results/churn_alpha_1p1_5k_5p5k.csv" > "benchmarks/results/churn_alpha_1p1_5k_5p5k.log" 2>&1

uv run python -m lora_serving.benchmark.run \
  --engine hf --model "BAAI/bge-m3" --dtype fp16 \
  --adapters 20000 \
  --batch-sizes 32 --lora-ranks 8\
  --seq-len 128 --warmup 50 --iters 1000 --seeds 1 2\
  --churn --churn-num-tenants 22000 --churn-zipf-alpha 1.1 \
  --out "benchmarks/results/churn_alpha_1p1_20k_22k.csv" > "benchmarks/results/churn_alpha_1p1_20k_22k.log" 2>&1

uv run python -m lora_serving.benchmark.run \
  --engine hf --model "BAAI/bge-m3" --dtype fp16 \
  --adapters 40000 \
  --batch-sizes 32 --lora-ranks 8\
  --seq-len 128 --warmup 50 --iters 1000 --seeds 1 2\
  --churn --churn-num-tenants 42500 --churn-zipf-alpha 1.1 \
  --out "benchmarks/results/churn_alpha_1p1_40k_42p5k.csv" > "benchmarks/results/churn_alpha_1p1_40k_42p5k.log" 2>&1

uv run python -m lora_serving.benchmark.run \
  --engine hf --model "BAAI/bge-m3" --dtype fp16 \
  --adapters 47000 \
  --batch-sizes 32 --lora-ranks 8\
  --seq-len 128 --warmup 50 --iters 1000 --seeds 1 2\
  --churn --churn-num-tenants 50000 --churn-zipf-alpha 1.1 \
  --out "benchmarks/results/churn_alpha_1p1_47k_50k.csv" > "benchmarks/results/churn_alpha_1p1_47k_50k.log" 2>&1


echo "ALL DONE"