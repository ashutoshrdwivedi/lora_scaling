#!/bin/bash
# Rebuttal pod B -- L40S-48GB -- hardware-generality row (OPTIONAL).
#
# Runs the paper's reference model (bge-m3) on an inference-class GPU -- Ada,
# GDDR6, ~864 GB/s vs the A100's HBM2e ~2 TB/s -- to show the O(1)-in-N property
# and the PEFT speedups are not artifacts of the A100.
#
# SCOPE CAVEAT (keep this straight in the rebuttal): a second GPU *class* is
# hardware generality. It does NOT answer yeZ9 reject-#2, which is about
# multi-GPU / multi-node scaling; that stays future work (Appendix C).
# No reviewer asked for this row -- it is the first thing to cut if the
# deadline tightens.
#
# Ceiling scales with VRAM exactly as Finding 4 predicts: 1.57 MB/adapter on
# ~45 GB usable -> ~28k adapters, vs 47k on the 80GB A100.
#
# Measurement kernel identical to run_sxm80.sh; grid is the N-spine at B=32
# plus a B-crossbar at N=1000 (see run_rebuttal_deberta.sh for the rationale).
# The PEFT arm runs on this same node so the speedup column is never a
# cross-node comparison.
#
# Runtime ~1h30m. Independent of the other rebuttal scripts.
set -u
export HOME=/root
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
# Model downloads go to the mounted volume, not the container disk (see
# run_rebuttal_deberta.sh: a default 20G container disk fills mid-sweep).
export HF_HOME=${HF_HOME:-/workspace/hf_cache}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/workspace/uv_cache}
export TMPDIR=${TMPDIR:-/workspace/tmp}
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"
cd /root/lora_scaling
R=benchmarks/results
mkdir -p "$R"

M="BAAI/bge-m3"
TAG=l40s

echo "=== [0/5] smoke test (fail fast before burning the pod) ==="
uv run python -m lora_serving.benchmark.run \
  --model "$M" --dtype fp16 --adapters 100 --batch-sizes 8 \
  --lora-ranks 8 --seq-len 128 --warmup 5 --iters 10 \
  --out "$R/smoke_$TAG.csv" > "$R/smoke_$TAG.log" 2>&1
rc=$?; echo "  latefuse smoke rc=$rc"
[ $rc -ne 0 ] && { echo "SMOKE FAILED -- see $R/smoke_$TAG.log"; tail -20 "$R/smoke_$TAG.log"; exit 1; }

echo "=== [1/5] model + env metadata ==="
uv run python -m benchmarks.profiling.model_metadata \
  > "$R/model_metadata_$TAG.log" 2>&1
echo "  metadata rc=$?"

echo "=== [2/5] LateFuse sweep, 5 seeds (~35 min) ==="
uv run python -m lora_serving.benchmark.run \
  --model "$M" --dtype fp16 \
  --adapters 100 1000 5000 10000 20000 28000 \
  --batch-sizes 32 --lora-ranks 8 \
  --extra-configs 1000:8:8 1000:128:8 \
  --seq-len 128 --warmup 50 --iters 200 \
  --seeds 1 2 3 4 5 \
  --out "$R/sweep_bgem3_$TAG.csv" > "$R/sweep_bgem3_$TAG.log" 2>&1
echo "  sweep rc=$?"

echo "=== [3/5] capacity probe (OOM ceiling on 48GB) ==="
uv run python -m lora_serving.benchmark.run \
  --model "$M" --dtype fp16 \
  --adapters 26000 28000 30000 32000 --batch-sizes 32 --lora-ranks 8 \
  --seq-len 128 --warmup 50 --iters 200 \
  --out "$R/sweep_capacity_$TAG.csv" > "$R/sweep_capacity_$TAG.log" 2>&1
echo "  capacity rc=$?"

echo "=== [4/5] PEFT mixed baseline, same node (~40 min) ==="
uv run python -m benchmarks.baselines.peft_swap \
  --model "$M" --dtype fp16 --lora-ranks 8 --seq-len 128 \
  --batch-sizes 8 32 128 --adapters 100 1000 --mode mixed \
  --warmup 10 --iters 50 \
  --out "$R/peft_mixed_$TAG.csv" > "$R/peft_mixed_$TAG.log" 2>&1
echo "  peft mixed rc=$?"

echo "=== [5/5] PEFT base ceiling ==="
uv run python -m benchmarks.baselines.peft_swap \
  --model "$M" --dtype fp16 --lora-ranks 8 --seq-len 128 \
  --batch-sizes 8 32 128 --adapters 1 --mode base \
  --warmup 10 --iters 50 \
  --out "$R/peft_base_$TAG.csv" > "$R/peft_base_$TAG.log" 2>&1
echo "  peft base rc=$?"

echo "ALL DONE ($TAG)"
