#!/bin/bash
# Rebuttal pod A3 (short) -- A100-80GB -- result-scatter instrumentation.
#
# Fills the [X]/[Y] placeholders in the consolidated CPU-path answer (yeZ9 Q1 +
# Q3 + reject-#5): the per-tenant slice + softmax cost, to show it is negligible
# next to batch assembly. Must run on an A100-80GB so the numbers compose with
# the already-measured assembly_bench results rather than mixing hardware.
#
# Measures on-device scatter only, matching the forward's boundary -- host D2H
# and serialization are serving-layer and stay out of scope (a caveat the
# rebuttal keeps explicitly).
#
# New output filenames: does NOT clobber the submitted assembly_bench.txt /
# assembly_bench_minilm.txt results.
#
# Runtime ~20 min. Independent of the other rebuttal scripts -- pairs well
# appended to run_rebuttal_electra.sh on the same pod.
set -u
export HOME=/root
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
# Model downloads go to the mounted volume, not the container disk (see
# run_rebuttal_deberta.sh: a default 20G container disk fills mid-sweep).
export HF_HOME=${HF_HOME:-/workspace/hf_cache}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/root/.cache/uv}
export TMPDIR=${TMPDIR:-/workspace/tmp}
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"
cd /root/lora_scaling
R=benchmarks/results/rebuttal_scatter
mkdir -p "$R"

echo "=== [1/2] assembly + scatter, bge-m3 (paper's model) ==="
uv run python -m benchmarks.profiling.assembly_bench \
  --model "BAAI/bge-m3" --dtype fp16 --num-adapters 2000 \
  --batch-sizes 8 16 32 64 128 --lora-rank 8 --seq-len 128 \
  --warmup 30 --iters 200 --seeds 1 2 3 4 5 \
  --out "$R/assembly_bench_scatter.txt" > "$R/assembly_bench_scatter.log" 2>&1
echo "  bge-m3 rc=$?"

# MiniLM is the >10k req/s CPU-regime demo: a fast, shallow model makes the
# CPU assembly share (and hence the scatter share) maximally visible.
echo "=== [2/2] assembly + scatter, all-MiniLM-L6-v2 (CPU regime) ==="
uv run python -m benchmarks.profiling.assembly_bench \
  --model "sentence-transformers/all-MiniLM-L6-v2" --dtype fp16 --num-adapters 2000 \
  --batch-sizes 8 16 32 64 128 256 --lora-rank 8 --seq-len 128 \
  --warmup 30 --iters 200 --seeds 1 2 3 4 5 \
  --out "$R/assembly_bench_minilm_scatter.txt" > "$R/assembly_bench_minilm_scatter.log" 2>&1
echo "  minilm rc=$?"

echo "ALL DONE (scatter)"
