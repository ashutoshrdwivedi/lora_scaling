#!/bin/bash
# Rebuttal pod A1 -- A100-80GB -- DeBERTa-v2-xlarge generality row.
#
# Answers 5qtX ("a larger encoder would strengthen this") and yeZ9 Q4 ("beyond
# BERT/RoBERTa"): 900M params, d=1536, disentangled attention. Runs on the
# stock HF forward with hook-injected LoRA (--engine hf) because the repo's
# custom encoder cannot load disentangled attention; the hook path is
# PEFT-parity tested in tests/test_hf_wrapper.py.
#
# Measurement kernel is identical to the paper's run_sxm80.sh: fp16, seq=128,
# r=8, q+v projections, warmup 50 / iters 200, 5 seeds (PEFT arms at warmup 10
# / iters 50). Only the GRID is trimmed: an N-spine at the operating point
# B=32 plus a B-crossbar at N=1000, which is exactly what the reported columns
# (p50 drift as N->ceiling, throughput, speedup vs PEFT-mixed) consume. The
# full N x B cross product is already in Table 2 for bge-m3.
#
# Ceiling: 2 modules x 2 (A,B) x 24 layers x 1536 x r8 x 2B = 2.36 MB/adapter
# -> ~32k adapters on 80GB.
#
# PEFT caveat to disclose, not engineer around: deberta-v2-xlarge sets
# share_att_key=true, so PEFT's LoRA on query_proj also fires on the shared
# relative-position projection. Per-sample adapters cannot target a
# batch-independent tensor, so our hooks skip that call by construction. The
# extra PEFT work is <1% of its forward.
#
# Runtime ~1h45m. Independent of the other rebuttal scripts -- safe to run
# concurrently on a separate pod.
set -u
export HOME=/root
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
# Model downloads go to the mounted volume, not the container disk. The HF
# caches (deberta-v2-xlarge 3.4G, bge-m3 2.6G, electra 1.3G) plus the venv
# (~7G) overflow a default 20G container disk and abort a sweep mid-run.
export HF_HOME=${HF_HOME:-/workspace/hf_cache}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/workspace/uv_cache}
export TMPDIR=${TMPDIR:-/workspace/tmp}
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"
cd /root/lora_scaling
R=benchmarks/results
mkdir -p "$R"

M="microsoft/deberta-v2-xlarge"
TAG=deberta

echo "=== [0/5] smoke test (fail fast before burning the pod) ==="
uv run python -m lora_serving.benchmark.run \
  --model "$M" --engine hf --dtype fp16 --adapters 100 --batch-sizes 8 \
  --lora-ranks 8 --seq-len 128 --warmup 5 --iters 10 \
  --out "$R/smoke_$TAG.csv" > "$R/smoke_$TAG.log" 2>&1
rc=$?; echo "  latefuse smoke rc=$rc"
[ $rc -ne 0 ] && { echo "SMOKE FAILED -- see $R/smoke_$TAG.log"; tail -20 "$R/smoke_$TAG.log"; exit 1; }

uv run python -m benchmarks.baselines.peft_swap \
  --model "$M" --dtype fp16 --lora-ranks 8 --seq-len 128 --batch-sizes 8 \
  --adapters 2 --mode mixed --target-modules query_proj value_proj \
  --warmup 2 --iters 5 \
  --out "$R/smoke_peft_$TAG.csv" > "$R/smoke_peft_$TAG.log" 2>&1
rc=$?; echo "  peft smoke rc=$rc"
[ $rc -ne 0 ] && { echo "PEFT SMOKE FAILED -- see $R/smoke_peft_$TAG.log"; tail -20 "$R/smoke_peft_$TAG.log"; exit 1; }

echo "=== [1/5] model + env metadata ==="
uv run python -m benchmarks.profiling.model_metadata \
  > "$R/model_metadata_$TAG.log" 2>&1
echo "  metadata rc=$?"

# N-spine at B=32 (the operating point) + B-crossbar at N=1000 via
# --extra-configs, the same mechanism the paper uses to fold in rank cells so
# the shared (1000, 32) cell is measured exactly once per seed.
echo "=== [2/5] LateFuse sweep, 5 seeds (~42 min) ==="
uv run python -m lora_serving.benchmark.run \
  --model "$M" --engine hf --dtype fp16 \
  --adapters 100 1000 5000 10000 20000 30000 \
  --batch-sizes 32 --lora-ranks 8 \
  --extra-configs 1000:8:8 1000:128:8 \
  --seq-len 128 --warmup 50 --iters 200 \
  --seeds 1 2 3 4 5 \
  --out "$R/sweep_${TAG}_a100.csv" > "$R/sweep_${TAG}_a100.log" 2>&1
echo "  sweep rc=$?"

echo "=== [3/5] capacity probe (OOM ceiling) ==="
uv run python -m lora_serving.benchmark.run \
  --model "$M" --engine hf --dtype fp16 \
  --adapters 30000 32000 34000 --batch-sizes 32 --lora-ranks 8 \
  --seq-len 128 --warmup 50 --iters 200 \
  --out "$R/sweep_${TAG}_capacity.csv" > "$R/sweep_${TAG}_capacity.log" 2>&1
echo "  capacity rc=$?"

# Only the 'mixed' mode is cited in the rebuttal (speedup vs PEFT's native
# mixed-batch API) plus 'base' for the single-tenant ceiling. grouped /
# homogeneous / sequential are paper-only and skipped here.
echo "=== [4/5] PEFT mixed baseline, same node (~35 min) ==="
uv run python -m benchmarks.baselines.peft_swap \
  --model "$M" --dtype fp16 --lora-ranks 8 --seq-len 128 \
  --batch-sizes 8 32 128 --adapters 100 1000 --mode mixed \
  --target-modules query_proj value_proj --warmup 10 --iters 50 \
  --out "$R/peft_mixed_$TAG.csv" > "$R/peft_mixed_$TAG.log" 2>&1
echo "  peft mixed rc=$?"

echo "=== [5/5] PEFT base ceiling ==="
uv run python -m benchmarks.baselines.peft_swap \
  --model "$M" --dtype fp16 --lora-ranks 8 --seq-len 128 \
  --batch-sizes 8 32 128 --adapters 1 --mode base \
  --target-modules query_proj value_proj --warmup 10 --iters 50 \
  --out "$R/peft_base_$TAG.csv" > "$R/peft_base_$TAG.log" 2>&1
echo "  peft base rc=$?"

echo "ALL DONE ($TAG)"
