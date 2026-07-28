#!/bin/bash
# Rebuttal pod A2 -- A100-80GB -- ELECTRA-large generality row.
#
# Answers yeZ9 Q4 ("beyond BERT/RoBERTa") with a model the reviewer named.
# ELECTRA is a replaced-token-detection discriminator, a different pretraining
# family from bge-m3's masked-LM XLM-RoBERTa, with WordPiece 30k vs
# SentencePiece 250k vocabulary and absolute (arange) position ids rather than
# RoBERTa's mask-offset scheme -- so it exercises the other branch of
# EncoderWithLora.position_ids.
#
# Crucially this row runs on the PAPER'S OWN engine (the custom encoder), not
# the HF hook wrapper, so it is direct evidence about the shipped system.
# ELECTRA checkpoints have no pooler; from_pretrained_serving tolerates exactly
# that gap and stays strict about everything else.
#
# Same L=24, d=1024 adapter geometry as bge-m3 -> identical 1.57 MB/adapter and
# the same ~47k ceiling, which is itself a clean confirmation that Finding 4's
# ceiling is set by adapter geometry (L, d, r), not by model size or vocabulary.
#
# Grid and measurement kernel both mirror run_sxm80.sh: the full N x B cross
# product plus the same three rank cells at the (1000, 32) operating point, so
# every Table 2 cell has an ELECTRA counterpart and the rebuttal can quote any
# of them without a second pod session.
# 6 N x 5 B + 3 rank cells = 33 configs x 5 seeds = 165 runs.
#
# Runtime ~2h30m. Independent of the other rebuttal scripts. If you have a
# spare slot on this pod, run_rebuttal_scatter.sh (~20 min) pairs well here.
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
R=benchmarks/results/rebuttal_electra
mkdir -p "$R"

M="google/electra-large-discriminator"
TAG=electra

echo "=== [0/5] smoke test (fail fast before burning the pod) ==="
uv run python -m lora_serving.benchmark.run \
  --model "$M" --dtype fp16 --adapters 100 --batch-sizes 8 \
  --lora-ranks 8 --seq-len 128 --warmup 5 --iters 10 \
  --out "$R/smoke_$TAG.csv" > "$R/smoke_$TAG.log" 2>&1
rc=$?; echo "  latefuse smoke rc=$rc"
[ $rc -ne 0 ] && { echo "SMOKE FAILED -- see $R/smoke_$TAG.log"; tail -20 "$R/smoke_$TAG.log"; exit 1; }

uv run python -m benchmarks.baselines.peft_swap \
  --model "$M" --dtype fp16 --lora-ranks 8 --seq-len 128 --batch-sizes 8 \
  --adapters 2 --mode mixed --warmup 2 --iters 5 \
  --out "$R/smoke_peft_$TAG.csv" > "$R/smoke_peft_$TAG.log" 2>&1
rc=$?; echo "  peft smoke rc=$rc"
[ $rc -ne 0 ] && { echo "PEFT SMOKE FAILED -- see $R/smoke_peft_$TAG.log"; tail -20 "$R/smoke_peft_$TAG.log"; exit 1; }

echo "=== [1/5] model + env metadata ==="
uv run python -m benchmarks.profiling.model_metadata \
  --model "google/electra-large-discriminator:query,value" --out-dir "$R" \
  > "$R/model_metadata_$TAG.log" 2>&1
echo "  metadata rc=$?"

echo "=== [2/5] LateFuse sweep, full grid, 5 seeds (~1h50m) ==="
uv run python -m lora_serving.benchmark.run \
  --model "$M" --dtype fp16 \
  --adapters 100 1000 5000 10000 20000 47000 \
  --batch-sizes 8 16 32 64 128 --lora-ranks 8 \
  --extra-configs 1000:32:4 1000:32:16 1000:32:32 \
  --seq-len 128 --warmup 50 --iters 200 \
  --seeds 1 2 3 4 5 \
  --out "$R/sweep_${TAG}_a100.csv" > "$R/sweep_${TAG}_a100.log" 2>&1
echo "  sweep rc=$?"

echo "=== [3/5] capacity probe (OOM ceiling) ==="
uv run python -m lora_serving.benchmark.run \
  --model "$M" --dtype fp16 \
  --adapters 47000 49000 51000 --batch-sizes 32 --lora-ranks 8 \
  --seq-len 128 --warmup 50 --iters 200 \
  --out "$R/sweep_${TAG}_capacity.csv" > "$R/sweep_${TAG}_capacity.log" 2>&1
echo "  capacity rc=$?"

echo "=== [4/5] PEFT mixed baseline, same node (~30 min) ==="
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
