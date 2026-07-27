#!/bin/bash
# Rebuttal pod A1 -- A100-80GB REQUIRED -- XLM-RoBERTa-XL scale row.
#
# Answers 5qtX ("the empirical evidence could be even stronger if a larger
# encoder were tested"): 3.48B params, d=2560, L=36 -- 6.1x bge-m3's 568M.
#
# Runs on the stock HF forward with hook-injected LoRA (--engine hf). This is
# not optional: XLM-R-XL is pre-LN, so the repo's custom encoder cannot load it
# (146 state-dict key mismatches). The wrapper runs HF's own forward, so the
# base architecture is faithful by construction.
#
# NO SHARED-PROJECTION GAP. Unlike DeBERTa-v2 (share_att_key=true), XLM-R-XL's
# q/k/v are applied only to the per-sample hidden states -- verified by hooking
# every projection and confirming each is called exactly once per layer with
# the full batch dimension. So this row keeps the paper's query+value protocol
# unchanged and its PEFT comparison is exact, with no parity caveat to explain.
#
# SCOPE FRAMING: a 3.5B encoder is a stress test, not a recommended serving
# configuration. The claim is that the late-fusion decomposition still holds at
# this scale and that the per-GPU ceiling falls exactly as Finding 4 predicts --
# not that anyone should serve 3.5B for classification.
#
# Grid and measurement kernel mirror the paper's run_sxm80.sh: full N x B cross
# product at fp16, seq=128, r=8, warmup 50 / iters 200, 5 seeds, plus the same
# three rank cells at the (1000, 32) operating point. PEFT arms at warmup 10 /
# iters 50 and B=8/32/128, exactly as in the paper.
# 7 N x 5 B + 3 rank cells = 38 configs x 5 seeds = 190 runs.
#
# Ceiling: 2 modules x 2 (A,B) x 36 layers x 2560 x r8 x 2B = 5.90 MB/adapter.
# The model itself is ~7 GB fp16, leaving ~76 GB -> formula ~13k; applying the
# ~12% overestimate the paper saw for bge-m3 (formula 52.8k vs measured 47k)
# gives a practical ~11.4k. The sweep runs to 11k and the probe brackets it.
# This is Finding 4 behaving as predicted, not a regression: a 6x larger model
# with 2.5x the width buys correspondingly fewer resident tenants.
#
# DISK: the checkpoint is ~14 GB. Ensure the volume has headroom before
# starting or the download dies mid-run (see the HF_HOME note below).
#
# Runtime ~7-8h -- the long pole among the rebuttal scripts. The forward is
# ~9.4x bge-m3's (per-layer cost scales as 4d^2 + 2*d*ffn, so 2.5x width and
# 1.5x depth compound), and run_single_config reloads the 7 GB base model for
# every cell. Start this one FIRST. Independent of the other rebuttal scripts.
set -u
export HOME=/root
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
# Model downloads go to the mounted volume, not the container disk. XLM-R-XL
# alone is ~14 GB; a default 20 GB container disk cannot hold it plus the venv.
export HF_HOME=${HF_HOME:-/workspace/hf_cache}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/workspace/uv_cache}
export TMPDIR=${TMPDIR:-/workspace/tmp}
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"
cd /root/lora_scaling
R=benchmarks/results/rebuttal_xlmr_xl
mkdir -p "$R"

M="facebook/xlm-roberta-xl"
TAG=xlmrxl

echo "=== [0/5] preflight: GPU size + disk headroom ==="
uv run python -c "
import torch, shutil, sys
vram = torch.cuda.get_device_properties(0).total_memory/1e9
free = shutil.disk_usage('${HF_HOME}').free/1e9
print(f'  GPU: {torch.cuda.get_device_name(0)}  {vram:.1f} GB')
print(f'  free on \$HF_HOME: {free:.1f} GB')
if vram < 70: print('  WARNING: this row is sized for an 80GB A100; a smaller card lowers the ceiling')
if free < 20: print('  WARNING: <20 GB free; the ~14 GB checkpoint may not fit'); sys.exit(1)
" || { echo "PREFLIGHT FAILED"; exit 1; }

echo "=== [0b/5] smoke test (fail fast before burning the pod) ==="
uv run python -m lora_serving.benchmark.run \
  --model "$M" --engine hf --dtype fp16 --adapters 100 --batch-sizes 8 \
  --lora-ranks 8 --target-modules query value --seq-len 128 --warmup 5 --iters 10 \
  --out "$R/smoke_$TAG.csv" > "$R/smoke_$TAG.log" 2>&1
rc=$?; echo "  latefuse smoke rc=$rc"
[ $rc -ne 0 ] && { echo "SMOKE FAILED -- see $R/smoke_$TAG.log"; tail -20 "$R/smoke_$TAG.log"; exit 1; }

uv run python -m benchmarks.baselines.peft_swap \
  --model "$M" --dtype fp16 --lora-ranks 8 --seq-len 128 --batch-sizes 8 \
  --adapters 2 --mode mixed --target-modules query value \
  --warmup 2 --iters 5 \
  --out "$R/smoke_peft_$TAG.csv" > "$R/smoke_peft_$TAG.log" 2>&1
rc=$?; echo "  peft smoke rc=$rc"
[ $rc -ne 0 ] && { echo "PEFT SMOKE FAILED -- see $R/smoke_peft_$TAG.log"; tail -20 "$R/smoke_peft_$TAG.log"; exit 1; }

echo "=== [1/5] model + env metadata ==="
uv run python -m benchmarks.profiling.model_metadata \
  --model "facebook/xlm-roberta-xl:query,value" --out-dir "$R" \
  > "$R/model_metadata_$TAG.log" 2>&1
echo "  metadata rc=$?"

# --extra-configs folds the r!=8 cells into this same run so the shared
# (1000, 32, r8) cell is measured exactly once and all four ranks are timed
# back-to-back within each seed pass (one thermal envelope), matching the
# paper's rationale for not running ranks as a separate later sweep.
echo "=== [2/5] LateFuse sweep, full grid, 5 seeds (~6h) ==="
uv run python -m lora_serving.benchmark.run \
  --model "$M" --engine hf --dtype fp16 \
  --adapters 100 1000 2000 4000 6000 8000 11000 \
  --batch-sizes 8 16 32 64 128 --lora-ranks 8 --target-modules query value \
  --extra-configs 1000:32:4 1000:32:16 1000:32:32 \
  --seq-len 128 --warmup 50 --iters 200 \
  --seeds 1 2 3 4 5 \
  --out "$R/sweep_${TAG}_a100.csv" > "$R/sweep_${TAG}_a100.log" 2>&1
echo "  sweep rc=$?"

echo "=== [3/5] capacity probe (OOM ceiling) ==="
uv run python -m lora_serving.benchmark.run \
  --model "$M" --engine hf --dtype fp16 \
  --adapters 11000 12000 13000 --batch-sizes 32 --lora-ranks 8 \
  --target-modules query value --seq-len 128 --warmup 50 --iters 200 \
  --out "$R/sweep_${TAG}_capacity.csv" > "$R/sweep_${TAG}_capacity.log" 2>&1
echo "  capacity rc=$?"

# Only 'mixed' is cited in the rebuttal (speedup vs PEFT's native mixed-batch
# API) plus 'base' for the single-tenant ceiling. grouped / homogeneous /
# sequential are paper-only and skipped here.
echo "=== [4/5] PEFT mixed baseline, same node (~1h) ==="
uv run python -m benchmarks.baselines.peft_swap \
  --model "$M" --dtype fp16 --lora-ranks 8 --seq-len 128 \
  --batch-sizes 8 32 128 --adapters 100 1000 --mode mixed \
  --target-modules query value --warmup 10 --iters 50 \
  --out "$R/peft_mixed_$TAG.csv" > "$R/peft_mixed_$TAG.log" 2>&1
echo "  peft mixed rc=$?"

echo "=== [5/5] PEFT base ceiling ==="
uv run python -m benchmarks.baselines.peft_swap \
  --model "$M" --dtype fp16 --lora-ranks 8 --seq-len 128 \
  --batch-sizes 8 32 128 --adapters 1 --mode base \
  --target-modules query value --warmup 10 --iters 50 \
  --out "$R/peft_base_$TAG.csv" > "$R/peft_base_$TAG.log" 2>&1
echo "  peft base rc=$?"

echo "ALL DONE ($TAG)"
