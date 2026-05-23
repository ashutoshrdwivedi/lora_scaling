# Multi-seed quality benchmark — runbook for a larger GPU

This runbook is for re-running the LoRA-SetFit vs vanilla SetFit comparison
with **10 seeds** (matching SetFit paper) across **5 method-LR configs**
(vanilla@2e-5, lora@2e-5, lora@1e-4, lora@3e-4, lora@5e-4) on **7 datasets**.

**Total: 7 × 5 × 10 = 350 training runs.**

## What changed since the last single-seed run

- Multi-seed: `--seeds` accepts a list (default 0..9, matching paper's 10 seeds)
- LoRA LR sweep: `--lora-lrs` runs LoRA at additional LRs beyond `body_lr`
- AmazonCF now uses Matthews Correlation Coefficient (matches paper Table 6)
- Resume support: `--resume` skips configs already in the CSV (safe re-runs)
- CSV schema: now includes `body_lr`, `metric`, `score` (renamed from `accuracy`)

The script lives at `benchmarks/quality/setfit_compare.py`.

## GPU recommendation

| GPU | VRAM | Est. wall-clock | Notes |
|---|---|---|---|
| T4 16GB (current) | 14.5 GB | ~5-6 hr | Tight on memory; we hit OOM at seq=512 |
| **L4 24GB (recommended)** | 24 GB | **~3-4 hr** | Best $/perf for this workload |
| A10G 24GB | 24 GB | ~3 hr | Slightly faster than L4 |
| A100 40GB | 40 GB | ~2 hr | Overkill; bottleneck is small-batch overhead, not raw FLOPs |
| H100 80GB | 80 GB | ~1.5 hr | Massive overkill |

**Recommendation: L4 24GB.** The workload is 110M-param MPNet at batch=16, seq=256
— the bottleneck is Python loop overhead and many short training runs, not raw
compute throughput. L4 has enough memory to never OOM (even if we want to bump
seq_length later) and is much cheaper than A100/H100.

If L4 is not available, A10G 24GB is roughly equivalent.

## Setup on the new box

```bash
# 1. Clone repo (or rsync your working tree if you have local changes)
git clone <repo-url> && cd lora_scaling
git checkout peft_equivalance   # current branch

# 2. Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install Python deps including quality extras
uv sync --extra quality

# 4. Smoke test (single dataset, single seed — ~1 min)
uv run python benchmarks/quality/setfit_compare.py \
    --base-model sentence-transformers/paraphrase-mpnet-base-v2 \
    -t q -t v \
    --datasets SetFit/sst2 \
    --seeds 42 \
    --lora-lrs 1e-4 \
    --out /tmp/smoke.csv
```

The smoke test should produce 3 rows (vanilla@2e-5, lora@2e-5, lora@1e-4) and
print a summary. If that works, kick off the full run.

## The full run

```bash
uv run python -u benchmarks/quality/setfit_compare.py \
    --base-model sentence-transformers/paraphrase-mpnet-base-v2 \
    -t q -t v \
    --datasets SetFit/sst2 \
    --datasets SetFit/sst5 \
    --datasets SetFit/CR \
    --datasets SetFit/amazon_counterfactual_en \
    --datasets SetFit/emotion \
    --datasets SetFit/enron_spam \
    --datasets SetFit/ag_news \
    --seeds 0 --seeds 1 --seeds 2 --seeds 3 --seeds 4 \
    --seeds 5 --seeds 6 --seeds 7 --seeds 8 --seeds 9 \
    --lora-lrs 1e-4 --lora-lrs 3e-4 --lora-lrs 5e-4 \
    --max-seq-length 256 \
    --body-lr 2e-5 \
    --n-per-class 8 \
    --lora-rank 8 \
    --out benchmarks/quality/setfit_mpnet_multiseed.csv \
    --resume \
    2>&1 | tee benchmarks/quality/setfit_mpnet_multiseed.log
```

Notes:
- `python -u` disables stdout buffering so progress prints in real time.
- `--resume` lets you re-invoke after a crash and pick up where it left off.
- `tee` captures logs to disk; useful if SSH disconnects.
- Run inside `tmux` or `nohup` if the connection isn't stable.

## Sanity checks

1. **Vanilla numbers should reproduce SetFit paper Table 2** (mean across 10
   seeds, within ±1σ of paper's std):

   | Dataset | Paper N=8 | Expected vanilla mean |
   |---|---|---|
   | SST-5 | 43.6 ± 3.0 | within ±3.0 |
   | AmazonCF (MCC) | 40.3 ± 11.8 | within ±11.8 |
   | CR | 88.5 ± 1.9 | within ±1.9 |
   | Emotion | 48.8 ± 4.5 | within ±4.5 |
   | EnronSpam | 90.1 ± 3.4 | within ±3.4 |
   | AG News | 82.9 ± 2.8 | within ±2.8 |

   If vanilla is more than 2σ off, something is wrong with the setup.

2. **LoRA at body_lr=2e-5 (the "paper LR" condition) should sit ~6-7pp below
   vanilla** based on our single-seed prior run.

3. **At least one of {1e-4, 3e-4, 5e-4} should close most of the LoRA gap.**
   If all three diverge (collapse to random like 1e-3 did on vanilla), the LR
   range was too high; rerun with smaller values.

## Expected output

- `benchmarks/quality/setfit_mpnet_multiseed.csv` — 350 rows.
- `benchmarks/quality/setfit_mpnet_multiseed.log` — full stdout/stderr.

CSV columns:
```
dataset, method, lora_rank, body_lr, n_per_class, seed, metric, score,
trainable_body_params, train_seconds, n_eval
```

## After the run finishes

Once you have the CSV back, regenerate `RESULTS.md` with mean±std rows. Tell me
"the multiseed run is done" and I'll do the analysis and write up the table.

## If something goes wrong

- **OOM** → reduce `--batch-size` to 8 (the only paper-deviation we'd accept) and re-run with `--resume`.
- **Crash mid-run** → just re-invoke with `--resume`. CSV is flushed after each row, so at most one config is lost.
- **A specific dataset hangs/errors** → comment its `--datasets` line out and run the rest; come back to it later with `--resume`.
- **Some seeds produce NaN scores** → SetFit's contrastive loss occasionally collapses on bad seeds. Note in the writeup; don't drop those rows from the mean unless you can justify a reason.
