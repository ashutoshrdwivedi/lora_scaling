# EMNLP Submission — Readiness TODO

Project: Multi-tenant LoRA serving for encoder models.
Status as of 2026-05-12: working PyTorch implementation, 48-config benchmark on a single L4, synthetic adapters only, no published baselines, no real-task evaluation, no paper draft.

This document tracks everything that needs to happen before the work is submission-ready. Tasks are also mirrored in the TaskList for execution tracking.

---

## A. Honest current-state assessment

**What we have:**
- Working `EncoderWithLora` that batches different LoRA adapters per sample via `torch.bmm` with pre-allocated buffers.
- Per-tenant logistic-regression heads (zero-padded BMM).
- 48-config sweep (4 adapter counts × 4 batch sizes × 3 ranks) on `intfloat/multilingual-e5-small` on an L4. Results are in `docs/03-benchmark-results.md` / `docs/benchmark_results.csv`.
- Unit tests for ops and weight assembly; one GPU-gated end-to-end test.
- Gap analysis vs. Punica / S-LoRA / vLLM / LoRAX / TEI / PetS in `docs/02-novelty-and-gap-analysis.md`.
- FastAPI/k8s deployment skeleton (`deploy/`) — unmeasured.

**What is missing for an EMNLP-grade contribution:**
- No empirical baselines — only our own numbers. Latency comparisons in ARCHITECTURE.md are estimated, not measured.
- No real downstream-task accuracy — adapters are random tensors with B=0.
- Single GPU (L4), single base model (e5-small). Reviewers will not accept this generalisation claim.
- No statistical analysis (single-seed runs, no CIs).
- No ablation of the design choices (BMM buffers, target modules, rank uniformity).
- No paper draft, no figures, no related-work section.
- `store.load_from_file` is untested against a real PEFT-saved adapter.

---

## B. Suggested target venue

Best fit is one of:
- **EMNLP Industry Track** — applied serving systems with industrial evaluation. Strong fit if real-deployment metrics (QPS, SLA, cost) are added (Task #10).
- **EMNLP Findings** — main-track-quality, slightly narrower scope. Fits if we add real-task accuracy + baselines but stay efficiency-focused.
- **EMNLP Demo** — 4-page demo paper around the FastAPI server. Lowest bar but smallest contribution.

Recommendation: aim for Industry Track. Falls back gracefully to Findings if eval is solid but deployment story is weak.

→ Task #15 confirms 2026 deadlines, page limits, anonymity policy.

---

## C. Work plan grouped by priority

### P0 — blockers (cannot submit without these)

1. **Baselines** (Task #2): At minimum HF PEFT sequential serving, naive merge/un-merge, and a documented vLLM/LoRAX attempt on encoders. Without these the latency claims have no comparison point.
2. **Real-task evaluation** (Task #3): Train real adapters on GLUE / MASSIVE / Banking77 / MTEB-classification. Confirm output matches PEFT-merged reference at `atol≤1e-5` and report task accuracy.
3. **End-to-end real-adapter validation** (Task #9): Wire up `store.load_from_file` against real `pytorch_adapter.bin`, integration-test correctness.
4. **Paper draft** (Task #1): abstract, intro, method, experiments scaffolding. Block other writing tasks on this.
5. **Venue + deadline confirmed** (Task #15).

### P1 — must-have for a competitive submission

6. **Multi-GPU × multi-model matrix** (Task #4): minimum 3 GPU classes × 3 base encoders. Otherwise reviewers reject the generality claim.
7. **Statistical rigor** (Task #6): ≥5 seeds, mean±std, 95% CI. Persist env snapshot.
8. **Ablation table** (Task #7): isolate buffer reuse, contiguous store, target modules, fp16/bf16, gather kernel.
9. **Profile-driven latency breakdown** (Task #11): replace estimated numbers in ARCHITECTURE.md with measured nsys / torch.profiler data.
10. **Related work section** (Task #12): expand `02-novelty-and-gap-analysis.md` into a proper survey.
11. **Reproducibility package** (Task #13): pinned deps, single repro command, EMNLP reproducibility checklist filled.
12. **Publication-quality figures** (Task #14).

### P2 — strengthens the paper, expand if time allows

13. **Scaling break-points** (Task #5): push adapter count to OOM, plot the cliff.
14. **Mixed-rank serving** (Task #8): turn a known limitation into an evaluated extension.
15. **Serving harness benchmarks** (Task #10): QPS-at-SLA against the FastAPI server for Industry-Track framing.
16. **Negative results section** (Task #16): regimes where the system loses to baselines.

---

## D. Suggested paper outline (working draft target)

1. **Abstract** — encoder-side LoRA multi-tenant serving gap; late-fusion BMM; results headline.
2. **Introduction** — SaaS classification setting, cost constraint, why decoder-side work doesn't transfer.
3. **Background** — LoRA math; serving-system primitives; encoder vs. decoder inference loop differences.
4. **Method** — late-fusion identity, batch assembly, pre-allocated BMM buffers, per-tenant LR head.
5. **Implementation notes** — pure-PyTorch path (no custom CUDA), deployability.
6. **Experiments** —
   - 6.1 Setup (GPUs, models, datasets, baselines, seeds).
   - 6.2 Throughput and latency vs. adapter count and batch size.
   - 6.3 Comparison against PEFT-sequential and weight-merging baselines.
   - 6.4 Real-task accuracy vs. PEFT-merged reference.
   - 6.5 Multi-GPU × multi-model generalisation.
   - 6.6 Ablations.
7. **Discussion** — limitations, mixed-rank extension, comparison to Punica/S-LoRA.
8. **Related work**.
9. **Conclusion**.
10. **Reproducibility statement** and **Limitations** sections (EMNLP requires both).

---

## E. Open questions to resolve early

- Which real adapter set do we train? Need 100+ adapters with meaningfully different label spaces — GLUE only gives ~8. Candidates: per-locale MASSIVE intents (60 locales × intent classifier each), per-domain MTEB classification subsets, or our own collection of synthetically-fine-tuned BERT adapters on news/tweet topic datasets.
- Do we ship a single-encoder paper (e5-small only with strong eval) or a generality paper (3+ encoders, lighter eval per model)? Reviewers usually prefer the latter for a serving systems paper.
- Anonymous artifact submission policy for 2026 — does the GitHub repo need to be re-hosted anonymously?

---

## F. Tracking

Live task list is in TaskList. This file mirrors the plan in human-readable form; update both when scope changes.
