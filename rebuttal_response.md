# Rebuttal Response — Working Notes

Working scratchpad for the EMNLP 2026 Industry Track rebuttal. Not the final response — captures findings, measured data, draft answers, and framing decisions so far.

## Logistics

- **Venue:** EMNLP 2026 **Industry Track** (direct submission, *not* ARR — the July 7–13 ARR window does not apply).
- **Author-response deadline:** **2026-07-29, 11:59pm AoE (UTC-12).** Verify in the OpenReview portal (authoritative).
- **Format:** **Text-only rebuttal.** No PDF re-upload. Report new numbers in text; promise camera-ready edits. Notification 2026-08-20; camera-ready 2026-09-20 (gets +1 page, 6→7; Limitations/refs/appendix excluded from the limit).

## Scores

| Reviewer | Overall | Soundness | Excitement | Conf. | Read |
|---|---|---|---|---|---|
| 5qtX | **4 (Accept)** | 4 | 3.5 | 4 | Champion; sole weakness = "one model". AIK=4 but *wrong guess* (no public copy; anonymization verified clean). |
| yeZ9 | **3.5 (Borderline)** | 4 | 3.5 | 4 | Engaged, identity-blind → **the reviewer to convert (3.5→4)**. |
| aNEv | **3 (Workshop)** | 4 | 3 | 3 | Detractor, lowest confidence; "just implementation, not algorithmic". |

All three **Soundness 4** — no one disputes correctness. Fight is excitement/scope. "3 = Workshop" is a quality rung, **not** a workshop offer.

## Strategy (one line)

Lead with **new evidence**, concede the two genuine limitations gracefully, reframe "just engineering" via the Industry-Track remit + the FLOP insight. Don't try to win all twelve bullets — spend effort on the score-movers.

**Priority order:** (1) generality / 2nd encoder (moves *two* reviewers) — the score-mover; (2) CPU-path answer (data in hand); (3) reframe aNEv; (4) everything else = text.

---

## Concern map & status

| Concern | Reviewer(s) | Plan | Status |
|---|---|---|---|
| **Only one model / generality** | 5qtX (sole weakness), yeZ9 Q4 | ELECTRA-base serving sweep (beyond BERT/RoBERTa, low-risk) | **OPEN — needs GPU run (score-mover)** |
| **End-to-end latency incl. CPU/scatter** | yeZ9 Q1 | Already end-to-end; + MiniLM data; measure scatter | Data ✅, draft ✅, scatter pending |
| **>10k req/s CPU bottleneck** | yeZ9 Q3, reject-#5 | MiniLM CPU-regime + index_select | **Data ✅, draft ✅** |
| Mixed-rank in one batch | yeZ9 Q5, reject-#1 | Argue from Finding 3 (pad-to-max = homogeneous max-rank batch = latency-neutral; cost is memory not compute; bucketing avoids it) | **Draft ✅** (optional mixed-batch confirmation on pod) |
| GPU memory / "KV cache headroom" | yeZ9 Q2 | Finding 4 numbers + correct decoder assumption (encoders have no KV cache) | Unwritten |
| No empirical custom-kernel cmp | yeZ9 reject-#3 | Clarify Finding 6 *is* empirical (9% ablation) | Unwritten |
| N=8 few-shot only | yeZ9 reject-#4 | Concede + context (SetFit protocol; serving is accuracy-agnostic) | Unwritten |
| Single-node only | yeZ9 reject-#2 | Concede honestly (Appendix C architectural) | Unwritten |
| Embedding/rerank not eval'd | yeZ9 reject-#6 | Clarify scope (bge-m3 IS retrieval; heads are task-agnostic) | Unwritten |
| "Just implementation" + adoption | aNEv | Reframe: Industry Track + FLOP insight (quote yeZ9) + library commit | Unwritten |
| Future directions | aNEv | Add short future-work note | Unwritten |
| Release as library | 5qtX Q | Commit to pip library at camera-ready | Unwritten |
| Cold-start under churn | yeZ9 Q6 | Frame Finding 7: registration is O(1) + off-path → ≈0% of serving time (flat in N); PEFT O(N²) saturates a core under churn | **Draft ✅** (optional interleaved add-under-load microbench on pod) |
| Plagiarized preprint? | (author worry) | Checked web + arXiv API → **none exists**; field is all decoder-side | ✅ Resolved |

---

## Measured findings (new experiments)

### 1. index_select assembler — bge-m3 (paper's model)
A100-80GB, N=2000, r=8, seq=128, fp16, 5 seeds. `benchmarks/results/assembly_bench.txt`.

- End-to-end speedup **1.17× (B=8) → 1.79× (B=128)**.
- Assembly time: baseline O(B), 3.8→77.6 ms; index_select **flat ~0.5 ms**.
- Assembly share of latency: baseline **17%→46%**; index_select **2.2%→0.5%**.
- Tail p99/p50: baseline 4.1× (B=64) / 2.5× (B=128); index_select **~1.0×** (spike gone).
- `torch.compile` adds nothing (eager index_select already optimal).

### 2. index_select — all-MiniLM-L6-v2 (the >10k / CPU-regime demo)
Same config, `benchmarks/results/assembly_bench_minilm.txt`. Fast/shallow model exposes the CPU bottleneck sharply.

| Batch | Baseline req/s | index_select req/s | Speedup | Baseline asm share |
|---:|---:|---:|---:|---:|
| 64 | 4,710 | 11,656 | 2.47× | 57.3% |
| 128 | 4,730 | 13,619 | 2.88× | 61.9% |
| 256 | 4,572 | 14,455 | 3.16× | 66.0% |

- **Baseline caps ~4.7k req/s** — cannot reach 10k; CPU assembly is 57–66% of latency at B≥64.
- **index_select crosses 10k → 11.7k–14.5k req/s** (2.5–3.2×), assembly share → 2–5%.
- Tail p99/p50: baseline 4.6× (B=128); index_select ~1.0×.
- **The cap formula:** throughput → `1 / (per-sample assembly + per-sample forward)`. Baseline ≈ `1/0.21ms ≈ 4.7k` (assembly-dominated); index_select ≈ `1/0.069ms ≈ 14.5k` (forward-bound). CPU gather was the bottleneck, not the GPU.

### 3. Result-scatter instrumentation
Added to `benchmarks/profiling/assembly_bench.py` (scatter_results + timing + tables). **Not yet run.** Bundle with the ELECTRA pod session. Decision to confirm: measure on-device scatter only (excludes host D2H/serialization, matching the forward's boundary).

---

## Draft answer — CPU path (Q1 + Q3 + reject-#5, consolidated)

> Our reported latency is already end-to-end, not GPU-only: every number is CPU batch assembly (the per-sample LoRA gather) plus the full GPU forward including the batched LR head (§Benchmark Protocol); Finding 5 decomposes it. To address the high-QPS regime directly, we ran the end-to-end decomposition on a widely-deployed lightweight encoder (all-MiniLM-L6-v2): the baseline CPU assembler **caps single-stream throughput at ~4.7k req/s** with CPU assembly at 57–66% of latency — it cannot reach 10k because the CPU gather is the ceiling. The GPU-resident `index_select` assembler we proposed in our Limitations removes this, lifting throughput past 10k (to **11.7k–14.5k req/s, 2.5–3.2×**) and eliminating the tail spike (4.6×→1.0×), in pure PyTorch. **Our paper's serving numbers use the baseline assembler and are therefore conservative — this optimization only widens the margins.** Result scatter (per-tenant slice + softmax) is O(B·c_max) tensor indexing, [X] ms ([Y]%), negligible vs. assembly. Full production end-to-end including request queueing under dynamic batching is a serving-layer property (Appendix C) we characterize architecturally but do not benchmark — future work.

Brackets [X]/[Y] fill from the scatter run.

---

## Draft answer — Mixed rank in one batch (yeZ9 Q5 + reject-#1)

**Key move: the reviewer's worst case *is* Finding 3, already measured — no new run needed.**
Padding every adapter to the batch's max rank produces exactly a homogeneous max-rank
batch. Finding 3 sweeps the *uniform* batch rank r ∈ {4,8,16,32} — that homogeneous case —
and shows it is latency-neutral. So Finding 3 is an exact upper bound on any mixed batch
whose max rank is r_max.

### Facts / numbers (all from Table 2 / Findings 3–4)
- **Rank sweep, N=1000, B=32.** p50: r4=**40.8**, r8=40.4, r16=40.4, r32=**40.2** ms (spread <2%, and *decreasing* — pure noise). p99: 44.8 / 44.7 / 42.9 / 42.7 (±0.9/3.9/2.0/3.7 seed s.d.) → 5% spread, within between-seed noise. Forward stays **≈26.0 ms across all four ranks**.
- **FLOP share of the delta path = 2r/d** (bge-m3 d=1024): r8 ≈ **1.6%**, r32 ≈ **6.25%** of *one* projection's arithmetic → a fraction of a percent of the full encoder forward. Base path W₀x is rank-independent and dominates.
- **The real cost of rank is memory, not compute** (Finding 4): doubling r halves the tenant ceiling — 47k (r8) → 23.5k (r16) → 11.75k (r32). This is a per-adapter *storage* property, not a mixed-batch penalty.
- **Padding is transient, not resident.** The baseline assembler (the paper's system) stores each adapter at its own rank; pad-to-max only inflates the per-batch buffer, freed after the forward. No resident-memory waste.
- **Avoidable in deployment:** rank is a fixed small per-adapter integer known at registration → rank-bucketed batch formation (an O(1) scheduling policy over the *same* assembler, no kernel change) gives rank-homogeneous batches with zero padding. Masked-BMM (our stated future work) removes even the need to bucket.

### Blockquote (paste-ready)
> **Mixed rank in one batch (Q5).** The overhead identified is exactly the cost of running a batch at its maximum rank — and Finding 3 already measures that this is latency-neutral in the operating regime. Padding inflates only the LoRA *delta* path (BᵢAᵢx); the shared base path W₀x is rank-independent and dominates the forward. While r ≪ d the delta path adds just 2r/d of a single projection's arithmetic — ≈1.6% of the bge-m3 forward at r=8, ≈6% at r=32. Finding 3 confirms this: sweeping the *uniform* batch rank r ∈ {4,8,16,32} — i.e. forcing every adapter to that rank, precisely the worst case of pad-to-max — leaves p50 flat (40.8→40.2 ms, <2%, within seed noise) with the forward at ≈26 ms across all four ranks; the p99 spread is 5%, comparable to the between-seed s.d. A batch that pads a rank-4 tenant up to rank-32 therefore pays essentially no latency penalty. The genuine cost of higher rank is *memory*, not compute — doubling r halves the per-GPU tenant ceiling (47k→23.5k→11.75k, Finding 4) — and it is a per-adapter storage property, not a mixed-batch penalty: each adapter is stored at its own rank, and padding only inflates the transient per-batch buffer, which is freed after the forward. In deployment padding is avoidable entirely: rank is a fixed, small per-adapter integer known at registration, so rank-bucketed batch formation — an O(1) scheduling policy over the same assembler, with no kernel change — yields rank-homogeneous batches with zero padding. The masked-BMM variant we flag as future work removes even that, letting a single batch mix ranks and target modules via per-sample masking. We will sharpen the Limitations paragraph accordingly at camera-ready.

### Notes
- Same argument covers "same **target modules**": an adapter omitting a module contributes a zero delta there — bounded by the same tiny delta-path share, and eliminated by bucketing (by module set) or masked-BMM. Keep it to one clause; don't over-explain.
- **Optional cheap confirmation** (bundle on the ELECTRA/DeBERTa pod, not required): run one *genuinely* heterogeneous batch (e.g. 50% r=4 / 50% r=32) and show it matches the homogeneous r=32 latency. Gives a literal "we tested a mixed batch" sentence and closes the paper's `% TODO (Task #8)` in the Limitations paragraph. The text answer above stands without it.
- **Do NOT** concede this as a severe/open limitation — it reads as one in the current Limitations paragraph ("wasting compute… potentially severe"), but the data says the compute cost is ~0 in-regime and the only real axis is memory. The camera-ready edit should reframe from "wasting compute" → "trades tenant density for rank; padding is latency-neutral (Finding 3) and avoidable via rank-bucketing."

---

## Draft answer — Cold-start / registration % under churn (yeZ9 Q6)

**This is a *friendly* question — the reviewer calls Finding 7 "compelling" and wants the prod
translation. The answer is structurally clean: LateFuse registration is O(1) *and off the
serving path*, so its share of serving time is ≈0 by construction — independent of N and churn.
PEFT's is O(N) per add (O(N²) cumulative) and grows with the pool. Make the contrast the answer.**

### Facts / numbers (Finding 7 / §4.4, all measured)
- **Per-adapter registration cost.** LateFuse `AdapterStore` preload: **0.05 s @ N=100, 0.29 s @ N=1000** → ~**0.29 ms/adapter, flat** (linear-time, O(1) per add). PEFT `add_adapter`: **14.4 s @ N=100 → 1180.7 s @ N=1000** (O(N²) cumulative, exponent ≈1.91).
- **Cold-start speedup: 288×–4071×** (14.4/0.05 = 288 at N=100; 1180.7/0.29 = 4071 at N=1000 — the gap *widens* with N).
- **PEFT marginal add at N=1000 ≈ 2.4 s** — *projected* from the O(N²) fit (c·N, c≈2.36 ms), i.e. ~**8,000× LateFuse's 0.29 ms/add**, and rising with N.
- **Why LateFuse's share is ≈0:** a new tenant = one store insertion (memcpy of A/B + head into the resident store); serving requests select resident adapters by *index* (O(1) gather, already counted in assembly / Finding 5). No per-request `set_adapter` swap, no ModuleDict rescan.

### Churn translation (the "% of serving time" the reviewer asked for)
Take an **aggressive** churn rate: the entire N=1000 pool turns over **once per hour** (1000 adds/3600 s ≈ 0.28 adds/s — far above realistic onboarding/retrain rates):
- **LateFuse:** 0.28 × 0.29 ms ≈ **0.008 % of wall-clock**, and off the serving path entirely. At daily turnover it's ~0.0003 %.
- **PEFT:** 0.28 × ~2.4 s ≈ **0.67 s of registration per second** — ~67 % of a core on registration alone at N=1000, and it *worsens quadratically* with pool size (at larger N, registration can't keep up with churn at all).

**Punchline:** LateFuse's registration share is negligible (**<0.01 %**) across any realistic churn and **flat in N**; PEFT's grows with the pool and saturates a core at moderate pools. That is the real prod impact.

### Blockquote (paste-ready)
> **Registration under churn (Q6).** In \sysname{} the share of serving time spent on registration is ≈0 by construction, independent of pool size and churn rate. A new tenant is a single `AdapterStore` insertion — a memcpy of its A/B factors and head into the resident store — measured at ~0.29 ms/adapter and flat in N (0.05 s to preload 100, 0.29 s for 1,000). Serving requests never register or swap: they select resident adapters by index (the O(1) gather already counted in our assembly cost, Finding 5). So even under an aggressive churn rate — the entire 1,000-adapter pool replaced every hour — registration is ~0.008 % of wall-clock, and it runs off the serving path. The contrast is the production point: PEFT's `add_adapter` rescans the per-layer `ModuleDict` on every call (O(N) per add, O(N²) cumulative — the result the reviewer highlights), so at N=1,000 a single registration costs ≈2.4 s (~8,000× ours) and grows with the pool; under the same churn PEFT would spend the majority of a core on registration alone, and cannot keep pace at larger pools. Cold-start preload already reflects this end to end: 288×–4071× faster over N=100→1,000, a gap that widens by construction. We measure the per-add cost and the O(N²)-vs-linear scaling directly; to report the churn share as a single measured number we will add an interleaved add-under-load microbenchmark in the camera-ready.

### Notes
- **Honesty:** we measure (a) per-add cost 0.29 ms flat and (b) the O(N²)/linear cold-start scaling. We have **not** measured concurrent add-while-serving — "off the serving path" is an architectural property of the store (insertion is an independent O(1) op), and the churn % above is a *projection* from the measured primitives. The optional microbenchmark below turns it into a direct measurement.
- **Optional cheap confirmation** (bundle on the ELECTRA/DeBERTa pod): interleave a stream of adds with the serving loop and report registration's measured share for \sysname{} vs a PEFT `add_adapter` baseline. Confirms the ≈0 % directly and gives a literal churn number. Not required for the claim.
- **Do NOT** overclaim "0 %" — say ≈0 / <0.01 % / negligible. And keep the PEFT 2.4 s labelled as *projected from the O(N²) fit*, not measured-at-a-single-add.
- Ties to Finding 4 (memory) and the mixed-rank answer: LateFuse's per-tenant cost is O(1) in *both* compute and registration; the only per-tenant axis that scales is resident memory.
- **⚠️ index_select caveat:** this O(1)/off-path claim is the **BASELINE** assembler, NOT index_select — the packed-tensor path snapshots the store at construction (`batch.py:150–153`) and has no incremental-add path (naive add = O(N²)). Keep the churn and index_select threads separate; see **Framing guardrail #6**. Do not let a "we also have index_select" sentence bleed into this answer.

---

## Generality — §4.5 Generalization (score-mover: Q4 + 5qtX)

DeBERTa / a larger encoder / bge-m3 on a second GPU all belong in **one new subsection, §4.5 "Generalization Across Models and Hardware"** — already stubbed as the `% TODO (Task #4)` block in `paper/main.tex`. This is body content (uses the camera-ready +1 page).

**These are generalization, NOT ablations.** Ablation = vary a component of *LateFuse* (no-LoRA path, buffers, `index_select` vs baseline, rank). Generalization = same system, different *external substrate* (model, GPU). Swapping the base model/GPU is the latter — and it must be *visible* (it answers reviewer concerns), not buried under "ablation."

| Experiment | Axis | Reviewer |
|---|---|---|
| DeBERTa-v3-base | multi-model (different architecture) | yeZ9 Q4 ("beyond BERT/RoBERTa") |
| larger encoder | multi-model (scale) | 5qtX (sole weakness) |
| bge-m3 on 2nd GPU | multi-hardware | "single A100" specificity |

⚠️ **Different GPU ≠ multi-node.** A 2nd GPU class shows hardware generality; it does NOT address reject-#2 (multi-GPU/node cluster) — that stays future work (Appendix C). Don't conflate.

### Compression rules (keeps it to ~⅓ column)
The detailed N×B×r behavior is already in Table 2 for bge-m3; §4.5 only shows it *transfers*, so:
1. **One row per (model, GPU)** — each row summarizes a whole sweep, not the grid.
2. **Report only the invariants** — p50 drift as N→ceiling (≈0 ⇒ O(1)-in-N holds) and speedup vs PEFT-mixed. Not raw latency at every N/B/r.
3. **Fix off-axis knobs** — r=8, operating-point batch. No new axes.
Detail (flat-curve figures, full tables) → appendix, off the page clock.

### Shape A — paper §4.5 (camera-ready)
> LateFuse's claims are properties of the late-fusion BMM decomposition, not of bge-m3 or the A100. Holding the system fixed, we vary the base encoder and the GPU class; we report only whether p50 stays flat as the pool scales (O(1)-in-N, Finding 1) and the PEFT-mixed speedup (Finding 7). Full sweeps in Appendix X.
>
> **Finding 8: The O(1)-in-N property and PEFT speedups hold across encoder families, sizes, and GPU classes.**

| Config | Params (L, d) | GPU | p50 drift (N: 100→ceiling) | Throughput (smp/s) | Speedup vs PEFT-mixed |
|---|---|---|---|---|---|
| **bge-m3** (ref) | 568M (24, 1024) | A100-80GB | ≤1% | 801 | 5.6–21.2× |
| DeBERTa-v3-base | 184M (12, 768) | A100-80GB | [·] | [·] | [·]× |
| *[larger encoder]* | [·] | A100-80GB | [·] | [·] | [·]× |
| bge-m3 | 568M (24, 1024) | *[GPU-2]* | [·] | [·] | [·]× |

### Shape B — rebuttal subset (report only MEASURED rows; no placeholders)
> **To 5qtX and yeZ9 (Q4):** bge-m3 alone understates generality. LateFuse has no model/hardware-specific code, and we demonstrate this on **DeBERTa-v3-base** (disentangled attention, beyond BERT/RoBERTa). The O(1)-in-N property and PEFT speedups hold:
>
> | Config | GPU | p50 drift (N: 100→ceiling) | Speedup vs PEFT-mixed |
> |---|---|---|---|
> | bge-m3 (paper) | A100-80GB | ≤1% | 5.6–21.2× |
> | DeBERTa-v3-base | A100-80GB | [x]% | [x]× |
>
> We will consolidate this into a new **§4.5 (Generalization Across Models and Hardware)** in the camera-ready, adding a larger encoder and a second GPU class.

The rebuttal table is a **subset of the §4.5 table in the same shape** — measure what you can by July 29, report those rows, §4.5 adds the rest. Run via the **main serving sweep (`run.py`)**, NOT `assembly_bench.py`; results → `benchmarks/results/` as new sweep CSVs. Do NOT include accuracy or the index_select/CPU-path work here (different threads). Fallback if OpenReview mangles the table: one inline sentence per row.

---

## Framing decisions (guardrails — do NOT violate)

1. **Additive, not re-baseline.** Report index_select as a *measured optimization on the residual bottleneck*, reported alongside the baseline. Main table + Findings 1–7 + method stay unchanged. **Do not** rewire `run.py` or regenerate `sweep_main`.
2. **Baseline is "conservative."** Frame current numbers as lower bounds that index_select widens. This turns "we didn't use the fast path" into a strength.
3. **Promise only the additive result.** ❌ Never write "we will make index_select the default serving path" — implies a full re-baseline, signals provisional numbers, loses Finding 5, and gives **zero** acceptance upside over reporting the numbers. Re-baselining can only hurt/neutral, so don't.
4. **Honesty caveats to keep:**
   - assembly_bench is **single-stream** (assemble→forward sequential); "throughput" = batch/latency, not concurrent/pipelined QPS.
   - Result scatter number is measured on-device only; host serialization is serving-layer.
   - Never imply we measured network/queueing (production) latency.
5. **Never** reference reviewer identity-knowledge or confirm/deny authorship. 5qtX's AIK=4 was a wrong guess; anonymization verified clean (PDF metadata, source, no self-cites). No leak, no plagiarized preprint.
6. **index_select churn caveat — keep the Q6 (churn) and index_select (throughput) threads DECOUPLED.** The current `IndexSelectBatchAssembler` packs the store into one contiguous (N,L,H,R) tensor *at construction* and does **not** support incremental adds (`src/lora_serving/weights/batch.py:150–153`: "adapters added afterwards are not reflected"). A naive add = re-`stack` all N = **O(N) copy, O(N²) cumulative** — which would *reintroduce the PEFT pathology* Finding 7 beats. So **Q6's O(1)/off-path registration claim rests on the BASELINE assembler** (the paper's served system: per-adapter `AdapterStore`, O(1) dict insert), NOT index_select. index_select is an *assembly-throughput* optimization only (Finding 5 / CPU-path / >10k). ❌ Never claim index_select is the fast path AND O(1) under churn together — yeZ9 asks exactly these questions and would catch it. If pushed: making index_select O(1) under churn is **fixed-capacity slotting** (pre-allocate to the Finding-4 ceiling, 47k @ r=8; O(1) slot writes, eviction reuses freed rows) — note as design/future work, **NOT implemented, NOT measured**.

---

## Open items / TODO

- [ ] **ELECTRA-base serving sweep** (Q4 score-mover) — O(1) latency + throughput vs PEFT + capacity; serving sweep only, NOT accuracy. Needs a fresh GPU pod (prior A100 deleted).
- [ ] Run **scatter instrumentation** on MiniLM (bundle with ELECTRA pod). Fill [X]/[Y] in the CPU-path draft.
- [ ] Draft remaining reviewer answers (Q2, reject #2/#3/#4/#6, aNEv reframe, library commit). *(Q5 mixed-rank ✅, Q6 churn ✅ drafted.)*
- [ ] Assemble the shared "what's new since submission" opening (2 small tables: generality + index_select).
- [ ] Optional: `deberta-v3-base` as a stretch generality model (stronger claim, medium integration risk).

**Camera-ready only (do NOT do now):** tighten the Limitations ">10k req/s" wording to match the per-node data; add index_select as a labeled variant row in Table 2 (optional middle path, keeps Finding 5); de-anonymize repo + add library; restore authors/acks.
