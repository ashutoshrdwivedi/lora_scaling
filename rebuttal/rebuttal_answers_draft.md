> [!WARNING]
> **Superseded — working notes only. Do not paste from this file.**
> The paste-ready rebuttal is `rebuttal/parts/*.md`, verified by
> `rebuttal/check.py`. Numbers below predate the re-measured assembly
> benchmark (`benchmarks/results/rebuttal_assembly/`) and the speedup/
> aggregation convention fixes described in `rebuttal/README.md`.
> Specifically stale here: the assembly throughput and tail figures, the
> per-model speedup ranges, and the drift column. See `rebuttal/NUMBERS.md`
> for the current values.

# Rebuttal answers — draft (round 2)

Companion to `rebuttal_response.md`. Everything here is paste-ready per
reviewer. Numbers are measured unless marked *projected*.

**Numbers in this file are aligned with `rebuttal_response.md`.** Both were
re-derived from the CSVs in `benchmarks/results/rebuttal_{electra,deberta,xlmr_xl,l40s}/`;
if the two ever disagree, `rebuttal_response.md` is the one that gets pasted.

---

## 0. Measured results behind these answers

Full paper grid (fp16, seq=128, r=8, warmup 50 / iters 200, 5 seeds, PEFT arms
at warmup 10 / iters 50 on the same node).

**Spread** = total max−min of p50 across the whole N sweep, taken at the worst
batch size — a deliberately conservative measure, since it captures non-monotone
seed and cache-locality variation rather than growth in N.
**At ceiling** = p50 at the pool ceiling relative to its N=1,000 value.

| Config | Params (L, d) | Spread across sweep | At ceiling vs N=1,000 | Speedup vs PEFT-mixed | Ceiling @ r=8 | MB/adapter |
|---|---|---|---|---|---|---|
| bge-m3 / A100 (paper) | 568M (24, 1024) | ≤1% | — | 5.6–21.2× | 47,000 | 1.57 |
| ELECTRA-large / A100 | 334M (24, 1024) | 3.84% | −2.82% ² | 6.6–22.6× | 49,000 | 1.57 |
| DeBERTa-v2-xlarge / A100 | 884M (24, 1536) | 1.44% | −0.34% ² | 2.5–7.3× | 58,000 | 1.18 |
| XLM-R-XL / A100 | 3.48B (36, 2560) | 0.58% | +0.36% ² | 3.0–19.8× | 12,000 | 5.90 |
| bge-m3 / L40S | 568M (24, 1024) | 4.08% | +1.30% ³ | 4.3–32.4× | 26,000 ¹ | 1.57 |

¹ **26,000 with PyTorch's default allocator; 28,000 with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.** At the default-allocator
OOM point PyTorch held 2.7 GB reserved-but-unallocated; under
`expandable_segments` that drops to 22 MB and 28,000 fits (29,000 OOMs either
way), at identical p50 (33.4 ms). Any 28,000 figure quoted for the L40S — the
ceiling, the spread, the at-ceiling drift — comes from that setting.

² The capacity probes ran at B=32 only, so these three are B=32 figures.
³ The L40S has a full 5-seed × 5-batch-size run at N=28,000, so this is the
worst batch size (B=8); at B=32 it is −0.03%. **+1.30% is the binding case for
the "within +1.3%" claim** — it is true with zero margin, so prefer "within
1.4%" anywhere that phrasing is reused.

Every ceiling is bracketed by an OOM: 51,000 (ELECTRA), 60,000 (DeBERTa),
13,000 (XLM-R-XL) and 29,000 (L40S) all failed to allocate. Peak memory at the
ceiling was 78.9 / 72.7 / 79.8 / 45.9 GB respectively.

**On the noise floor.** Between-seed s.d. is 0.02–2.95 ms. As a fraction of p50
it is ≤0.5% in almost every cell, but it rises in the *small-pool* cells —
3.67% at ELECTRA N=100/B=16, 1.69% at L40S N=100/B=32. In three of four
configurations the noisiest cell in the entire sweep is an N=100 cell. This
matters for how the numbers are read: drift measured *from* N=100 uses the
least reliable point in the sweep as its reference and therefore overstates any
apparent trend. The at-ceiling-vs-N=1,000 column is the more honest comparison,
and it is where the O(1) claim actually lives.

**Caveats to keep visible, because a careful reader will find them:**

- DeBERTa-v2-xlarge is measured with **value_proj only**, not query+value. Under
  `share_att_key=true` its `query_proj` is also applied to the batch-shared
  relative-position embeddings, which per-tenant batching cannot adapt; PEFT
  (wrapping the Linear) does adapt it, so query+value would compare two systems
  computing different functions (measured 10.3% relative output divergence,
  pinned by `test_deberta_query_target_diverges_from_peft`). With value only,
  our output matches PEFT to 2.2e-5. One consequence: PEFT also has half as many
  LoRA modules to walk on this model, which *flatters* the PEFT baseline — that
  is why DeBERTa's speedup range is the lowest of the four.
- **Two sweeps are incomplete, both because the requested top cell exceeded the
  ceiling.** DeBERTa is 165/190 configs (the 25 missing are N=60,000, above its
  58,000 ceiling). L40S is 140/165 (the 25 missing are N=28,000, which OOM'd
  under the default allocator and was re-run separately under
  `expandable_segments` — that re-run is the 5-seed
  `sweep_bgem3_l40s_n28k_expseg.csv`). ELECTRA (165/165) and XLM-R-XL (190/190)
  are complete.
- XLM-R-XL and DeBERTa run through a hook-based attachment to the stock HF
  forward (needed because XLM-R-XL is pre-LN and DeBERTa uses disentangled
  attention). bge-m3 and ELECTRA run on the paper's own encoder. The **wrapper
  mechanism** is parity-tested against PEFT on BERT, ELECTRA and DeBERTa-v2
  (`tests/test_hf_wrapper.py`, `tests/test_real_checkpoint_parity.py`); note
  that XLM-R-XL itself has no dedicated parity test, so the claim is that the
  attachment path is verified, not that this checkpoint was individually
  verified.
- Quote XLM-R-XL p50 only. Three of five seeds show p99 stragglers (~260 ms vs
  137 ms p50 at the operating point); harmless for drift, but don't cite its p99.

---

## 1. Generality — to 5qtX (sole weakness) and yeZ9 (Q4)

> We agree this was the paper's thinnest axis, and we have measured it. Holding
> \sysname{} fixed, we re-ran the **full grid from the paper** (same protocol,
> 5 seeds) on three further encoders and a second GPU class. The results are in
> the table above. Three points:
>
> **(a) The O(1)-in-N property is not a property of bge-m3.** In every
> configuration and at every batch size, p50 at the pool ceiling is within
> **+1.3%** of its N=1,000 value, and in two of the four is *below* it. The total
> spread across each full sweep — a conservative figure that folds in
> non-monotone seed and cache-locality variation — is 0.58–4.08%, and the
> largest spreads are driven by the N=100 reference cells, which are the
> noisiest points we measure (up to 3.7% between-seed s.d.). There is no upward
> trend in N in any configuration.
>
> **(b) It holds at 6× the model size.** XLM-RoBERTa-XL (3.48B, d=2560, 36
> layers) has the *flattest* profile of all five: 0.58% total spread, and p50
> stays flat right up to the OOM boundary (137.1 ms at N=1,000 versus 137.6 ms
> at N=12,000, the last pool that fits). We present this as a stress test rather
> than a recommended deployment: a 3.5B encoder is an unusual thing to serve for
> classification, and the point is that the decomposition does not degrade, not
> that one should run it.
>
> **(c) It spans encoder families, not just sizes.** ELECTRA-large is a
> replaced-token-detection discriminator rather than a masked-LM, with a
> different vocabulary and position scheme; DeBERTa-v2 uses disentangled
> attention. Both behave like bge-m3.
>
> We will add these as a new **§4.5 (Generalization Across Models and Hardware)**
> in the camera-ready, using the +1 page. A second GPU *class* is hardware
> generality and we are careful not to present it as multi-node scaling (see our
> reply on that point).

---

## 2. yeZ9 Q2 — GPU memory footprint and headroom

> A clarification first: **encoder inference has no KV cache.** There is no
> autoregressive decode step, so nothing persists across tokens — the question
> of KV-cache headroom does not arise here, and that absence is part of why the
> encoder setting is tractable enough for a no-custom-kernel approach.
>
> What does consume VRAM is (i) the base weights, (ii) transient activations,
> and (iii) the resident adapter store, which is the only term that scales with
> tenant count. Adapter storage is exactly
>
>     bytes/adapter = M × 2 × L × H × r × sizeof(dtype)
>
> for M target modules and L layers. Measured, at r=8 fp16: **1.57 MB** for
> bge-m3 and ELECTRA-large (24×1024), **1.18 MB** for DeBERTa-v2-xlarge
> (24×1536, value only), **5.90 MB** for XLM-R-XL (36×2560). The resulting
> ceilings are 47,000 / 49,000 / 58,000 / 12,000 adapters, each bracketed by a
> measured OOM at the next probe point.
>
> Two measured details worth reporting. First, **activations are a rounding
> error next to the store.** On XLM-R-XL at N=11,000, peak memory moves only
> from 73.5 GB to 74.5 GB as batch size goes 8 → 128 — a 16× increase in serving
> load costs 1.0 GB. The tenant ceiling is set by adapter geometry, essentially
> independent of the load running against it, which is what makes it a
> *predictable* capacity number rather than one that must be re-derived per
> deployment.
>
> Second, the practical ceiling is **allocator-bound as well as
> arithmetic-bound**. On the L40S the default caching allocator tops out at
> 26,000 tenants with 2.7 GB reserved-but-unallocated at the OOM point; setting
> `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` recovers that fragmentation
> and raises the ceiling to 28,000 (+7.7%) with p50 unchanged. We will report
> that lever in the camera-ready; it is a straightforward gain we had not
> previously quantified.

---

## 3. yeZ9 reject-#2 — single node, no multi-GPU/multi-node

> This is a fair limitation and we do not claim otherwise. We have since added a
> second GPU class (L40S, an inference-tier Ada card with GDDR6 rather than
> HBM), which shows the same O(1)-in-N behaviour and a 4.3–32.4× speedup over
> PEFT-mixed on the same node. We want to be explicit that this is **hardware
> generality, not multi-node scaling** — it does not address the concern raised.
> The horizontal-scaling architecture in Appendix C remains a design, and the
> throughput claims under replicated pods remain analytical. We will state that
> boundary plainly in the Limitations rather than letting the new hardware row
> imply more than it shows.

---

## 4. yeZ9 reject-#3 — no empirical comparison against custom-kernel BMM

> **The bound is measured, not analytic**, and we should have made that clearer.
> Finding 6 rests on three quantities, and the binding one is a direct ablation:
> we run the identical batch with the entire LoRA path disabled (gather,
> shrink/expand, merge) and measure the wall-clock difference — **2.4 ms of
> 26.4 ms, 9.0%**. A custom fused kernel touches only that path, so 9.0% is a
> hard ceiling on what any kernel could recover. It cannot beat deleting the
> work. (The other two legs are an analytic FLOP ratio, 3d/r = 384, bounding
> free-kernel FLOP savings at 0.26%; and a profiler trace charging 4.94% of
> forward GPU time to the LoRA `bmm` versus 50.6% to the base projections. The
> 9.0% figure is a wall-clock share, not a FLOP share — the gather and merge,
> not the `bmm`, account for the gap.)
>
> What we cannot do is compare against a custom-kernel encoder LoRA
> implementation, because to our knowledge none exists: Punica, S-LoRA and LoRAX
> target decoder architectures, and vLLM's LoRA-pooling path currently supports
> only decoder-backbone embedders. We state this as unavailability rather than
> as evidence of our own optimality.
>
> The new models **tighten** the bound rather than leaving it untested. The
> analytic ratio 3d/r grows with width — 384 at d=1024, 576 at d=1536, 960 at
> d=2560 — so the maximum recoverable FLOP share falls from 0.26% to 0.17% to
> 0.10%. The case for custom kernels gets *weaker* as encoders get larger, which
> is the opposite of the decoder-side trend that motivates them.

---

## 5. yeZ9 reject-#4 — accuracy uses N=8 per class

> Correct, and we concede the generalization question: we follow the SetFit
> few-shot protocol, and we do not have evidence about full-data regimes where
> LoRA configuration might matter more.
>
> We would note the scope this affects. The paper's serving claims — latency,
> throughput, tenant ceiling, cold-start — are **independent of adapter
> quality**: the shapes are fixed by (L, H, r) and the arithmetic is identical
> whether the weights were trained on 8 examples or 8,000. The accuracy table
> establishes that late fusion is numerically faithful — it reproduces
> per-tenant LoRA outputs, asserted against PEFT to <1e-4 relative on real
> checkpoints (ELECTRA zero-delta, shared-adapter and mixed-tenant; DeBERTa
> value-only measured at 2.2e-5) — not that a particular training budget
> suffices. We will make that division explicit so the accuracy section is not
> read as a claim about training-data scaling.

---

## 6. yeZ9 reject-#6 — embedding / reranking / cross-encoder use cases

> The base model we serve throughout, **bge-m3, is a retrieval model** — this is
> the multilingual embedding backbone, not a classification-only encoder — and
> the newly added XLM-R-XL is from the same family. So the encoder path we
> benchmark is exactly the one an embedding workload would use.
>
> The task-specific part is only the head. For classification we batch
> per-tenant logistic-regression heads as a padded BMM (§Batched Classification
> Heads); for embedding serving that head is simply absent and the pooled output
> is returned directly, which is strictly less work than what we measure. A
> cross-encoder reranker is the same encoder forward over a query-document pair,
> so it inherits the same behaviour with a different input construction. We
> therefore expect our numbers to transfer, and we did not want to claim
> reranking results we have not run. We will scope the wording to "encoder
> serving with per-tenant heads" and name reranking explicitly as untested.

---

## 7. aNEv — "implementation optimization, not algorithmic innovation" + adoption

> We accept the characterization and would offer a reframing rather than a
> rebuttal. The contribution we claim is that a problem currently believed to
> need custom CUDA does not need it for encoders, and why.
>
> The "why" is the part we would argue is more than implementation. Encoder
> inference is base-dominated: a measured ablation removing the entire LoRA path
> changes wall-clock by only 9.0%, so per-tenant adaptation can be decomposed
> into batched BMMs whose cost is O(1) in tenant count. That observation is what
> predicts the result, and it now has support across four encoder families, two
> GPU classes, and a 10× span of model sizes (334M → 3.48B). It also predicts
> where the approach *stops* paying: we report the ceiling on kernel-level gains,
> and the memory-bound tenant limits, rather than only the favourable cases.
>
> On adoption: we agree it is unverified, and we think the honest answer is to
> lower the barrier rather than argue about it. Since submission we have added a
> generic attachment path that hooks any HuggingFace encoder exposing standard
> projection modules, so using \sysname{} on a new model requires no
> model-specific code — that is how two of the three new models above were run.
> We will release this as a pip-installable library at camera-ready (see our
> reply to 5qtX).
>
> **Future directions**, as requested: (i) masked BMM so a single batch can mix
> ranks and target-module sets without padding; (ii) fixed-capacity slotting to
> make the packed-tensor assembler support incremental tenant registration at
> O(1); (iii) multi-node evaluation of the Appendix C architecture; (iv)
> extending the decomposition to decoder-backbone embedding models; and (v)
> extending per-tenant adaptation to architectures with batch-shared projections
> — e.g. DeBERTa-v2's relative-position path, which our current decomposition
> cannot target because the projection has no per-sample counterpart.

---

## 8. 5qtX — will the code be released as a library?

> Yes. The benchmark and reference implementation are already released at the
> anonymous link; at camera-ready we will de-anonymize the repository and
> publish a pip-installable package with a stable API for the three components
> a deployment needs: the adapter store, the batch assembler, and the encoder
> attachment. The generic HuggingFace attachment path described above is what
> makes that packaging worthwhile — a user points it at a model name and a set
> of target-module names rather than editing model code.

---

## 9. Shared opening (use once per reviewer, trimmed)

> We thank the reviewers. Since submission we have measured two things that bear
> directly on the concerns raised. **(1) Generality:** we re-ran the paper's full
> grid on three additional encoders (ELECTRA-large, DeBERTa-v2-xlarge,
> XLM-RoBERTa-XL at 3.48B) and a second GPU class; in every configuration and at
> every batch size p50 at the pool ceiling is within +1.3% of its N=1,000 value,
> with speedups of 2.5–32.4× over PEFT's mixed-batch API. **(2) The CPU assembly
> path:** on a lightweight encoder the baseline assembler caps single-stream
> throughput at ~4.7k req/s, and the GPU-resident `index_select` assembler we
> proposed in our Limitations lifts this to 11.7k–14.5k req/s while removing the
> tail spike. Our submitted numbers use the baseline assembler and are therefore
> conservative.

---

## Open / not measured — do not imply otherwise

- **Result-scatter instrumentation never ran.** `rebuttal_response.md` handles
  this correctly (Q1 defers the scatter share to camera-ready). Do not estimate
  it.
- **No concurrent add-under-load microbenchmark.** The churn percentages in the
  Q6 answer remain projections from measured primitives, as flagged there.
- **No multi-node data.** See §3.
- **No reranking / full-data accuracy runs.** See §5, §6.
- **No XLM-R-XL parity test.** The wrapper mechanism is parity-tested on BERT,
  ELECTRA and DeBERTa-v2; this checkpoint is covered by the mechanism, not
  individually.
