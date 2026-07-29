# Rebuttal answers — draft (round 2)

Companion to `rebuttal_response.md` (which lives untracked in the main checkout).
Everything here is paste-ready per reviewer. Numbers are measured unless marked
*projected*.

---

## 0. Measured results behind these answers

Full paper grid (fp16, seq=128, r=8, warmup 50 / iters 200, 5 seeds, PEFT arms
at warmup 10 / iters 50 on the same node). "Drift" = p50 change from the
smallest to the largest adapter pool at the B=32 operating point.

| Config | Params | Pool range | p50 drift | Speedup vs PEFT-mixed | Measured ceiling | MB/adapter |
|---|---|---|---|---|---|---|
| bge-m3 / A100 (paper) | 568M | 100 → 47,000 | **−0.60%** | 5.6–21.2× | 47,000 | 1.57 |
| ELECTRA-large / A100 | 335M | 100 → 47,000 | **−0.64%** | 6.2–22.8× | ≥49,000 | 1.57 |
| DeBERTa-v2-xlarge / A100 | 885M | 100 → 40,000 | **+0.55%** | 2.4–7.3× | 58,000 | 1.18 |
| XLM-R-XL / A100 | 3.48B | 100 → 11,000 | **−0.31%** | 2.9–19.9× | 12,000 | 5.90 |
| bge-m3 / L40S | 568M | 100 → 20,000 | +2.82% ¹ | 4.1–32.4× | 26,000 ² | 1.57 |

¹ From N=1,000 onward the drift is **+0.39%**; the N=100 cell is ~2% faster than
all larger pools, so the headline figure is dominated by that one point rather
than by any trend with N.
² 28,000 with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (see Q2).

Between-seed s.d. is 0.04–0.61 ms across every configuration (≤0.5% of p50), so
all five drift figures sit inside measurement noise.

**Caveats to keep visible, because a careful reader will find them:**
- DeBERTa-v2-xlarge is measured with **value_proj only**, not query+value. Under
  `share_att_key=true` its `query_proj` is also applied to the batch-shared
  relative-position embeddings, which per-tenant batching cannot adapt; PEFT
  (wrapping the Linear) does adapt it, so query+value would compare two systems
  computing different functions (measured 10.3% relative output divergence).
  With value only, our output matches PEFT to 2.2e-5. One consequence: PEFT also
  has half as many LoRA modules to walk on this model, which *flatters* the PEFT
  baseline — that is why DeBERTa's speedup range is the lowest of the four.
- DeBERTa's sweep is 165/190 configs; the 25 missing are all N=60,000, above the
  measured 58,000 ceiling.
- XLM-R-XL and DeBERTa run through a hook-based attachment to the stock HF
  forward (needed because XLM-R-XL is pre-LN and DeBERTa uses disentangled
  attention). bge-m3 and ELECTRA run on the paper's own encoder. Parity of the
  two paths is asserted in the test-suite.

---

## 1. Generality — to 5qtX (sole weakness) and yeZ9 (Q4)

> We agree this was the paper's thinnest axis, and we have measured it. Holding
> \sysname{} fixed, we re-ran the **full grid from the paper** (same protocol,
> 5 seeds) on three further encoders and a second GPU class. The results are in
> the table above. Three points:
>
> **(a) The O(1)-in-N property is not a property of bge-m3.** Across every
> configuration, p50 changes by **<0.7%** between the smallest and the largest
> adapter pool — in three of five cases the largest pool is nominally *faster*,
> which is noise. Between-seed s.d. is ≤0.5% of p50, so these drifts are not
> distinguishable from zero.
>
> **(b) It holds at 6× the model size.** XLM-RoBERTa-XL (3.48B, d=2560, 36
> layers) shows −0.31% drift out to 11,000 tenants, and p50 stays flat right up
> to the OOM boundary (137.6 ms at N=12,000, the last pool that fits). We
> present this as a stress test rather than a recommended deployment: a 3.5B
> encoder is an unusual thing to serve for classification, and the point is that
> the decomposition does not degrade, not that one should run it.
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
> ceilings are 47,000 / ≥49,000 / 58,000 / 12,000 adapters respectively, each
> confirmed by pushing to OOM.
>
> Two measured details worth reporting. First, **activations are a rounding
> error next to the store**: on XLM-R-XL at N=11,000, peak memory moves only
> from 73.50 GB to 74.55 GB as batch size goes 8 → 128. Serving load barely
> affects the ceiling; adapter geometry sets it. Second, the practical ceiling
> is **allocator-bound, not arithmetic-bound**: at the OOM boundary PyTorch held
> ~12 GiB reserved-but-unallocated, and setting
> `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` raised the bge-m3/L40S
> ceiling from 26,000 to 28,000 tenants. We will report that lever in the
> camera-ready; it is a straightforward gain we had not previously quantified.

---

## 3. yeZ9 reject-#2 — single node, no multi-GPU/multi-node

> This is a fair limitation and we do not claim otherwise. We have since added a
> second GPU class (L40S, an inference-tier Ada card with GDDR6 rather than
> HBM), which shows the same O(1)-in-N behaviour and a 4.1–32.4× speedup over
> PEFT-mixed on the same node. We want to be explicit that this is **hardware
> generality, not multi-node scaling** — it does not address the concern raised.
> The horizontal-scaling architecture in Appendix C remains a design, and the
> throughput claims under replicated pods remain analytical. We will state that
> boundary plainly in the Limitations rather than letting the new hardware row
> imply more than it shows.

---

## 4. yeZ9 reject-#3 — no empirical comparison against custom-kernel BMM

> Finding 6 is empirical rather than analytic, and we should have made its basis
> clearer. It rests on a direct ablation: we run the identical batch with the
> LoRA delta path disabled (`apply_lora=False`) and measure the wall-clock
> difference, which bounds what *any* faster delta implementation — custom
> kernel or otherwise — could recover. A kernel cannot beat deleting the work.
>
> What we cannot do is compare against a custom-kernel encoder LoRA
> implementation, because to our knowledge none exists: Punica, S-LoRA and LoRAX
> target decoder architectures, and vLLM's LoRA-pooling path currently supports
> only decoder-backbone embedders. We state this as unavailability rather than
> as evidence of our own optimality.
>
> The new models sharpen the bound rather than weaken it. The delta path adds
> 2r/d of a targeted projection's arithmetic, so it *shrinks* as models widen:
> ≈1.6% at d=1024 (bge-m3, ELECTRA), ≈1.0% at d=1536 (DeBERTa), ≈0.6% at d=2560
> (XLM-R-XL). The ceiling on kernel-level gains therefore narrows precisely in
> the regime where one would most expect custom kernels to pay off.

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
> establishes that late fusion is numerically faithful (it reproduces per-tenant
> LoRA outputs, verified against PEFT to ~1e-6 on the new models as well), not
> that a particular training budget suffices. We will make that division
> explicit so the accuracy section is not read as a claim about training-data
> scaling.

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
> rebuttal. This is a submission to the **Industry Track**, whose remit is
> deployable systems work; the contribution we claim is that a problem currently
> believed to need custom CUDA does not need it for encoders, and why.
>
> The "why" is the part we would argue is more than implementation. The
> observation is that encoder inference is base-dominated — the LoRA delta is a
> 2r/d fraction of a projection's arithmetic — so per-tenant adaptation can be
> decomposed into batched BMMs whose cost is O(1) in tenant count. That
> observation is what predicts the result, and it now has support across four
> encoder families, two GPU classes, and a 10× span of model sizes (335M →
> 3.48B). It also predicts where the approach *stops* paying: we report the
> ceiling on kernel-level gains, and the memory-bound tenant limits, rather than
> only the favourable cases.
>
> On adoption: we agree it is unverified, and we think the honest answer is to
> lower the barrier rather than argue about it. Since submission we have added a
> generic attachment path that hooks any HuggingFace encoder exposing standard
> projection modules, so using \sysname{} on a new model requires no
> model-specific code — that is how the three new models above were run. We will
> release this as a pip-installable library at camera-ready (see our reply to
> 5qtX).
>
> **Future directions**, as requested: (i) masked BMM so a single batch can mix
> ranks and target-module sets without padding; (ii) fixed-capacity slotting to
> make the packed-tensor assembler support incremental tenant registration at
> O(1); (iii) multi-node evaluation of the Appendix C architecture; (iv) extending
> per-tenant adaptation to architectures with batch-shared projections (e.g.
> DeBERTa's relative-position path), which our current decomposition cannot
> target.

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
> XLM-RoBERTa-XL at 3.48B) and a second GPU class; p50 drift across the tenant
> range is <0.7% everywhere, with speedups of 2.4–32.4× over PEFT's mixed-batch
> API. **(2) The CPU assembly path:** on a lightweight encoder the baseline
> assembler caps single-stream throughput at ~4.7k req/s, and the GPU-resident
> `index_select` assembler we proposed in our Limitations lifts this to
> 11.7k–14.5k req/s while removing the tail spike. Our submitted numbers use the
> baseline assembler and are therefore conservative.

---

## Open / not measured — do not imply otherwise

- **Result-scatter instrumentation never ran.** The `[X]/[Y]` placeholders in
  the CPU-path answer in `rebuttal_response.md` cannot be filled. Either drop
  that sentence or keep scatter as stated future work; do not estimate it.
- **No concurrent add-under-load microbenchmark.** The churn percentages in the
  Q6 answer remain projections from measured primitives, as already flagged there.
- **No multi-node data.** See §3.
- **No reranking / full-data accuracy runs.** See §5, §6.
