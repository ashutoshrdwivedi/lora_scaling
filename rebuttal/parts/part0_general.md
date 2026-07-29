We thank all reviewers for their careful reading. All three reviews rate Soundness 4, and the recurring requests are (a) evidence beyond a single model, (b) evidence beyond a single GPU, and (c) characterization of the CPU-side assembly ceiling. Since submission we ran new experiments on all three axes using the paper's exact protocol (5 seeds, fp16, seq 128, $r{=}8$, warmup 50 / 200 timed iters).

**(a) Generality across encoder families and scale, and (b) across GPU class.** We hold the system fixed — no model- or hardware-specific code — and vary the encoder and the GPU. Both headline properties replicate:

| Config | Params (L, d) | GPU | p50 at ceiling vs $N{=}1000$ | Spread over sweep | Ceiling @ $r{=}8$ | Speedup vs PEFT-mixed |
|---|---|---|---|---|---|---|
| bge-m3 (paper) | 568M (24, 1024) | A100-80GB | +0.65% | 3.31% | 47,000 | 5.6–21.2× |
| ELECTRA-large | 334M (24, 1024) | A100-80GB | −2.57% | 2.97% | 49,000 | 6.2–22.8× |
| DeBERTa-v2-xlarge | 885M (24, 1536) | A100-80GB | −0.25% | 1.32% | 58,000 | 2.4–7.3× |
| XLM-RoBERTa-XL | 3.5B (36, 2560) | A100-80GB | +0.22% | 0.69% | 12,000 | 2.9–19.9× |
| bge-m3 | 568M (24, 1024) | L40S-48GB | +1.27% | 2.90% | 28,000 | 4.1–32.4× |

*Fallback if the table does not render:* ELECTRA-large, p50 at its 49,000 ceiling is 2.57% **below** its $N{=}1000$ value, total sweep spread 2.97%, 6.2–22.8× over PEFT-mixed. DeBERTa-v2-xlarge: −0.25%, 1.32%, 58,000, 2.4–7.3×. XLM-RoBERTa-XL: +0.22%, 0.69%, 12,000, 2.9–19.9×. bge-m3 on L40S-48GB: +1.27%, 2.90%, 28,000, 4.1–32.4×.

Reading the two latency columns: **"at ceiling"** is where the $O(1)$-in-$N$ claim lives — growing the pool by up to 58× over $N{=}1000$ moves p50 by at most **+1.27%**, and on two configs it is *negative*. **"Spread"** is the deliberately conservative total max−min across the whole sweep at its worst batch size; it is anchored at $N{=}100$, which is the *noisiest* point we measure (between-seed s.d. reaches 3.63% there on ELECTRA versus ≤0.5% in most cells), so it overstates any apparent trend. Speedups are throughput ratios over the $N \in \{100, 1000\}$ and $B \in \{8, 32, 128\}$ cells the PEFT arm covers. Ceilings are memory arithmetic, not performance limits — per-adapter bytes × $N$ filling the card (5.90 MB/adapter for XLM-R-XL → 12,000; 1.18 MB for DeBERTa → 58,000) — and each is bracketed by a measured OOM at the next probe. Rank-insensitivity (Finding 3) also replicates: sweeping $r \in \{4,8,16,32\}$ leaves p50 flat within 2.46% on all four new configs.

Two caveats we would rather state than have found. The ceiling points for the three A100 models are single-seed probes at $B{=}32$; only bge-m3 and the L40S have all five batch sizes at their ceiling. And the L40S ceiling is **allocator-bound as well as arithmetic-bound**: PyTorch's default caching allocator tops out at 26,000 with 2.7 GB reserved-but-unallocated, while `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` recovers that fragmentation and reaches 28,000 at identical p50, with 29,000 failing either way. The 28,000 figures use that setting.

DeBERTa note: adapters there target the value projections only. DeBERTa-v2 shares `query_proj` between per-sample hidden states and the batch-shared relative-position embeddings (`share_att_key`), a call with no per-tenant counterpart; value-only serving matches PEFT's output to $2\times10^{-5}$ relative (unit-tested). Its lower speedup range follows from the same choice — with one wrapped module per layer PEFT has less mixed-batch overhead to pay, so that row is conservative for us, not favorable.

**(c) CPU assembly ceiling.** Our reported latencies are already end-to-end (CPU assembly + full GPU forward; Finding 5 decomposes them). To probe the high-QPS regime we ran the decomposition on all-MiniLM-L6-v2, where the CPU is the bottleneck by design: the baseline assembler caps single-stream throughput at **6,804 samples/s** with assembly at 50–58% of latency for $B \geq 64$. The GPU-resident `index_select` assembler proposed in our Limitations removes this in pure PyTorch — **12,479–15,342 samples/s** (1.98–2.36×), assembly share down to 1.8–4.1%, and the p99/p50 tail ratio falling from 5.14× to 1.01×. On bge-m3 it gives 1.17× ($B{=}8$) to 1.51× ($B{=}128$). The paper's numbers use the baseline assembler and are therefore conservative.

At camera-ready (+1 page) we will consolidate (a)+(b) into a new subsection, "Generalization Across Models and Hardware", and fold (c) into Finding 5 and the Limitations. We would be grateful if these results are weighed in the final assessment.
