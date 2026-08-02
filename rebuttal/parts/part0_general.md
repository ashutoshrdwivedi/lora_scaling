We thank all reviewers for their careful reading. All three reviews rate Soundness 4. Across the reviews the main requests are (a) evidence beyond a single model, (b) evidence beyond a single GPU, and (c) characterization of the CPU-side assembly ceiling. Since submission we ran new experiments on all three axes using the paper's exact protocol (5 seeds, fp16, seq 128, $r{=}8$, warmup 50 / 200 timed iters).

**(a) Generality across encoder architectures and scale, and (b) across GPU class.** We hold the system fixed (without any model or hardware specific code) and vary the encoder and the GPU. Both headline results replicate on every configuration we tried: p50 stays flat in $N$ up to the memory ceiling, and the speedup over PEFT-mixed stays multi-fold:

| Config | Params (L, d) | GPU | p50 at ceiling vs $N{=}1000$ | Spread over sweep | Ceiling @ $r{=}8$ | Speedup vs PEFT-mixed |
|---|---|---|---|---|---|---|
| bge-m3 (paper) | 568M (24, 1024) | A100-80GB | +0.65% | 3.31% | 47,000 | 5.6–21.2× |
| ELECTRA-large | 334M (24, 1024) | A100-80GB | −0.16% | 1.36% | 51,000 | 6.2–20.9× |
| DeBERTa-v2-xlarge | 885M (24, 1536) | A100-80GB | +0.43% | 1.16% | 64,000 | 2.4–7.1× |
| XLM-RoBERTa-XL | 3.5B (36, 2560) | A100-80GB | −0.38% | 0.70% | 12,000 | 2.3–19.5× |
| bge-m3 | 568M (24, 1024) | L40S-48GB | +1.29% | 1.60% | 28,000 | 3.1–32.7× |


**Key Results & Replicated Findings:**
* **"p50 at ceiling vs $N{=}1000$":** Evaluates the $O(1)$-in-$N$ claim directly. Growing the pool by up to 64× over $N{=}1000$ moves p50 by at most **+1.29%**, and on two configs it is *negative*.
* **"Spread over sweep":** Measures the total max−min latency across the full sweep at its worst batch size. Because it is anchored at $N{=}100$, our noisiest cell (where seed-to-seed variance reaches 3.41% on the L40S versus $\le 0.5\%$ across the rest of the sweep), this metric naturally overstates any apparent trend.
* **"Speedup vs PEFT-mixed":** Compares throughput (samples/sec) against PEFT across the test settings PEFT is able to run ($N \in \{100, 1000\}$ and batch sizes $B \in \{8, 32, 128\}$).
* **"Ceiling @ $r{=}8$":** Represents GPU memory capacity limits. VRAM simply fills as per-adapter bytes × $N$ (5.90 MB/adapter for XLM-R-XL → 12,000; 1.18 MB for DeBERTa → 64,000), verified by an OOM error at the next test point.
* **Rank-insensitivity (Finding 3):** Sweeping $r \in \{4,8,16,32\}$ leaves p50 flat within 0.73% across all four new configs (measured in our broader sweep data beyond the table's $r{=}8$).

**(c) CPU assembly ceiling.** Our reported latencies are already end-to-end (CPU assembly + full GPU forward; Finding 5 decomposes them). To probe the high-QPS regime we ran the decomposition on all-MiniLM-L6-v2, where the CPU is the bottleneck by design: the baseline assembler caps single-stream throughput at **5,192 samples/s** with assembly at 58–65% of latency for $B \geq 64$. The GPU-resident `index_select` assembler proposed in our Limitations removes this in pure PyTorch — **12,152–15,315 samples/s** (2.47–3.12×), assembly share down to 1.8–4.4%, and the p99/p50 tail ratio falling from 5.28× to 1.03×. On bge-m3 it gives 1.14× ($B{=}8$) to 1.80× ($B{=}128$). We ran this decomposition on two different host CPUs: the baseline figure tracks the CPU, while the `index_select` ceiling at large batch reproduces on both — itself evidence that the assembler has left the CPU. The paper's numbers use the baseline assembler and are therefore conservative.

**Notes on the table above.** (1) The ceiling points for the three new A100 configs are single-seed probes at $B{=}32$; only bge-m3 and the L40S have all five batch sizes at their ceiling. (2) The L40S ceiling is allocator-bound as well as arithmetic-bound: PyTorch's default caching allocator tops out at 27,000, while `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` recovers the fragmented reserve and reaches 28,000 at identical p50 — **3.7% more tenants for free**; the next probe OOMs under both. The 28,000 figures use that setting. (3) On DeBERTa, adapters target the value projections only: DeBERTa-v2 shares `query_proj` between per-sample hidden states and the batch-shared relative-position embeddings (`share_att_key`), a call with no per-tenant counterpart; value-only serving matches PEFT's output to $2\times10^{-5}$ relative (unit-tested). Its lower speedup range follows from the same choice — with one wrapped module per layer PEFT has less mixed-batch overhead to pay, so that row is conservative for us, not favorable.

At camera-ready (+1 page) we will consolidate (a)+(b) into a new subsection, "Generalization Across Models and Hardware", and fold (c) into Finding 5 and the Limitations. We hope these address the main concerns.
