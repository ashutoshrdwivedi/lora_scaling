We thank the reviewer for the assessment and address both concerns.

**"Implementation optimization, not algorithmic innovation."** We agree the artifact is a practical systems contribution, but the claim is not a code-path optimization alone. What we contribute is an observation with a measured consequence: per-tenant adaptation decomposes as $W_0 x + B_i A_i x$, and encoder inference is *base-dominated*, so sharing the base path across a mixed-tenant batch makes serving cost $O(1)$ in tenant count **by construction** rather than by tuning, up to the memory ceiling set by adapter geometry.

The evidence that this is the operative mechanism, not an implementation artifact, is an ablation: running the identical batch with the entire LoRA path disabled — the per-layer gather of $(A_i, B_i)$, shrink/expand and merge — changes the GPU forward by **2.4 ms of 26.4 ms, or 9.0%**. Batch assembly is identical in both arms and sits outside the timed region, so this isolates the delta path itself. That is what makes the negative result in Finding 6 sharp: a custom fused kernel can only touch that path, so 9.0% bounds what any kernel could recover.

Since submission, we added a generic HuggingFace attachment wrapper for encoders exposing standard projection modules. We use it across four encoders spanning three pretraining families, two GPU classes, and a 10× span of model sizes (334M → 3.5B), with the same serving path throughout. This reproduces the paper's results on each (general-response table).

This is why we view the contribution as addressing a live deployment gap: vLLM's LoRA-pooling path covers only decoder-backbone embedders, and PEFT's mixed-batch API is 5.6–21.2× slower at our operating points while registering adapters in $O(N^2)$ cumulative time.

**Adoption by the open-source community.** We agree this cannot be claimed in advance, so our focus is on lowering the barrier to evaluation and reuse. The benchmark repository is already public (anonymized), and since submission we have added a generic attachment path that hooks any HuggingFace encoder exposing standard projection modules, so a new encoder needs no reimplementation of the base forward — at most a target-module name mapping. That is how all three new encoders above were run. At camera-ready we will release the serving path as a pip-installable library.

**Future directions.** We plan to release our work as a pip installable library for  multi-tenant LoRa serving, do a quantitative multi-node evaluation, add extend the 
wrapper to support Reranking and Sequence Classification as well (Embedding Generation / Search / Classification is already supported).

