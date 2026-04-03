# Why This Is Novel: Gap Analysis in Multi-Tenant LoRA Serving

## The Problem

When serving ML models for multiple tenants (accounts), each with their own fine-tuned model, the naive approach is to run N separate inference processes — one per tenant. This doesn't scale: GPU memory, model loading overhead, and compute cost all grow linearly with the number of tenants.

The solution — batching requests across tenants and sharing the base model computation — is well-studied for **decoder/generative LLMs** but has no production-ready open-source implementation for **encoder models used in classification and embedding tasks**.

This project fills that gap.

---

## What Exists Today

### 1. Punica (MLSys 2024)
**Paper**: "Punica: Multi-Tenant LoRA Serving" — Chen et al., 2024
**GitHub**: https://github.com/punica-ai/punica

Punica introduced the SGMV (Segmented Gather Matrix-Vector Multiplication) CUDA kernel, which batches LoRA computations across different adapters in a single GPU pass. It is the foundational work in this space.

**Limitations for our use case**:
- Exclusively designed for **decoder-only generative LLMs** (LLaMA, Mistral)
- Tied to autoregressive inference with KV-cache management
- No support for encoder models (no bidirectional attention path, no pooling, no classification head)
- The Punica paper itself cites PetS as prior work on encoder serving and explicitly frames Punica as solving a *different* (generative) problem

---

### 2. S-LoRA (MLSys 2024)
**Paper**: "S-LoRA: Serving Thousands of Concurrent LoRA Adapters" — Sheng et al., 2024
**Blog**: https://lmsys.org/blog/2023-11-15-slora/

S-LoRA extends Punica with unified paging for LoRA adapter weights and KV caches, enabling thousands of concurrent adapters. It is the state-of-the-art for LLM multi-tenant serving.

**Limitations for our use case**:
- Explicitly targets **autoregressive (decoder-only) inference**
- The S-LoRA paper directly references PetS as the encoder-serving counterpart and treats encoder and decoder serving as separate problems
- Unified paging is architecturally coupled to KV caches, which don't exist in encoder inference
- No classification head, no pooling path

---

### 3. vLLM (Multi-LoRA)
**GitHub**: https://github.com/vllm-project/vllm
**Docs**: https://docs.vllm.ai/en/latest/features/lora/

vLLM integrates Punica/S-LoRA's BGMV/SGMV kernels for multi-adapter serving. It is the most widely used LLM serving framework.

**Limitations for our use case**:
- Multi-LoRA support is implemented **only for decoder-based generative models**
- Encoder model support is partial and actively broken for LoRA: GitHub issue #12808 documents that LoRA weights are silently not loaded for embedding/pooling models
- Multi-adapter LoRA for embedding models (issue #30058) has been open since late 2025 with no resolution
- No timeline for fixing encoder + multi-LoRA

---

### 4. LoRAX (Predibase)
**GitHub**: https://github.com/predibase/lorax

LoRAX supports heterogeneous continuous batching of different LoRA adapters for fine-tuned LLMs. Claims BERT support in their roadmap and mentions a customer use case with BERT-based classification at ~8.5M daily requests.

**Limitations for our use case**:
- Architecture and documentation centre on **decoder-only generative models**
- Whether heterogeneous multi-adapter batching (multiple different BERT LoRA adapters in a single forward pass) is supported — vs. single-adapter BERT serving — is unconfirmed in documentation
- BGMV/SGMV kernels are tuned for the token-generation pattern, not prefill-only encoder serving

---

### 5. HuggingFace Text Embeddings Inference (TEI)
**GitHub**: https://github.com/huggingface/text-embeddings-inference

TEI is HuggingFace's production serving stack specifically for encoder/embedding models (BERT, XLM-RoBERTa, E5-family, ModernBERT). It supports high-throughput batching with Flash Attention and cuBLASLt.

**Limitations for our use case**:
- **No LoRA adapter support whatsoever** — no single-adapter loading, no multi-adapter batching
- Fixed-weight serving only
- Would need to be extended from scratch to support per-tenant LoRA

---

### 6. PetS (USENIX ATC 2022)
**Paper**: "PetS: A Unified Framework for Parameter-Efficient Transformers Serving" — Zhou et al., 2022
**Link**: https://www.usenix.org/conference/atc22/presentation/zhou-zhe

PetS is the **only published system that directly targets multi-adapter serving for encoder-only models**. It supports batching requests across different PET (Parameter-Efficient Transformer) adapters — including adapters, BitFit, MaskBERT — on BERT and DistilBERT, reporting 4–26x more concurrent tasks and 1.5–1.6x throughput improvement.

**Limitations for our use case**:
- Published in 2022, **before LoRA became the dominant PEFT method** — does not support LoRA-style low-rank matrix adapters
- Designed for BERT/DistilBERT at a single-GPU scale; not architected for thousands of adapters
- Not updated or maintained for the LoRA era
- Does not handle per-tenant classification heads (logistic regression or otherwise)

---

### 7. Jina Embeddings v3
**Paper**: arXiv 2409.10173

Jina-embeddings-v3 ships a single encoder model with 5 task-specific LoRA adapters built in. Batch inputs specify which adapter to use via an integer task descriptor, enabling mixed-adapter batching.

**Limitations for our use case**:
- This is a **model design**, not a serving framework — the adapters are fixed and baked into the model weights at training time
- Not a general solution for serving N independently fine-tuned tenant adapters
- Does not address the problem of loading and serving arbitrary external adapters

---

### 8. Other Recent Work (2024–2025)
All recent systems address generative LLMs exclusively:
- **MixLoRA** (ICPP 2024): CUDA-stream-based multi-LoRA for LLMs
- **CaraServe**: CPU-assisted rank-aware LoRA for generative LLMs
- **EdgeLoRA** (2025): Multi-tenant LLM serving on edge devices
- **Symbiosis** (arXiv 2507.03220): Multi-adapter inference; evaluated on LLaMA2, GPT2-XL, Gemma2, StarCoder — no encoder models

---

## Gap Summary

| System | Encoder Support | Multi-Adapter in One Pass | LoRA-Specific | Scales to 1000s | Classification Head |
|---|---|---|---|---|---|
| Punica | No | Yes (decoders) | Yes | Yes | No |
| S-LoRA | No | Yes (decoders) | Yes | Yes | No |
| vLLM multi-LoRA | Broken | Yes (decoders) | Yes | Partial | No |
| LoRAX | Unconfirmed | Unconfirmed | Yes | Yes | No |
| TEI | Yes | No | No | Yes | No |
| PetS | Yes (BERT) | Yes | No (pre-LoRA) | No | No |
| Jina-v3 | Yes | Yes (fixed adapters) | Yes | No | No |
| **This project** | **Yes** | **Yes** | **Yes** | **Yes (pre-loaded)** | **Yes (per-tenant LR)** |

---

## What Makes This Novel

### 1. Encoder-native multi-tenant LoRA batching
No existing open-source system correctly batches multiple independent LoRA adapters in a single encoder forward pass. This project is effectively a **port of Punica's core batching idea to the encoder setting**, a gap the Punica and S-LoRA authors themselves acknowledged but did not address.

### 2. Per-tenant classification heads in the same forward pass
Beyond LoRA, each tenant has a different logistic regression head with a different number of output classes. The system handles variable-length output via zero-padding and `BatchedLogisticRegressionWeights`, executing all classification heads as a single BMM. This is not addressed by any existing system.

### 3. No weight merging
Unlike naive approaches that merge LoRA weights into the base model per tenant (requiring a full model reload per request), this system applies LoRA as an additive delta (`y = W₀x + BAx`) without mutating base weights. This enables mixed-tenant batching that would otherwise be impossible.

### 4. Practical deployment without custom CUDA kernels
Punica and S-LoRA require custom CUDA kernel compilation (SGMV/BGMV), which is often impractical in managed environments (Databricks, SageMaker, etc.). This system achieves the same batching semantics using standard PyTorch `torch.bmm` with pre-allocated output buffers, making it deployable anywhere PyTorch runs — at the cost of some kernel efficiency.

### 5. Pre-loaded adapter cache
All ~1,680 tenant adapters are pre-loaded into GPU memory, eliminating adapter loading latency from the critical inference path. Weight assembly (stacking tensors into `LayerwiseBatchedWeights`) is the only per-request overhead before the forward pass.

---

## Tradeoffs vs. Punica/S-LoRA

| Dimension | Punica/S-LoRA | This Project |
|---|---|---|
| Kernel efficiency | Custom SGMV CUDA kernels | Standard PyTorch BMM |
| Model type | Decoder LLMs | Encoder classifiers |
| Deployment | Requires CUDA compilation | Runs anywhere with PyTorch |
| Task head | Token generation | Per-tenant logistic regression |
| Adapter scale | Thousands (paged) | Pre-loaded GPU memory |
| Correctness validation | Implicit | Explicit atol=1e-6 vs. SetFit baseline |
