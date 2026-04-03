# Multi-Tenant LoRA Serving: Project Overview

## What This Does

This project implements a GPU-accelerated inference system that serves **multiple tenant-specific text classification models simultaneously** using a single shared encoder. Each tenant has a unique LoRA (Low-Rank Adaptation) fine-tuned model and a logistic regression classification head. Instead of running a separate model per tenant — which doesn't scale — the system batches requests across tenants and applies each tenant's LoRA weights in a single GPU forward pass.

The base model is `intfloat/multilingual-e5-small` (a 12-layer transformer encoder, hidden size 384). All tenant adapters are pre-loaded into GPU memory. At inference time, a mixed-tenant batch is processed in one forward pass, with each sample's unique LoRA delta applied via batched matrix multiplication (BMM).

---

## Core Mechanism

LoRA modifies a weight matrix `W` as:

```
y = W₀x + (α/r) · B · A · x
```

Where `W₀` is the frozen base weight, and `A` (shrink) and `B` (expand) are the small per-tenant low-rank matrices. This system computes the `BAx` delta **without merging weights into the base model**, enabling heterogeneous tenants to be batched together.

The full batch forward pass for a batch of size N with different LoRA adapters:

1. Base projection: `proj = W₀ · x` (shared, single matmul)
2. LoRA shrink: `output_A = BMM(x, A_batch)` — each sample uses its tenant's A matrix
3. LoRA expand: `output_B = BMM(output_A, B_batch)` — each sample uses its tenant's B matrix
4. Fusion: `proj += output_B`
5. LR head: `logits = BMM(pooled, coef_batch) + intercept_batch`

Steps 2–5 use `torch.bmm` with pre-allocated output buffers to minimise memory allocation overhead.

---

## System Architecture

```
Incoming Requests (tenant_id, text)
           │
           ▼
   Tenant Lookup & Batch Assembly
   ┌─────────────────────────────────────┐
   │  - Fetch LoRA weights from GPU cache │
   │  - Stack into LayerwiseBatchedWeights│
   │  - Fetch LR head weights            │
   │  - Stack into BatchedLRWeights      │
   └─────────────────────────────────────┘
           │
           ▼
   Single GPU Forward Pass
   ┌─────────────────────────────────────────────────────┐
   │  TransformerEncoderForSequenceClassificationWithLora │
   │                                                     │
   │  Embeddings (word + position + token_type)          │
   │       │                                             │
   │  [Layer 0..11] TransformerEncoderLayerWithLora      │
   │       ├── Self-Attention + LoRA delta (Q, V)        │
   │       ├── Output projection + LayerNorm             │
   │       └── FFN (intermediate + output) + LayerNorm   │
   │       │                                             │
   │  Pooler (CLS token → dense → tanh)                  │
   │       │                                             │
   │  LRHeadServing (batched logistic regression)        │
   └─────────────────────────────────────────────────────┘
           │
           ▼
   Per-tenant class probabilities (batch_size × max_labels)
```

---

## Code Structure

```
lora_scaling/
├── adapter-scaling.md                          # Architecture design doc
├── scratchpad.py                               # Databricks benchmark notebook
└── adapter_scaling/
    ├── pyproject.toml
    ├── setup.py
    └── src/lora_serving/
        ├── models/
        │   └── transformer_encoder_lora_new.py  # Full model definition
        ├── ops/
        │   ├── lora_serve.py                    # BMM-based LoRA shrink/expand
        │   └── head_serve.py                    # Batched LR head inference
        └── utils/
            ├── lora.py                          # LoraWeight, TenantLoraWeights (ABC)
            ├── logreg.py                        # TenantLRWeights
            ├── pydantic_types.py                # LayerwiseBatchedWeights,
            │                                    # BatchedLogisticRegressionWeights
            ├── constants.py                     # QKV layer name constants
            └── cuda_errors.py                   # CUDA error types
```

### Key Classes

| Class | File | Role |
|---|---|---|
| `TransformerEncoderForSequenceClassificationWithLora` | `models/transformer_encoder_lora_new.py` | Top-level model; owns embeddings, encoder layers, pooler, LR head |
| `TransformerEncoderAttentionWithLora` | `models/transformer_encoder_lora_new.py` | Applies LoRA delta to Q and V projections via `LoraServing` |
| `LoraServing` | `ops/lora_serve.py` | Stateful class-level buffers for BMM; `lora_shrink` and `lora_expand` ops |
| `LRHeadServing` | `ops/head_serve.py` | Batched logistic regression via BMM + additive intercept |
| `LoraWeight` | `utils/lora.py` | Per-tenant storage of A and B matrices across all layers |
| `TenantLoraWeights` | `utils/lora.py` | Abstract base for tenant weight containers |
| `LayerwiseBatchedWeights` | `utils/pydantic_types.py` | Accumulates Q/K/V/output shrink+expand tensors per layer for a batch |
| `BatchedLogisticRegressionWeights` | `utils/pydantic_types.py` | Accumulates coef + intercept tensors for a batch |

---

## Benchmark Setup (`scratchpad.py`)

The Databricks notebook validates both correctness and latency:

- **Base model**: `intfloat/multilingual-e5-small` (384 hidden, 12 layers)
- **LoRA rank**: 8, applied to Q and V projections
- **Batch size**: 32
- **Adapters loaded**: 7 real accounts × 240 simulated accounts = ~1,680 unique (account, prediction_class) adapter pairs
- **Iterations**: 100 per benchmark
- **Correctness check**: `np.allclose(custom_output, setfit_output, atol=1e-6)` across all 32 samples in a batch
- **Baseline**: SetFit with the `adapters` library — one adapter activation + one LR head swap per forward pass (sequential, single-tenant)

### Metrics Reported
- Average, std, p50, p75, p90, p99 latency for custom LoRA serving
- Same metrics for SetFit baseline
- Weight loading time separately measured
