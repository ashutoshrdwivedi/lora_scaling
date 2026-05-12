# Multi-Tenant LoRA Serving: Problem, Theory & Implementation

A deep technical explanation of the `lora_serving` system — what problem it solves, why naive approaches fail, and how this implementation achieves efficient multi-tenant inference on a single GPU.

---

## Table of Contents

1. [The Business Problem](#1-the-business-problem)
2. [Background: How Fine-Tuning Works](#2-background-how-fine-tuning-works)
3. [Background: What LoRA Is](#3-background-what-lora-is)
4. [The Serving Problem](#4-the-serving-problem)
5. [Approach 1: One Model Per Tenant (Naive)](#5-approach-1-one-model-per-tenant-naive)
6. [Approach 2: Weight Swapping (Standard LoRA Serving)](#6-approach-2-weight-swapping-standard-lora-serving)
7. [Approach 3: Late Fusion with Batched BMM (This Implementation)](#7-approach-3-late-fusion-with-batched-bmm-this-implementation)
8. [Code Walkthrough](#8-code-walkthrough)
9. [Memory Analysis](#9-memory-analysis)
10. [Latency Analysis](#10-latency-analysis)
11. [Limitations & Future Work](#11-limitations--future-work)

---

## 1. The Business Problem

Consider a SaaS platform that provides text classification (spam detection, sentiment analysis, intent recognition) to hundreds of enterprise customers. Each customer has their own labeled data, their own label taxonomy, and expects high accuracy on *their* specific domain.

**The requirement:** Serve a "custom model" for each customer, where:
- Customer A has a 3-class spam filter trained on their support emails
- Customer B has a 25-class intent classifier trained on their chatbot logs
- Customer C has a binary toxicity detector trained on their user-generated content
- All 500+ customers need sub-100ms latency and 99.9% uptime

**The constraint:** GPUs cost $2–$30/hour each. You cannot afford one GPU per customer.

---

## 2. Background: How Fine-Tuning Works

A pretrained encoder model (like BERT, RoBERTa, or multilingual-E5) has learned general language representations from massive corpora. Fine-tuning adapts these representations to a specific task by updating *all* model weights on the customer's labeled data.

For a model with `d = 384` hidden dimensions and `L = 12` layers:
- Each attention layer has Q, K, V projection matrices, each of size `(384, 384)` = 147,456 parameters
- Total trainable parameters: ~33 million

**Full fine-tuning creates a complete copy of the model.** If you have 500 customers, you need to store and serve 500 × 33M = 16.5 billion parameters. At float32 (4 bytes each), that's **66 GB** of GPU memory — and this is for a *small* model.

---

## 3. Background: What LoRA Is

### The Core Insight

LoRA (Low-Rank Adaptation) is based on a key observation from [Aghajanyan et al., 2020]: when you fine-tune a large pretrained model, the weight updates have **low intrinsic rank**. That is, the matrix $\Delta W = W_{fine-tuned} - W_{pretrained}$ can be approximated by the product of two much smaller matrices.

### The Math

For a pretrained weight matrix $W_0 \in \mathbb{R}^{d \times d}$, instead of learning a full $\Delta W \in \mathbb{R}^{d \times d}$, LoRA decomposes it as:

$$\Delta W = B \cdot A$$

where:
- $A \in \mathbb{R}^{r \times d}$ — the **"shrink"** matrix (projects from `d` dimensions down to `r`)
- $B \in \mathbb{R}^{d \times r}$ — the **"expand"** matrix (projects from `r` dimensions back to `d`)
- $r \ll d$ — the **rank** (typically 4, 8, or 16)

The forward pass becomes:

$$h = W_0 x + B A x$$

### Parameter Savings

For one projection matrix with `d = 384` and `r = 8`:
- Full fine-tuning: `384 × 384 = 147,456` parameters
- LoRA: `384 × 8 + 8 × 384 = 6,144` parameters

That's a **24× reduction** per projection. Across all layers and target modules, a LoRA adapter is typically ~0.5–2% of the base model size.

### Initialization

During training:
- Matrix $A$ is initialized from a Gaussian distribution $\mathcal{N}(0, \sigma^2)$
- Matrix $B$ is initialized to **zeros**

This zero initialization of $B$ is mathematically critical: at the start of training, $\Delta W = B \cdot A = 0 \cdot A = 0$. This means the model starts exactly at the pretrained weights and the LoRA gradually learns the task-specific adaptation. Without this, you'd start from a random perturbation of the pretrained model, which destroys the pretrained representations.

### What Gets Trained

During LoRA fine-tuning:
- The base model weights $W_0$ are **frozen** (no gradients computed)
- Only $A$ and $B$ receive gradients and are updated

This reduces:
1. **Trainable parameters** (fewer weights to optimize)
2. **Optimizer state memory** (Adam stores 2 states per parameter — this scales with parameter count)
3. **Gradient memory** (no gradients needed for the frozen base)

The VRAM savings come primarily from (2) and (3), not (1). Adam optimizer states alone use 2× the parameter memory, so freezing 98% of the model saves enormous amounts of VRAM even though the LoRA matrices themselves are tiny.

---

## 4. The Serving Problem

Training LoRA is well-solved by libraries like PEFT and Hugging Face. The hard problem is **serving** hundreds of different LoRA adapters simultaneously, with low latency, on shared GPU hardware.

The fundamental tension:

| Goal | Requirement |
|---|---|
| High throughput | Large batch sizes (GPUs are parallel processors) |
| Multi-tenant | Different adapters per request in the same batch |
| Low latency | No per-request overhead for adapter loading |
| Cost efficiency | Minimize GPU count |

These goals conflict with each other in naive implementations.

---

## 5. Approach 1: One Model Per Tenant (Naive)

**Strategy:** Load a complete fine-tuned model for each tenant into GPU memory.

```
GPU Memory:
┌─────────────────────────┐
│ Model for Tenant A (33M)│  130 MB
│ Model for Tenant B (33M)│  130 MB
│ Model for Tenant C (33M)│  130 MB
│          ...             │
│ Model for Tenant N (33M)│  130 MB
└─────────────────────────┘
```

**Why it fails:**
- A single model instance takes ~130 MB (float32) to ~65 MB (float16)
- An A100 (80 GB) can hold ~600 model instances at float32
- But you also need memory for activations, KV cache, and batch tensors
- Realistically, you get ~100–200 tenants per GPU
- At $2/hour per GPU, 500 tenants = ~3–5 GPUs = $5,000–$10,000/month just for the models
- Scaling to 10,000 tenants is economically impossible

**Also:** Each model is a separate process or thread. You cannot batch requests across tenants, so each request runs at batch_size=1, which wastes ~95% of the GPU's parallel compute capacity.

---

## 6. Approach 2: Weight Swapping (Standard LoRA Serving)

**Strategy:** Keep one base model in GPU memory. Before each request, merge the tenant's LoRA weights into the base model, run inference, then un-merge them.

```python
# Pseudocode for weight swapping
def serve_request(tenant_id, input_text):
    adapter = load_adapter(tenant_id)
    
    # Merge: W = W0 + B*A
    for layer in model.layers:
        layer.query.weight.data += adapter.B_q @ adapter.A_q
        layer.value.weight.data += adapter.B_v @ adapter.A_v
    
    output = model(tokenize(input_text))
    
    # Un-merge: W = W0 (restore original)
    for layer in model.layers:
        layer.query.weight.data -= adapter.B_q @ adapter.A_q
        layer.value.weight.data -= adapter.B_v @ adapter.A_v
    
    return output
```

**Why it fails at scale:**

### Problem 1: Serial Execution
The merge/un-merge cycle takes ~5–10ms. During this time, the GPU is blocked — no other request can be processed. With 1000 requests/second, you spend 5–10 seconds/second just swapping weights. The math doesn't work.

### Problem 2: No Cross-Tenant Batching
Because the base model is physically mutated with one tenant's adapter, you cannot process Tenant A and Tenant B in the same batch. Every request is batch_size=1, which means:
- GPU utilization: ~5–10% (GPUs are designed for batch_size=32+)
- Throughput: ~100 requests/second instead of ~3000

### Problem 3: Memory Fragmentation
`weight.data += ...` triggers in-place modification of CUDA tensors. PyTorch's memory allocator does not handle frequent in-place mutations gracefully — it leads to fragmentation, where the GPU has enough total free memory but can't allocate a contiguous block. This causes random OOM crashes under load.

### Problem 4: Cross-Tenant Leakage Risk
If the merge happens but the un-merge doesn't (crash, exception, timeout), the next tenant's request runs through a model contaminated with the previous tenant's adapter. This is a correctness and security disaster.

---

## 7. Approach 3: Late Fusion with Batched BMM (This Implementation)

### The Key Idea

**Never modify the base model weights.** Instead, compute the base output and the LoRA delta **separately**, then add them together.

Recall the LoRA forward pass:
$$h = W_0 x + B A x$$

This can be decomposed into two independent computations:
1. **Base path:** $h_{base} = W_0 x$ — uses the frozen base model
2. **Delta path:** $h_{delta} = B (A x)$ — uses the LoRA matrices

Since the base model is never touched, we can:
- Run the base path for an entire **mixed-tenant batch** at once
- Compute the delta path per-tenant using **Batch Matrix Multiplication (BMM)**
- Add them together

### Why BMM Enables Cross-Tenant Batching

Consider a batch of 4 requests from 3 different tenants:

```
Batch item 0: Tenant A, text="hello world"
Batch item 1: Tenant B, text="great product"
Batch item 2: Tenant A, text="another query"
Batch item 3: Tenant C, text="test input"
```

**Base path** — identical for all items (same frozen weights):
```
input:  (4, S, H) — 4 items, S tokens, H dimensions
W_0:    (H, H)    — same weight matrix for all items
output: (4, S, H) — standard batched matrix multiply
```

**Delta path** — different A/B matrices per item:
```
A_weights: (4, H, R)  — stacked: [A_tenantA, A_tenantB, A_tenantA, A_tenantC]
B_weights: (4, R, H)  — stacked: [B_tenantA, B_tenantB, B_tenantA, B_tenantC]

Step 1 (shrink):  torch.bmm(input, A_weights) → (4, S, R)
   - Item 0: (1, S, H) @ (1, H, R) → (1, S, R)  using Tenant A's A matrix
   - Item 1: (1, S, H) @ (1, H, R) → (1, S, R)  using Tenant B's A matrix
   - Item 2: (1, S, H) @ (1, H, R) → (1, S, R)  using Tenant A's A matrix
   - Item 3: (1, S, H) @ (1, H, R) → (1, S, R)  using Tenant C's A matrix
   All 4 happen in ONE GPU kernel call.

Step 2 (expand): torch.bmm(shrink_out, B_weights) → (4, S, H)
   Same pattern — one kernel call, all 4 items processed in parallel.
```

**`torch.bmm` vs `torch.matmul`:** Standard `matmul` broadcasts the weight matrix, applying the *same* matrix to every item in the batch. `bmm` (Batch Matrix Multiply) applies a *different* matrix to each item. This is what enables heterogeneous batching — each item in the batch can use a completely different LoRA adapter.

### The Fusion Step

```python
output = base_output + delta_output  # element-wise add, (B, S, H)
```

This produces the same result as if we had merged the LoRA weights into the base model, but:
- The base model was never modified
- All tenants were processed in a single batch
- The GPU did 2 BMM calls instead of N sequential merge/inference/un-merge cycles

---

## 8. Code Walkthrough

### `config.py` — Configuration

```python
@dataclass
class LoraServingConfig:
    model_name: str                    # e.g., "intfloat/multilingual-e5-small"
    lora_rank: int                     # e.g., 8
    batch_size: int                    # e.g., 32
    max_seq_len: int                   # e.g., 512
    target_modules: list[str]          # e.g., ["query", "value"]
    device: torch.device               # e.g., cuda:0
    dtype: torch.dtype                 # e.g., torch.float32
```

All architecture dimensions (hidden_size, num_layers, num_heads) are derived automatically from the HuggingFace model config. Nothing is hardcoded — changing `model_name` automatically reconfigures everything.

### `weights/store.py` — Adapter Storage

**`LoraWeight`** stores the A and B matrices for one adapter across all layers:
```python
class LoraWeight:
    wa: Tensor  # (num_layers, hidden_size, lora_rank) — A matrices
    wb: Tensor  # (num_layers, lora_rank, hidden_size) — B matrices
```

Note the transposed storage: A is stored as `(H, R)` not `(R, H)`. This is because during inference we compute `x @ A` where x is `(B, S, H)`, so A needs to be `(H, R)` for the matrix multiply to work without transposing at runtime.

**`AdapterStore`** is a dictionary mapping adapter IDs to `LoraWeight` objects, all pre-loaded on GPU:
```python
store = AdapterStore(config)
store.load_synthetic("tenant_42", seed=42)  # random weights for benchmarking
store.load_from_file("tenant_43", "/path/to/adapter.bin", key_fn)  # real weights
weight = store.get("tenant_42")  # retrieve for inference
```

### `weights/batch.py` — Batch Assembly

This is the bridge between "per-tenant storage" and "batched GPU tensors."

**`LayerwiseBatchedWeights`** holds, for one transformer layer, the stacked A/B matrices for every item in the batch:
```python
class LayerwiseBatchedWeights:
    a: dict[str, list[Tensor]]  # module_name → [A_item0, A_item1, ..., A_itemB]
    b: dict[str, list[Tensor]]  # module_name → [B_item0, B_item1, ..., B_itemB]
```

**`BatchAssembler.assemble()`** gathers the weights:
```python
for adapter_id in batch_adapter_ids:
    weight = store.get(adapter_id)
    for layer_idx in range(num_layers):
        for module in target_modules:
            lora_weights[layer_idx].a[module].append(weight.wa[layer_idx].unsqueeze(0))
            lora_weights[layer_idx].b[module].append(weight.wb[layer_idx].unsqueeze(0))
```

The `unsqueeze(0)` adds a batch dimension so that `torch.cat(list)` produces `(B, H, R)`.

### `ops/lora.py` — The Hot Path

```python
class LoraOps:
    def __init__(self, config):
        B, S, H, R = config.batch_size, config.max_seq_len, config.hidden_size, config.lora_rank
        self._out_A = torch.empty(B, S, R, ...)  # pre-allocated buffer
        self._out_B = torch.empty(B, S, H, ...)  # pre-allocated buffer

    def shrink(self, x, a_weights):
        # x: (B, S, H), a_weights: (B, H, R)
        torch.bmm(x, a_weights, out=self._out_A)  # → (B, S, R)

    def expand(self, b_weights):
        # b_weights: (B, R, H)
        torch.bmm(self._out_A, b_weights, out=self._out_B)  # → (B, S, H)
```

**Why `out=self._out_A`?** Without the `out` parameter, `torch.bmm` allocates a new tensor every call. During inference, this means:
- 2 allocations per target_module per layer = 2 × 2 × 12 = 48 CUDA malloc calls per forward pass
- Each allocation requires the CUDA memory allocator to find a free block, which takes ~10μs
- Total: ~0.5ms wasted on memory management per batch

By pre-allocating buffers once at initialization and reusing them via `out=`, we eliminate all 48 allocations. The tensors are written in-place, which is a zero-allocation hot path.

### `model/encoder.py` — The Full Forward Pass

**`AttentionWithLora`** is where the Late Fusion happens:

```python
def forward(self, hidden_states, attention_mask, lora_weights):
    # 1. Base path: standard linear projections (frozen weights)
    projections = {
        "query": self.query(hidden_states),   # W_q @ x
        "key":   self.key(hidden_states),     # W_k @ x
        "value": self.value(hidden_states),   # W_v @ x
    }

    # 2. Delta path: per-tenant LoRA correction
    for module in self.target_modules:       # e.g., ["query", "value"]
        a = torch.cat(lora_weights.a[module])  # (B, H, R)
        b = torch.cat(lora_weights.b[module])  # (B, R, H)
        self.lora_ops.shrink(projections[module], a)  # shrink: x @ A → (B, S, R)
        self.lora_ops.expand(b)                        # expand: (B, S, R) @ B → (B, S, H)

        # 3. Fusion: add delta to base output
        projections[module] = projections[module] + self.lora_ops.output
    
    # 4. Continue with standard multi-head attention...
```

The key insight is that steps 1 and 2 are **completely independent**. The base path doesn't know about LoRA. The delta path doesn't know about the base model. They only meet at step 3 (element-wise addition).

### `ops/head.py` — Task-Specific Head

After the encoder produces a pooled representation `(B, H)`, we need to classify it. Each tenant has a different classifier (different number of classes, different decision boundaries).

```python
class LRHeadOps:
    @staticmethod
    def predict_proba(pooled, coef, intercept, out):
        # pooled:    (B, 1, H)         — encoder output
        # coef:      (B, max_labels, H) — per-tenant LR weights
        # intercept: (B, 1, max_labels) — per-tenant LR bias
        torch.bmm(pooled, coef.transpose(1, 2), out=out)  # (B, 1, max_labels)
        torch.add(out, intercept.unsqueeze(1), out=out)
```

Again, `torch.bmm` enables different classifiers for each item in the batch. Tenant A's 3-class LR head and Tenant B's 25-class LR head are both applied in a single BMM call (zero-padded to `max_labels`).

---

## 9. Memory Analysis

### Base Model Memory
For `intfloat/multilingual-e5-small` (12 layers, H=384):
- Parameters: ~33M
- float32: ~130 MB
- **Loaded once, shared by all tenants**

### Per-Adapter Memory
For `r=8`, targeting `query` and `value` projections:
- Per layer: `2 × (384 × 8) × 4 bytes = 24,576 bytes` ≈ 24 KB
- Per adapter (12 layers): `12 × 24 KB = 288 KB` ≈ 0.3 MB
- **1,000 adapters: ~300 MB**
- **10,000 adapters: ~3 GB**

### Activation Memory (Per Batch)
- Input embeddings: `B × S × H × 4 = 32 × 512 × 384 × 4 = 25 MB`
- LoRA buffers: `2 × B × S × R × 4 = 2 × 32 × 512 × 8 × 4 = 1 MB`
- Attention scores: `B × heads × S × S × 4 = 32 × 6 × 512 × 512 × 4 = 201 MB`

### Total GPU Memory Budget (A100 80GB)
```
Base model:              0.13 GB
5,000 adapters:          1.5  GB
Activation memory:       0.23 GB
PyTorch overhead:        ~2   GB
────────────────────────────────
Total:                   ~4   GB
Available for more:      ~76  GB  (could hold ~250K adapters!)
```

Compare with Approach 1 (one model per tenant):
```
5,000 models × 0.13 GB = 650 GB  → needs 9 A100s
```

---

## 10. Latency Analysis

### Where Time Is Spent

For a batch of 32 items, sequence length 128, on an A100:

| Phase | Time | Notes |
|---|---|---|
| Tokenization | ~2ms | CPU, via HuggingFace tokenizer |
| Batch assembly | ~1ms | CPU, stacking adapter tensors |
| Embedding lookup | ~0.1ms | GPU, table lookup |
| 12× Encoder layers | ~15ms | GPU, attention + FFN + LoRA |
| ├─ Base attention | ~10ms | Standard Q/K/V projections |
| ├─ LoRA BMM (shrink) | ~1ms | 2 BMM calls per layer (query + value) |
| ├─ LoRA BMM (expand) | ~1ms | Same |
| └─ FFN + LayerNorm | ~3ms | Standard feedforward |
| Pooling + LR head | ~0.1ms | GPU, 1 BMM + add |
| **Total** | **~18ms** | **~1,700 samples/sec throughput** |

### Comparison

| Approach | Latency (32 items) | Throughput |
|---|---|---|
| One model per tenant | 32 × 18ms = 576ms (serial) | ~55 samples/sec |
| Weight swapping | 32 × (5ms + 18ms) = 736ms (serial) | ~43 samples/sec |
| **Late Fusion (this)** | **18ms (batched)** | **~1,700 samples/sec** |

The Late Fusion approach is **30–40× faster** than the alternatives because it processes all 32 items in parallel on the GPU, regardless of how many different tenants are in the batch.

---

## 11. Limitations & Future Work

### Current Limitations

1. **Fixed batch size buffers:** `LoraOps` pre-allocates buffers for `max_batch_size × max_seq_len`. If the actual batch is smaller, memory is wasted. A production system would use dynamic buffer sizing or multiple buffer pools.

2. **Same adapter structure:** All adapters must have the same rank and target the same modules. In practice, different tenants might benefit from different ranks (rank 4 for simple tasks, rank 32 for complex ones). Supporting mixed ranks requires padding to the maximum rank.

3. **No adapter eviction:** `AdapterStore` holds all adapters in GPU memory permanently. For 100K+ tenants, you need an LRU cache that evicts cold adapters to CPU/disk and loads them on demand.

4. **CPU batch assembly:** The `BatchAssembler` runs on CPU and involves Python loops + `torch.cat`. For very high throughput (>10K requests/sec), this becomes the bottleneck. A custom CUDA kernel that indexes directly into the adapter store would eliminate this overhead.

5. **Padded classification heads:** The LR head zero-pads all tenants to `max_labels` in the batch. If one tenant has 2 labels and another has 100 labels, the 2-label tenant wastes 98 multiply-accumulate operations.

### Future Directions

- **Paged LoRA:** Analogous to PagedAttention (vLLM), store adapter weights in non-contiguous pages and use a page table for lookup. Eliminates fragmentation for dynamic adapter loading.
- **Fused LoRA kernels:** Merge the shrink + expand + add into a single CUDA kernel to eliminate intermediate tensor writes.
- **Speculative adapter loading:** Use the consistent-hash ring to predict which adapters a pod will need and pre-load them before requests arrive.
- **Rank-adaptive serving:** Allow different adapters to have different ranks, with dynamic padding/masking in the BMM.
