import time, torch
from pathlib import Path
from lora_serving.config import LoraServingConfig
from lora_serving.benchmark.churn import corpus_key_fn, PinnedStager
from lora_serving.weights.store import LoraWeight

dev = torch.device("cuda:0")
cfg = LoraServingConfig(model_name="BAAI/bge-m3", lora_rank=8, batch_size=8,
                        max_seq_len=128, target_modules=["query", "value"],
                        device=dev, dtype=torch.float16)
p = Path("/root/smoke_corpus/corpus_000000.bin")

def t(fn, n=20):
    fn(); torch.cuda.synchronize()
    s = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - s) / n * 1000

def load_cpu(): return torch.load(p, map_location="cpu", weights_only=True)
def load_cuda(): return torch.load(p, map_location=dev, weights_only=True)
print("torch.load->cpu       %8.2f ms" % t(load_cpu))
print("torch.load->cuda      %8.2f ms" % t(load_cuda))

sd = load_cpu()
def stack():
    for m in cfg.target_modules:
        torch.stack([sd[corpus_key_fn(i, m)[0]] for i in range(cfg.num_layers)]).transpose(1, 2)
print("stack+transpose(cpu)  %8.2f ms" % t(stack))

st = PinnedStager(cfg)
def into_pinned():
    for m in cfg.target_modules:
        st.a[m].copy_(torch.stack([sd[corpus_key_fn(i, m)[0]] for i in range(cfg.num_layers)]).transpose(1, 2))
print("  -> into pinned buf  %8.2f ms" % t(into_pinned))

w = LoraWeight(cfg)
def h2d():
    for m in cfg.target_modules:
        w.wa[m].copy_(st.a[m], non_blocking=True)
        w.wb[m].copy_(st.b[m], non_blocking=True)
print("H2D from pinned       %8.2f ms" % t(h2d))
print("LoraWeight alloc      %8.2f ms" % t(lambda: LoraWeight(cfg)))
hc = torch.randn(1, 10, cfg.hidden_size, dtype=cfg.dtype).pin_memory()
print("head .to(dev)         %8.2f ms" % t(lambda: hc.to(dev, non_blocking=True)))
