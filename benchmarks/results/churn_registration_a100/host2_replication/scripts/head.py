import time, torch
from lora_serving.config import LoraServingConfig
from lora_serving.benchmark.synthetic import make_synthetic_lr_weights

dev = torch.device("cuda:0")
cfg = LoraServingConfig(model_name="BAAI/bge-m3", lora_rank=8, batch_size=8,
                        max_seq_len=128, target_modules=["query", "value"],
                        device=dev, dtype=torch.float16)

def t(fn, n=200):
    fn(); torch.cuda.synchronize()
    s = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - s) / n * 1000

# What the ORIGINAL CSVs paid inside the timed region: on-device randn.
gen = t(lambda: make_synthetic_lr_weights(cfg, 10))
# What the corrected harness pays: H2D copy from pinned host memory.
hc = torch.randn(1, 10, cfg.hidden_size, dtype=cfg.dtype).pin_memory()
hi = torch.zeros(1, 10, dtype=cfg.dtype).pin_memory()
def copy_head():
    hc.to(dev, non_blocking=True); hi.to(dev, non_blocking=True)
cp = t(copy_head)
print("on-device head synthesis (original) %7.4f ms" % gen)
print("pinned head copy         (fixed)    %7.4f ms" % cp)
print("contamination of original data      %7.4f ms" % (gen - cp))
print("as %% of the 11.897 ms file figure   %7.3f %%" % (100*(gen-cp)/11.897))
