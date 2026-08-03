export HOME=/root
export PATH=/root/.local/bin:$PATH
export HF_HOME=/root/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/lora_scaling
uv run python -m lora_serving.benchmark.churn \
  --engine hf --model BAAI/bge-m3 --dtype fp16 --lora-rank 8 --seq-len 128 \
  --batch-size 32 --corpus-dir /root/adapter_corpus --corpus-size 2000 --seeds 1 2 3 \
  --resident 1000 47000 --registration-paths synthetic file --churn-modes blocking \
  --admission-rates 10 --duration 30 --warmup 5 \
  --out /root/host2_registration_paths.csv > /root/host2.log 2>&1
echo HOST2_DONE
