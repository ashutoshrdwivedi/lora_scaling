set -e
export HOME=/root
export UV_CACHE_DIR=/workspace/uv_cache
export HF_HOME=/root/hf_cache
export PATH=/root/.local/bin:$PATH
curl -LsSf https://astral.sh/uv/install.sh | sh
cd /root/lora_scaling
uv sync --extra dev
uv run python -c "import torch;print(\"torch\",torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))"
echo SETUP_DONE
