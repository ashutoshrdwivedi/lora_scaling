"""Compare custom mixed-tenant DeBERTa-v2 LoRA embeddings against PEFT.

This is the LoRA-path companion to ``compare_deberta_v2_base_embeddings.py``.
It creates four random LoRA adapters to mimic four tenants, applies one tenant
per sentence, and compares:

* PEFT reference: one sentence at a time with ``set_adapter(...)``
* Custom path: one mixed-tenant batch with per-sample LoRA weights

SetFit's LoRA training path wraps its encoder body with PEFT, so PEFT is the
adapter-math reference here. The script does not require SetFit to be installed.

Usage:
    PYTHONPATH=src python benchmarks/quality/compare_deberta_v2_lora_tenants.py \
        --model microsoft/deberta-v2-xlarge
"""

from __future__ import annotations

import argparse

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoModel, AutoTokenizer

from lora_serving.config import LoraServingConfig
from lora_serving.model.deberta_v2 import DebertaV2EncoderWithLora
from lora_serving.weights.batch import LayerwiseBatchedWeights


SENTENCES = [
    "A tenant can have its own adapter while sharing the base model.",
    "The LoRA path adds a low-rank delta to selected projections.",
    "Mixed batches should match a reference loop over individual adapters.",
    "This sentence uses the fourth randomly initialized adapter.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="microsoft/deberta-v2-xlarge", help="HuggingFace DeBERTa-v2 model name")
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank")
    parser.add_argument("--max-length", type=int, default=128, help="Tokenizer max sequence length")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for adapter weights")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for allclose")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance for allclose")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device used for both models",
    )
    return parser.parse_args()


def random_adapter_weights(
    *,
    num_layers: int,
    hidden_size: int,
    rank: int,
    target_modules: list[str],
    seed: int,
    dtype: torch.dtype,
) -> dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    adapters: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    for tenant_idx in range(len(SENTENCES)):
        adapter_name = f"tenant_{tenant_idx}"
        adapters[adapter_name] = {}
        for module_idx, module_name in enumerate(target_modules):
            generator = torch.Generator(device="cpu").manual_seed(seed + tenant_idx * 100 + module_idx)
            a = torch.randn(num_layers, rank, hidden_size, generator=generator, dtype=dtype) * 0.02
            b = torch.randn(num_layers, hidden_size, rank, generator=generator, dtype=dtype) * 0.02
            adapters[adapter_name][module_name] = (a, b)
    return adapters


def install_peft_adapters(peft_model, adapters, target_modules: list[str], lora_config: LoraConfig, device: torch.device) -> None:
    adapter_names = list(adapters)
    for adapter_name in adapter_names[1:]:
        peft_model.add_adapter(adapter_name, lora_config)

    layers = peft_model.base_model.model.encoder.layer
    for adapter_name, by_module in adapters.items():
        for layer_idx, layer in enumerate(layers):
            for module_name in target_modules:
                a, b = by_module[module_name]
                target = getattr(layer.attention.self, module_name)
                target.lora_A[adapter_name].weight.data.copy_(a[layer_idx].to(device))
                target.lora_B[adapter_name].weight.data.copy_(b[layer_idx].to(device))


def build_custom_lora(adapters, target_modules: list[str], num_layers: int, device: torch.device) -> list[LayerwiseBatchedWeights]:
    layers = [LayerwiseBatchedWeights() for _ in range(num_layers)]
    adapter_names = list(adapters)
    for layer_idx in range(num_layers):
        for module_name in target_modules:
            layers[layer_idx].a[module_name] = []
            layers[layer_idx].b[module_name] = []
            for adapter_name in adapter_names:
                a, b = adapters[adapter_name][module_name]
                # PEFT A/B: A=(L,R,H), B=(L,H,R). Custom path: A=(B,H,R), B=(B,R,H).
                layers[layer_idx].a[module_name].append(a[layer_idx].transpose(0, 1).unsqueeze(0).to(device))
                layers[layer_idx].b[module_name].append(b[layer_idx].transpose(0, 1).unsqueeze(0).to(device))
    return layers


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.float32
    target_modules = ["query_proj", "value_proj"]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encoded = tokenizer(
        SENTENCES,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    token_type_ids = encoded.get("token_type_ids")

    hf_config = AutoConfig.from_pretrained(args.model)
    if hf_config.model_type != "deberta-v2":
        raise ValueError(f"Expected a DeBERTa-v2 model, got model_type={hf_config.model_type!r}")

    serving_config = LoraServingConfig(
        model_name=args.model,
        lora_rank=args.rank,
        batch_size=len(SENTENCES),
        max_seq_len=input_ids.shape[1],
        target_modules=target_modules,
        device=device,
        dtype=dtype,
    )

    base_model = AutoModel.from_pretrained(args.model, torch_dtype=dtype).to(device)
    custom_model = DebertaV2EncoderWithLora(hf_config, serving_config).to(device=device, dtype=dtype)
    custom_model.load_state_dict(base_model.state_dict(), strict=True)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    adapters = random_adapter_weights(
        num_layers=serving_config.num_layers,
        hidden_size=serving_config.hidden_size,
        rank=args.rank,
        target_modules=target_modules,
        seed=args.seed,
        dtype=dtype,
    )
    peft_model = get_peft_model(base_model, lora_config, adapter_name=next(iter(adapters)))
    install_peft_adapters(peft_model, adapters, target_modules, lora_config, device)
    custom_lora = build_custom_lora(adapters, target_modules, serving_config.num_layers, device)

    peft_model.eval()
    custom_model.eval()

    peft_outputs = []
    with torch.no_grad():
        for sentence_idx, adapter_name in enumerate(adapters):
            peft_model.set_adapter(adapter_name)
            row = {key: value[sentence_idx : sentence_idx + 1] for key, value in encoded.items()}
            peft_outputs.append(peft_model(**row).last_hidden_state)
        peft_last_hidden = torch.cat(peft_outputs, dim=0)

        custom_embeddings = custom_model.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        custom_last_hidden = custom_model.encoder(
            custom_embeddings,
            attention_mask,
            custom_lora,
            apply_lora=True,
        )

    full_diff = (custom_last_hidden - peft_last_hidden).abs()
    cls_diff = (custom_last_hidden[:, 0] - peft_last_hidden[:, 0]).abs()

    print(f"model: {args.model}")
    print(f"device: {device}")
    print(f"rank: {args.rank}")
    print(f"target_modules: {target_modules}")
    print(f"shape: {tuple(custom_last_hidden.shape)}")
    print(f"full embeddings max_abs_diff: {full_diff.max().item():.8g}")
    print(f"full embeddings mean_abs_diff: {full_diff.mean().item():.8g}")
    print(f"cls embeddings max_abs_diff:  {cls_diff.max().item():.8g}")
    print(f"cls embeddings mean_abs_diff: {cls_diff.mean().item():.8g}")
    print(f"allclose full: {torch.allclose(custom_last_hidden, peft_last_hidden, atol=args.atol, rtol=args.rtol)}")
    print(f"allclose cls:  {torch.allclose(custom_last_hidden[:, 0], peft_last_hidden[:, 0], atol=args.atol, rtol=args.rtol)}")

    for i, (adapter_name, sentence) in enumerate(zip(adapters, SENTENCES)):
        sentence_diff = (custom_last_hidden[i] - peft_last_hidden[i]).abs().max().item()
        cls_sentence_diff = (custom_last_hidden[i, 0] - peft_last_hidden[i, 0]).abs().max().item()
        print(f"[{i}] {adapter_name} token_max_abs_diff={sentence_diff:.8g} cls_max_abs_diff={cls_sentence_diff:.8g} :: {sentence}")


if __name__ == "__main__":
    main()
