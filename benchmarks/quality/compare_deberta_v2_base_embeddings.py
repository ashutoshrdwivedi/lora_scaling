"""Compare HuggingFace DeBERTa-v2 embeddings with the custom base encoder.

This intentionally disables the LoRA path. It checks whether the DeBERTa-v2
architecture we implemented matches HuggingFace for the same base weights.

Usage:
    PYTHONPATH=src python benchmarks/quality/compare_deberta_v2_base_embeddings.py \
        --model microsoft/deberta-v2-xlarge
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from lora_serving.config import LoraServingConfig
from lora_serving.model.deberta_v2 import DebertaV2EncoderWithLora
from lora_serving.weights.batch import LayerwiseBatchedWeights


SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "DeBERTa separates content and relative position information inside attention.",
    "Small numerical differences can appear when kernels or dtypes differ.",
    "A correct base encoder should match HuggingFace before adding adapters.",
    "This batch contains five sentences with different lengths.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="microsoft/deberta-v2-xlarge", help="HuggingFace DeBERTa-v2 model name")
    parser.add_argument("--max-length", type=int, default=128, help="Tokenizer max sequence length")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance for allclose")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance for allclose")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device used for both models",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.float32

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
        lora_rank=1,
        batch_size=len(SENTENCES),
        max_seq_len=input_ids.shape[1],
        target_modules=[],
        device=device,
        dtype=dtype,
    )

    hf_model = AutoModel.from_pretrained(args.model, torch_dtype=dtype).to(device)
    custom_model = DebertaV2EncoderWithLora(hf_config, serving_config).to(device=device, dtype=dtype)
    custom_model.load_state_dict(hf_model.state_dict(), strict=True)
    hf_model.eval()
    custom_model.eval()

    empty_lora = [LayerwiseBatchedWeights() for _ in range(serving_config.num_layers)]

    with torch.no_grad():
        hf_last_hidden = hf_model(**encoded).last_hidden_state
        custom_embeddings = custom_model.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        custom_last_hidden = custom_model.encoder(
            custom_embeddings,
            attention_mask,
            empty_lora,
            apply_lora=False,
        )

    full_diff = (custom_last_hidden - hf_last_hidden).abs()
    cls_diff = (custom_last_hidden[:, 0] - hf_last_hidden[:, 0]).abs()

    print(f"model: {args.model}")
    print(f"device: {device}")
    print(f"shape: {tuple(custom_last_hidden.shape)}")
    print(f"full embeddings max_abs_diff: {full_diff.max().item():.8g}")
    print(f"full embeddings mean_abs_diff: {full_diff.mean().item():.8g}")
    print(f"cls embeddings max_abs_diff:  {cls_diff.max().item():.8g}")
    print(f"cls embeddings mean_abs_diff: {cls_diff.mean().item():.8g}")
    print(f"allclose full: {torch.allclose(custom_last_hidden, hf_last_hidden, atol=args.atol, rtol=args.rtol)}")
    print(f"allclose cls:  {torch.allclose(custom_last_hidden[:, 0], hf_last_hidden[:, 0], atol=args.atol, rtol=args.rtol)}")

    for i, sentence in enumerate(SENTENCES):
        sentence_diff = (custom_last_hidden[i] - hf_last_hidden[i]).abs().max().item()
        cls_sentence_diff = (custom_last_hidden[i, 0] - hf_last_hidden[i, 0]).abs().max().item()
        print(f"[{i}] token_max_abs_diff={sentence_diff:.8g} cls_max_abs_diff={cls_sentence_diff:.8g} :: {sentence}")


if __name__ == "__main__":
    main()
