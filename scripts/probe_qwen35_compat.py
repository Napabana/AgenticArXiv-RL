#!/usr/bin/env python3
"""Inspect Qwen3.5 model/config/template compatibility before training.

Default mode downloads only config/tokenizer metadata. Pass --load-model to
load BF16 weights on the selected device and inspect actual module names.
This is a diagnostic utility, not a training script.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _jsonable_config(config):
    text_config = getattr(config, "text_config", None)
    return {
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "is_encoder_decoder": getattr(config, "is_encoder_decoder", None),
        "vision_config_present": getattr(config, "vision_config", None) is not None,
        "text_config_model_type": getattr(text_config, "model_type", None),
        "text_config_architectures": getattr(text_config, "architectures", None),
        "text_layer_types": getattr(text_config, "layer_types", None),
        "tie_word_embeddings": getattr(text_config, "tie_word_embeddings", getattr(config, "tie_word_embeddings", None)),
        "vocab_size": getattr(text_config, "vocab_size", getattr(config, "vocab_size", None)),
        "hidden_size": getattr(text_config, "hidden_size", getattr(config, "hidden_size", None)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--load-model", action="store_true", help="Actually load BF16 weights; requires a supported GPU.")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-memory", default=None, help='Optional device_map max_memory, e.g. "0=20GiB,cpu=48GiB".')
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", default="artifacts/qwen35_compatibility_report.json")
    args = parser.parse_args()

    import torch
    import transformers
    from transformers import AutoConfig, AutoProcessor

    report = {
        "model": args.model,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    report["config"] = _jsonable_config(config)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    tokenizer = getattr(processor, "tokenizer", processor)
    report["processor_class"] = type(processor).__name__
    report["tokenizer_class"] = type(tokenizer).__name__
    report["chat_template_present"] = bool(getattr(tokenizer, "chat_template", None))
    report["pad_token"] = getattr(tokenizer, "pad_token", None)
    report["eos_token"] = getattr(tokenizer, "eos_token", None)

    messages = [{"role": "user", "content": "Return one short ReAct action for searching an arXiv paper."}]
    try:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        report["no_thinking_template"] = "supported"
        report["rendered_prompt_prefix"] = rendered[:500]
    except TypeError as exc:
        report["no_thinking_template"] = f"unsupported: {exc}"
        try:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            report["rendered_prompt_prefix"] = rendered[:500]
        except Exception as inner:
            report["template_error"] = repr(inner)
    except Exception as exc:
        report["template_error"] = repr(exc)

    if args.load_model:
        if not torch.cuda.is_available():
            raise SystemExit("--load-model requires CUDA; run metadata-only mode on CPU.")
        # Prefer the full text+vision checkpoint class supported by current Transformers.
        # The Arxiv task environment itself remains text-only; the vision tower is not used.
        model_cls = None
        import_errors = {}
        for module_name, class_name in (
            # Text-only path first: this arXiv environment never supplies image/video inputs.
            ("transformers", "AutoModelForCausalLM"),
            # Fallbacks are useful to diagnose installations exposing only the multimodal AutoClass.
            ("transformers", "AutoModelForImageTextToText"),
            ("transformers", "AutoModelForMultimodalLM"),
        ):
            try:
                module = __import__(module_name, fromlist=[class_name])
                model_cls = getattr(module, class_name)
                report["loader_class"] = class_name
                break
            except Exception as exc:
                import_errors[class_name] = repr(exc)
        if model_cls is None:
            raise SystemExit(f"No supported multimodal AutoModel class found: {import_errors}")

        kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": args.device_map,
            "trust_remote_code": args.trust_remote_code,
        }
        if args.max_memory:
            limits = {}
            for item in args.max_memory.split(","):
                key, value = item.split("=", 1)
                limits[int(key) if key.isdigit() else key] = value
            kwargs["max_memory"] = limits

        torch.cuda.reset_peak_memory_stats()
        model = model_cls.from_pretrained(args.model, **kwargs)
        report["loaded_model_class"] = type(model).__name__
        report["parameter_count"] = sum(p.numel() for p in model.parameters())
        report["module_name_samples"] = [
            name for name, _ in model.named_modules()
            if any(token in name.lower() for token in (
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                "up_proj", "down_proj", "in_proj", "linear_attn",
            ))
        ][:300]
        counts = Counter(name.rsplit(".", 1)[-1] for name, _ in model.named_modules())
        report["module_leaf_counts"] = dict(counts.most_common(80))
        report["cuda_peak_allocated_gib"] = round(
            torch.cuda.max_memory_allocated() / 1024**3, 3
        )
        report["cuda_peak_reserved_gib"] = round(
            torch.cuda.max_memory_reserved() / 1024**3, 3
        )
        del model
        torch.cuda.empty_cache()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nReport written to: {output}")


if __name__ == "__main__":
    main()
