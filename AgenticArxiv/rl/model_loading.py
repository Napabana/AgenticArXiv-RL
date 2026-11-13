"""Model loading helpers for text-only Agent training and evaluation.

Qwen3.5 checkpoints are multimodal at the top-level config, but ArxivFetch
uses text-only tools. For Qwen3.5 we therefore load Qwen3_5ForCausalLM rather
than the vision-language ConditionalGeneration class.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


QWEN35_MODEL_TYPES = {"qwen3_5", "qwen3_5_text"}
# Frozen Hugging Face model-repository commit used for reproducible experiments.
QWEN35_DEFAULT_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
QWEN35_REQUIRED_LORA_SUFFIXES = {
    # Gated DeltaNet (linear-attention) projections.
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    # Full-attention and MLP projections.
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def is_adapter_directory(path: str | Path) -> bool:
    candidate = Path(path).expanduser()
    return candidate.is_dir() and (candidate / "adapter_config.json").is_file()


def resolve_base_model_ref(
    model_ref: str | Path,
    *,
    base_model: str | Path | None = None,
) -> str:
    """Resolve a base checkpoint from a model ID, model directory, or PEFT adapter."""
    if base_model:
        return str(base_model)

    candidate = Path(model_ref).expanduser()
    adapter_config_path = candidate / "adapter_config.json"
    if adapter_config_path.is_file():
        config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        base_ref = config.get("base_model_name_or_path")
        if not base_ref:
            raise ValueError(
                f"Adapter has no base_model_name_or_path: {adapter_config_path}"
            )
        return str(base_ref)
    return str(model_ref)


def get_model_type(
    model_ref: str | Path,
    *,
    base_model: str | Path | None = None,
    revision: str | None = None,
) -> str:
    """Read model_type locally when possible; otherwise resolve via AutoConfig."""
    base_ref = resolve_base_model_ref(model_ref, base_model=base_model)
    config_path = Path(base_ref).expanduser() / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        return str(config.get("model_type") or "")

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        base_ref,
        revision=revision,
        trust_remote_code=False,
        local_files_only=Path(base_ref).expanduser().exists(),
    )
    return str(getattr(config, "model_type", "") or "")


def is_qwen35_model(
    model_ref: str | Path,
    *,
    base_model: str | Path | None = None,
    revision: str | None = None,
) -> bool:
    return get_model_type(
        model_ref, base_model=base_model, revision=revision
    ) in QWEN35_MODEL_TYPES


def resolve_model_revision(
    model_ref: str | Path,
    *,
    revision: str | None = None,
    base_model: str | Path | None = None,
) -> str | None:
    """Resolve an explicit revision, saved adapter metadata, or Qwen3.5 pin."""
    if revision:
        return revision

    candidate = Path(model_ref).expanduser()
    if is_adapter_directory(candidate):
        for metadata_name in ("adapter_manifest.json", "training_manifest.json"):
            metadata_path = candidate / metadata_name
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                saved_revision = metadata.get("model_revision") or metadata.get("base_revision")
                if saved_revision:
                    return str(saved_revision)
        adapter_config = json.loads(
            (candidate / "adapter_config.json").read_text(encoding="utf-8")
        )
        saved_revision = adapter_config.get("revision")
        if saved_revision:
            return str(saved_revision)

    base_ref = resolve_base_model_ref(model_ref, base_model=base_model)
    if Path(base_ref).expanduser().exists():
        return None
    if get_model_type(base_ref, revision=revision) in QWEN35_MODEL_TYPES:
        return QWEN35_DEFAULT_REVISION
    return None


def load_text_causal_lm(
    model_ref: str | Path,
    *,
    base_model: str | Path | None = None,
    revision: str | None = None,
    **from_pretrained_kwargs: Any,
):
    """Load a text-only causal LM, selecting Qwen3.5's explicit text model class."""
    base_ref = resolve_base_model_ref(model_ref, base_model=base_model)
    resolved_revision = resolve_model_revision(
        base_ref, revision=revision, base_model=base_model
    )
    model_type = get_model_type(base_ref, revision=resolved_revision)
    if resolved_revision:
        from_pretrained_kwargs.setdefault("revision", resolved_revision)

    if model_type in QWEN35_MODEL_TYPES:
        try:
            from transformers import Qwen3_5ForCausalLM
        except ImportError as exc:
            raise ImportError(
                "Qwen3.5 text-only loading requires a Transformers v5 release "
                "that exports Qwen3_5ForCausalLM. Install the pinned "
                "requirements-qwen35.txt environment."
            ) from exc
        model_cls = Qwen3_5ForCausalLM
    else:
        from transformers import AutoModelForCausalLM

        model_cls = AutoModelForCausalLM

    return model_cls.from_pretrained(base_ref, **from_pretrained_kwargs)


def resolve_lora_target_modules(
    model,
    *,
    require_qwen35_coverage: bool = False,
) -> list[str]:
    """Select all text-backbone Linear modules, excluding embeddings and lm_head.

    For Qwen3.5 this must include both Gated DeltaNet and full-attention paths;
    the legacy Qwen2-style fixed target list misses the Gated DeltaNet projections.
    """
    import torch.nn as nn

    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf == "lm_head" or "embed_tokens" in name:
            continue
        if any(part in name.lower() for part in (".visual.", ".vision.", "vision_tower")):
            continue
        targets.append(name)

    targets = sorted(set(targets))
    if not targets:
        raise ValueError("No eligible nn.Linear modules found for LoRA")

    if require_qwen35_coverage:
        found_suffixes = {name.rsplit(".", 1)[-1] for name in targets}
        missing = sorted(QWEN35_REQUIRED_LORA_SUFFIXES - found_suffixes)
        if missing:
            raise ValueError(
                "Qwen3.5 LoRA target scan missed required hybrid-attention/MLP "
                f"projections: {missing}. Found suffixes: {sorted(found_suffixes)}"
            )
    return targets



def apply_non_thinking_chat_template(tokenizer, messages, **kwargs):
    """Render the project's ReAct messages without a model-generated thinking block."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError as exc:
        # Older tokenizer templates may not expose enable_thinking. Preserve
        # compatibility for legacy baselines; Qwen3.5 uses the explicit path.
        if "enable_thinking" not in str(exc):
            raise
        return tokenizer.apply_chat_template(messages, **kwargs)




def assert_default_template_is_non_thinking(tokenizer) -> None:
    """Ensure a model's default SFT template matches explicit no-thinking mode."""
    sample = [{"role": "user", "content": "Return one short ReAct action."}]
    default = tokenizer.apply_chat_template(
        sample,
        tokenize=False,
        add_generation_prompt=True,
    )
    try:
        explicit = tokenizer.apply_chat_template(
            sample,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError as exc:
        raise ValueError(
            "Tokenizer does not accept enable_thinking=False"
        ) from exc
    if default != explicit:
        raise ValueError(
            "The tokenizer default chat template differs from enable_thinking=False. "
            "This TRL SFTConfig version cannot forward chat_template_kwargs; "
            "use a tokenizer/template that defaults to non-thinking or add explicit "
            "SFT preprocessing before training."
        )


def validate_qwen35_lora_targets(target_modules) -> list[str]:
    """Fail if a saved Qwen3.5 adapter omits any hybrid-attention/MLP path."""
    if isinstance(target_modules, str):
        targets = [target_modules]
    else:
        targets = [str(name) for name in (target_modules or [])]
    suffixes = {name.rsplit(".", 1)[-1] for name in targets}
    missing = sorted(QWEN35_REQUIRED_LORA_SUFFIXES - suffixes)
    if missing:
        raise ValueError(
            "Saved Qwen3.5 LoRA adapter does not cover all hybrid-attention/MLP "
            f"projections; missing={missing}"
        )
    return targets


def trainable_parameter_fingerprint(model) -> str:
    """Hash trainable tensors to verify an optimizer step changed the adapter."""
    import hashlib
    import torch

    digest = hashlib.sha256()
    seen = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        seen += 1
        digest.update(name.encode("utf-8"))
        tensor = parameter.detach().to(device="cpu", dtype=torch.float32).contiguous()
        digest.update(tensor.numpy().tobytes())
    if not seen:
        raise ValueError("Model has no trainable parameters to fingerprint")
    return digest.hexdigest()


def build_bf16_lora_config(
    model,
    *,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    require_qwen35_coverage: bool = False,
):
    from peft import LoraConfig

    targets = resolve_lora_target_modules(
        model, require_qwen35_coverage=require_qwen35_coverage
    )
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    return config, targets


def load_text_model_and_tokenizer(
    model_ref: str | Path,
    *,
    adapter_path: str | Path | None = None,
    base_model: str | Path | None = None,
    revision: str | None = None,
    **from_pretrained_kwargs: Any,
):
    """Load a text model plus tokenizer, optionally attaching a saved PEFT adapter."""
    from transformers import AutoTokenizer

    model_path = Path(model_ref).expanduser()
    resolved_adapter = str(adapter_path) if adapter_path else None
    if is_adapter_directory(model_path):
        resolved_adapter = str(model_path)
    resolved_base = resolve_base_model_ref(
        model_ref,
        base_model=base_model,
    ) if resolved_adapter == str(model_path) else (
        str(base_model) if base_model else str(model_ref)
    )

    resolved_revision = resolve_model_revision(
        model_ref,
        revision=revision,
        base_model=base_model,
    )
    tokenizer_ref = (
        resolved_adapter
        if resolved_adapter and (Path(resolved_adapter) / "tokenizer_config.json").is_file()
        else resolved_base
    )
    tokenizer_kwargs = {
        "local_files_only": Path(tokenizer_ref).expanduser().exists(),
        "trust_remote_code": False,
    }
    if resolved_revision:
        tokenizer_kwargs["revision"] = resolved_revision
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_text_causal_lm(
        resolved_base,
        base_model=base_model,
        revision=resolved_revision,
        **from_pretrained_kwargs,
    )
    if resolved_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, resolved_adapter)
    return model, tokenizer
