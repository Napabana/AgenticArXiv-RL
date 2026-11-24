"""Pinned format-only generation protocol shared by probe, training and evaluation."""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from .task_generator import prompt
from .validator import score

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "reports/format_control"
DATA = ROOT / "data/format_control"
TOKENS = 128
V1 = "789af8cf28960892a1fde034d32f9b8ddf230e8450552be70d15024c964ed6c8"


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def save(path, value):
    with path.open("x", encoding="utf8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def split_rows(split):
    path = DATA / (split + ".jsonl")
    if digest(path) != read(DATA / "manifest.json")["splits"][split]["sha256"]:
        raise ValueError(f"Changed {split} split")
    return [json.loads(x) for x in path.read_text(encoding="utf8").splitlines()]


def train_prompt_sha():
    prompts = [prompt(row) for row in split_rows("train")]
    return hashlib.sha256(json.dumps(prompts, ensure_ascii=False).encode()).hexdigest()


def verify_assets(base, adapter, adapter_sha):
    origin = ROOT / "reports/native_sft_final_20261121"
    for line in (origin / "base_model.sha256").read_text().splitlines():
        wanted, name = line.split()
        if digest(base / name) != wanted:
            raise ValueError(f"Base identity mismatch: {name}")
    if digest(adapter / "adapter_model.safetensors") != adapter_sha:
        raise ValueError("Adapter weights mismatch")
    if adapter_sha == V1 and digest(adapter / "adapter_config.json") != read(origin / "training.json")["adapter_config_sha256"]:
        raise ValueError("Native SFT v1 adapter config mismatch")


def load(base, adapter, *, train=False, adapter_sha=V1):
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    from rl.model_loading import load_text_causal_lm, validate_qwen35_lora_targets

    verify_assets(base, adapter, adapter_sha)
    for line in (ROOT / "requirements-qwen35.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            package, wanted = line.split("==")
            if version(package) != wanted:
                raise RuntimeError(f"{package} must be {wanted}")
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("One visible BF16 CUDA device is required")
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, padding_side="left")
    template_hash = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
    if template_hash != read(ROOT / "data/splits/native_grpo_qwen35_train_v2.json")["chat_template_sha256"]:
        raise ValueError("Chat template changed")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_text_causal_lm(base, dtype=torch.bfloat16, local_files_only=True).to("cuda")
    policy = PeftModel.from_pretrained(model, adapter, is_trainable=train)
    validate_qwen35_lora_targets(policy.peft_config["default"].target_modules)
    policy.generation_config.eos_token_id = ends(tokenizer)
    if not train:
        policy.eval()
        if any(p.requires_grad for p in policy.parameters()):
            raise RuntimeError("Sampling policy must be frozen")
    return policy, tokenizer


def ends(tokenizer):
    return list(dict.fromkeys([tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]))


def rendered(tokenizer, task):
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt(task)}],
                                         add_generation_prompt=True, enable_thinking=False, tokenize=False)


def observation(tokenizer, ids, task):
    stop = set(ends(tokenizer))
    n = next((i for i, token in enumerate(ids) if token in stop), len(ids))
    text = tokenizer.decode(ids[:n], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    diagnostics = score(text, task)
    if diagnostics["strict"] and n == len(ids):
        raise RuntimeError("Valid-looking completion reached its length limit without EOS")
    return dict(completion=text, tokens=min(len(ids), n + 1), ended=n < len(ids), **diagnostics)


def sample(model, tokenizer, task, seed, count):
    import torch
    torch.manual_seed(seed)
    inputs = tokenizer(rendered(tokenizer, task), return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=True, temperature=1.0, top_p=1., top_k=0,
                                   num_return_sequences=count, max_new_tokens=TOKENS, use_cache=True,
                                   eos_token_id=ends(tokenizer), pad_token_id=tokenizer.pad_token_id)
    return [dict(seed=seed, draw=i, **observation(tokenizer, seq.tolist(), task))
            for i, seq in enumerate(generated[:, inputs.input_ids.shape[1]:])]


def arguments(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True, help="Frozen v1 adapter directory")
    parser.add_argument("--output", type=Path, required=True)
    return parser
