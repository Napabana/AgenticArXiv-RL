"""Qwen3.5 text-only loading, LoRA coverage and TRL rollout compatibility tests.

Run from AgenticArxiv:
    python -m unittest tests.test_model_loading -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from rl.grpo_reward import _generate_single_turn_compat
from rl.model_loading import (
    QWEN35_REQUIRED_LORA_SUFFIXES,
    apply_non_thinking_chat_template,
    get_model_type,
    is_qwen35_model,
    resolve_base_model_ref,
    resolve_lora_target_modules,
    trainable_parameter_fingerprint,
    validate_qwen35_lora_targets,
)


class _HybridLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_attn = nn.Module()
        for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
            setattr(self.linear_attn, name, nn.Linear(8, 8, bias=False))
        self.self_attn = nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self.self_attn, name, nn.Linear(8, 8, bias=False))
        self.mlp = nn.Module()
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(self.mlp, name, nn.Linear(8, 8, bias=False))


class _HybridTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(16, 8)
        self.model.layers = nn.ModuleList([_HybridLayer()])
        self.lm_head = nn.Linear(8, 16, bias=False)


class TestQwen35ModelDetection(unittest.TestCase):
    def test_detects_qwen35_from_local_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(
                json.dumps({
                    "model_type": "qwen3_5",
                    "architectures": ["Qwen3_5ForConditionalGeneration"],
                    "vision_config": {"hidden_size": 16},
                }),
                encoding="utf-8",
            )
            self.assertEqual(get_model_type(root), "qwen3_5")
            self.assertTrue(is_qwen35_model(root))

    def test_resolves_base_model_from_adapter_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            adapter = root / "adapter"
            base.mkdir()
            adapter.mkdir()
            (base / "config.json").write_text(
                json.dumps({"model_type": "qwen3_5"}), encoding="utf-8"
            )
            (adapter / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": str(base)}), encoding="utf-8"
            )
            self.assertEqual(resolve_base_model_ref(adapter), str(base))
            self.assertTrue(is_qwen35_model(adapter))


class TestQwen35LoRATargets(unittest.TestCase):
    def test_targets_cover_gated_deltanet_attention_and_mlp(self):
        model = _HybridTextModel()
        targets = resolve_lora_target_modules(model, require_qwen35_coverage=True)
        suffixes = {name.rsplit(".", 1)[-1] for name in targets}
        self.assertTrue(QWEN35_REQUIRED_LORA_SUFFIXES.issubset(suffixes))
        self.assertFalse(any(name.endswith("lm_head") for name in targets))
        self.assertFalse(any("embed_tokens" in name for name in targets))

    def test_missing_hybrid_projection_fails_loudly(self):
        model = _HybridTextModel()
        del model.model.layers[0].linear_attn.in_proj_qkv
        with self.assertRaisesRegex(ValueError, "missed required"):
            resolve_lora_target_modules(model, require_qwen35_coverage=True)

    def test_saved_adapter_target_list_is_validated(self):
        model = _HybridTextModel()
        targets = resolve_lora_target_modules(model, require_qwen35_coverage=True)
        self.assertEqual(validate_qwen35_lora_targets(targets), targets)
        with self.assertRaisesRegex(ValueError, "does not cover"):
            validate_qwen35_lora_targets(["q_proj", "k_proj", "v_proj"])


class TestNonThinkingTemplate(unittest.TestCase):
    def test_passes_enable_thinking_false(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                return "rendered"

        tokenizer = Tokenizer()
        result = apply_non_thinking_chat_template(
            tokenizer, [{"role": "user", "content": "hello"}], tokenize=False
        )
        self.assertEqual(result, "rendered")
        self.assertIs(tokenizer.kwargs["enable_thinking"], False)

    def test_falls_back_for_legacy_template_without_keyword(self):
        class LegacyTokenizer:
            def apply_chat_template(self, messages, tokenize=False):
                return "legacy"

        self.assertEqual(
            apply_non_thinking_chat_template(
                LegacyTokenizer(), [{"role": "user", "content": "hello"}],
                tokenize=False,
            ),
            "legacy",
        )


class TestAdapterFingerprint(unittest.TestCase):
    def test_fingerprint_changes_when_trainable_parameter_changes(self):
        model = nn.Linear(4, 4, bias=False)
        before = trainable_parameter_fingerprint(model)
        with torch.no_grad():
            model.weight.add_(0.25)
        after = trainable_parameter_fingerprint(model)
        self.assertNotEqual(before, after)


class TestTRLGenerationCompatibility(unittest.TestCase):
    def test_current_trl_signature_receives_one_generation_per_expanded_prompt(self):
        class CurrentTrainer:
            def _generate_single_turn(
                self, prompt_ids, images, multimodal_fields, num_generations,
                has_tool_images=False,
            ):
                self.received = (prompt_ids, images, multimodal_fields, num_generations)
                return [[1, 2]], None, None

        trainer = CurrentTrainer()
        result = _generate_single_turn_compat(trainer, [[9, 8]])
        self.assertEqual(trainer.received, ([[9, 8]], None, {}, 1))
        self.assertEqual(result[0], [[1, 2]])

    def test_legacy_trl_signature_does_not_receive_num_generations(self):
        class LegacyTrainer:
            def _generate_single_turn(self, prompt_ids, images, generation_kwargs):
                self.received = (prompt_ids, images, generation_kwargs)
                return [[3]], None, None

        trainer = LegacyTrainer()
        _generate_single_turn_compat(trainer, [[7]])
        self.assertEqual(trainer.received, ([[7]], None, {}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
