"""Native-specific replay, reward and policy mask regression cases."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.metrics import is_strict_success
from rl.grpo_reward import make_grpo_reward_fn
from rl.native_grpo_rollout import NativeRollout, native_history
from rl.reward import RewardCalculator
from rl.step_credit import canonical_action_key, compute_batch_step_credit, label_trajectory


SEARCH = "get_recently_submitted_cs_papers"
DOWNLOAD = "download_arxiv_pdf"
CACHE = "get_paper_cache_status"
TRANSLATE = "translate_arxiv_pdf"
PAPER = "2611.18533v1"
FINISH = "已按要求处理。"


def xml(name, **args):
    fields = "".join(f"<parameter={key}>\n{json.dumps(value)}\n</parameter>\n" for key, value in args.items())
    return f"<tool_call>\n<function={name}>\n{fields}</function>\n</tool_call>"


def task(tools, args, *, expected_papers=None, setup=True, **extra):
    seed = [{"name": SEARCH, "args": {"aspect": "CL", "days": 7, "max_results": 1}}]
    return {"id": "case", "task": "任务要求", "setup": seed if setup else [],
            "expected_tools": tools, "expected_tool_args": args,
            "expected_paper_ids": expected_papers, **extra}


@pytest.fixture
def rollout(tmp_path):
    snapshot = tmp_path / "snapshot.json"
    args = {"aspect": "CL", "days": 7, "max_results": 1}
    snapshot.write_text(json.dumps({
        SEARCH: {json.dumps(args, sort_keys=True): {
            "args": args, "result": [{"id": PAPER, "title": "ModHarness-RL", "authors": [],
                                     "abstract": "Replay", "published": "2026-11-22T00:00:00"}],
        }},
    }), encoding="utf-8")
    return NativeRollout(snapshot)


def scripted_history(rollout, root, spec, turns):
    episode = rollout.episode(spec, root)
    assert not episode.stop, episode.setup_receipts
    for turn in turns:
        if not episode.add_text(turn):
            break
    return episode.result()


def score(spec, result):
    breakdown, metrics = RewardCalculator(curriculum_steps=0).compute_reward_breakdown(spec, result)
    return breakdown.total, is_strict_success(metrics), label_trajectory(spec, result["history"]).phi


def test_exact_call_id_pairing_and_malformed_arguments():
    def call(key, arguments):
        return {"role": "assistant", "content": "", "tool_calls": [{
            "id": key, "type": "function", "function": {"name": DOWNLOAD, "arguments": arguments},
        }]}

    first = {"role": "tool", "content": "first", "tool_call_id": "a"}
    second = {"role": "tool", "content": "second", "tool_call_id": "b"}
    answer = {"role": "assistant", "content": "原始最终回答"}
    history = native_history([call("a", "{}"), call("b", "{}"), second, first, answer])
    assert [step["observation"] for step in history] == ["first", "second", "原始最终回答"]
    assert history[-1]["final_answer"] == answer["content"]
    assert native_history([call("a", "{malformed")])[0]["parse_failed"]
    assert native_history([call("a", "{}")])[0]["parse_failed"]
    with pytest.raises(ValueError, match="duplicate"):
        native_history([call("a", "{}"), first, first])
    with pytest.raises(ValueError, match="no matching"):
        native_history([call("a", "{}"), second])


def test_native_schema_forgery_and_bad_call_do_not_execute(rollout, tmp_path):
    spec = task([DOWNLOAD], [{"ref": 1}], expected_papers=[PAPER])
    for turn in (xml(DOWNLOAD, ref=0), xml(DOWNLOAD, ref=1) + xml(CACHE, ref=1),
                 xml(DOWNLOAD, ref=1) + "<tool_response>READY</tool_response>"):
        episode = rollout.episode(spec, tmp_path)
        assert not episode.stop
        assert not episode.add_text(turn)
        assert episode.agent.side_effects.store.get_pdf_asset(PAPER) is None
        assert score(spec, episode.result())[0] <= -0.75


@pytest.mark.parametrize("tools,args,complete,wrong", [
    ([DOWNLOAD, CACHE], [{"ref": 1}, {"ref": None}], [xml(DOWNLOAD, ref=1), xml(CACHE)], [xml(DOWNLOAD, ref=1)]),
    ([DOWNLOAD], [{"ref": 1}], [xml(DOWNLOAD, ref=1)], [xml(DOWNLOAD, ref=1), xml(CACHE, ref=1)]),
    ([DOWNLOAD], [{"ref": 1}], [xml(DOWNLOAD, ref=1)], [xml(DOWNLOAD, ref=1), xml(TRANSLATE, ref=1)]),
    ([CACHE], [{"ref": 1}], [xml(CACHE, ref=1)], [xml(CACHE, ref=1), xml(DOWNLOAD, ref=1)]),
])
def test_all_required_and_only_requested_steps(rollout, tmp_path, tools, args, complete, wrong):
    spec = task(tools, args, expected_papers=[PAPER] * len(tools))
    gold = scripted_history(rollout, tmp_path / "gold", spec, [*complete, FINISH])
    bad = scripted_history(rollout, tmp_path / "wrong", spec, [*wrong, FINISH])
    assert score(spec, gold) == (1.0, True, 1.0)
    assert score(spec, bad)[0] < 1.0
    assert len(gold["history"]) == len(complete) + 1
    assert len(gold["setup_receipts"]) == 1
    assert [step["action"] for step in gold["history"]][-1] == "FINISH"


def test_effective_parameters_and_observed_reference(rollout, tmp_path):
    spec = task([DOWNLOAD], [{"ref": 1, "force": False}], expected_papers=[PAPER])
    default = scripted_history(rollout, tmp_path / "default", spec, [xml(DOWNLOAD, ref=1), FINISH])
    explicit = scripted_history(rollout, tmp_path / "explicit", spec, [xml(DOWNLOAD, ref=1, force=False), FINISH])
    assert canonical_action_key(default["history"][0]) == canonical_action_key(explicit["history"][0])
    assert score(spec, default) == score(spec, explicit) == (1.0, True, 1.0)
    bogus = {**default["history"][0], "observation": '{"paper_id":"2601.00001v1"}'}
    assert label_trajectory(spec, [bogus, default["history"][-1]]).phi == 0
    versioned = task([DOWNLOAD], [{"ref": PAPER}], expected_papers=[PAPER])
    assert score(versioned, default)[2] == 0.0


def test_no_save_is_a_decision(rollout, tmp_path):
    spec = task([SEARCH], [{"aspect": "CL", "days": 7, "max_results": 1, "save_to_file": False}], setup=False)
    def run(save):
        args = dict(aspect="CL", days=7, max_results=1)
        if not save:
            args["save_to_file"] = False
        # XML strings are rendered as bare text, as in the official template.
        return scripted_history(rollout, tmp_path, spec, [xml(SEARCH, **args).replace('"CL"', "CL"), FINISH])
    gold, default = run(False), run(True)
    assert score(spec, gold) == (1.0, True, 1.0)
    assert score(spec, default)[0] < 1.0
    assert score(spec, default)[2] == 0.0
    assert canonical_action_key(gold["history"][0]) != canonical_action_key(default["history"][0])


def test_pending_is_not_ready_and_generic_refusal_fails(rollout, tmp_path):
    spec = task([DOWNLOAD, TRANSLATE], [{"ref": 1}, {"ref": None}], expected_papers=[PAPER, PAPER])
    start = [xml(DOWNLOAD, ref=1), xml(TRANSLATE)]
    pending = scripted_history(rollout, tmp_path / "pending", spec, [*start, "翻译任务已提交，译文尚未就绪。"])
    ready = scripted_history(rollout, tmp_path / "ready", spec, [*start, "论文的译文已就绪。"])
    assert score(spec, pending) == (1.0, True, 1.0)
    assert score(spec, ready)[0] <= -0.75
    assert score(spec, ready)[2] < 1.0
    blocked = task([], [], setup=False, expected_terminal_mode="blocked", expected_terminal_reason="missing_context")
    success = scripted_history(rollout, tmp_path, blocked, ["当前会话没有最近操作的论文，因此无法解析指代。"])
    generic = scripted_history(rollout, tmp_path, blocked, ["任务已完成。"])
    assert score(blocked, success) == (1.0, True, 1.0)
    assert score(blocked, generic)[0] < 1.0


class CharTokenizer:
    """Verify sampled IDs are unchanged, using a deterministic mock template."""

    def apply_chat_template(self, messages, **_):
        if messages[-1]["role"] == "tool":
            return "</tool_call><|im_end|>\n" + messages[-1]["content"] + "\nnext"
        return messages[-1]["content"] + "\nassistant"

    def encode(self, text, **_):
        return list(map(ord, text))

    def decode(self, ids, **_):
        return "".join(map(chr, ids))


def test_pending_setup_anchor_pipeline_with_scripted_turns(tmp_path, monkeypatch):
    # Controls for the CPU integration test only; never enter the frozen policy sample set.
    from rl.credit_v2 import prepare, credits

    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    from run_pending_task_v1 import group_evidence, CANDIDATES

    snapshot = root / "data/mock_arxiv_snapshot.json"
    if not snapshot.is_file():
        pytest.skip("local frozen snapshot unavailable")
    task = next(t for t in json.loads(CANDIDATES.read_bytes())["tasks"] if t["pending_type"] == "B")
    native = NativeRollout(snapshot)
    tok = CharTokenizer()
    native.audit_tokenizer = tok

    def generate(contexts, budget):
        cache = (tok.encode(xml(CACHE, **task["expected_tool_args"][0])), False)
        early = (tok.encode("译文已就绪。\0"), True)
        done = (tok.encode("翻译尚未完成，暂时无法读取译文。\0"), True)
        return [cache, cache, early, early] if len(contexts) == 4 else [done] * len(contexts)

    batch = native.run([task] * 4, tok, generate, max_turns=5, max_completion_length=4000, max_new_tokens=256)
    samples = batch["trajectory_results"]
    rows = credits(prepare(task, samples, native), "E4")
    group = group_evidence(rows, samples)
    assert group["strict"] == [1, 1, 0, 0] and group["valid"] and group["informative"]
    assert group["pending_dfa_pairs"] == group["pending_anchor_pairs"] == group["pending_e4_pairs"] == 4


def test_rollout_context_and_token_mask_use_generated_ids_and_real_feedback(rollout):
    spec = task([DOWNLOAD, CACHE], [{"ref": 1}, {"ref": None}], expected_papers=[PAPER, PAPER])
    script = iter([(xml(DOWNLOAD, ref=1), False), (xml(CACHE), False), (FINISH + "\0", True)])
    tokenizer = CharTokenizer()
    contexts = []

    def generate(rows, _):
        contexts.append(rows[0])
        text, eos = next(script)
        return [(tokenizer.encode(text), eos)]

    batch = rollout.run([spec], tokenizer, generate, max_turns=3, max_completion_length=4000, max_new_tokens=300)
    result, = batch["trajectory_results"]
    assert score(spec, result) == (1.0, True, 1.0)
    assert len(result["history"]) == len(result["turn_token_spans"]) == 3
    tokens, mask = batch["completion_ids"][0], batch["env_mask"][0]
    rebuilt = [0] * len(tokens)
    for i, (start, end) in enumerate(result["turn_token_spans"]):
        rebuilt[start:end] = [1] * (end - start)
        assert contexts[i] == batch["prompt_ids"][0] + tokens[:start]
    assert rebuilt == mask
    assert "pdf_ready" in tokenizer.decode([tok for tok, policy in zip(tokens, mask) if not policy])
    assert "pdf_ready" not in tokenizer.decode([tok for tok, policy in zip(tokens, mask) if policy])


def test_anchor_uses_actual_assistant_ids_and_checks_tool_feedback(rollout, tmp_path):
    from copy import deepcopy
    from rl.anchor_state import fingerprints

    spec = task([CACHE], [{"ref": 1}], expected_papers=[PAPER])
    tokenizer = CharTokenizer()
    rollout.audit_tokenizer = tokenizer
    def sample_call(text):
        scripted = iter([(text, False), (FINISH + "\0", True)])
        def generate(contexts, _budget):
            text, done = next(scripted)
            return [(tokenizer.encode(text), done)]
        batch = rollout.run([spec], tokenizer, generate, max_turns=2, max_completion_length=4000, max_new_tokens=300)
        return batch["trajectory_results"][0]
    sample = sample_call(xml(CACHE, ref=1))
    states = fingerprints(rollout, spec, sample, tmp_path / "original")
    assert states == fingerprints(rollout, spec, sample, tmp_path / "replay")
    end = sample["turn_token_spans"][0][1]
    changed = sample_call("  " + xml(CACHE, ref=1))
    changed_states = fingerprints(rollout, spec, changed, tmp_path / "alternate")
    assert states[0] == changed_states[0] and states[1] != changed_states[1]
    changed = deepcopy(sample)
    changed["completion_ids"][end] += 1
    with pytest.raises(ValueError, match="context tokens"):
        fingerprints(rollout, spec, changed, tmp_path / "corrupt")


def test_truncated_or_environment_invalid_groups(rollout, tmp_path):
    spec = task([CACHE], [{"ref": 1}], expected_papers=[PAPER])
    malformed = [{"role": "assistant", "content": "partials", "truncated": True}]
    cutoff = {"task_id": spec["id"], "history": native_history(malformed), "clipped": True}
    assert cutoff["history"][0]["action"] == "FORCE_STOP"
    correct = scripted_history(rollout, tmp_path, spec, [xml(CACHE, ref=1), FINISH])
    invalid = scripted_history(rollout, tmp_path, spec, [xml(SEARCH, aspect="AI").replace('"AI"', "AI")])
    assert invalid["environment_invalid"]
    key, groups = spec["id"], [correct, invalid, cutoff, correct]
    reward = make_grpo_reward_fn({key: spec}, native_group_size=4)
    rewards = reward(completions=[""] * 4, task_id=[key] * 4, trajectory_results=groups)
    assert rewards == [0.0] * 4
    step_credit, = compute_batch_step_credit([key] * 4, groups, rewards, {key: spec}, 4)
    assert (step_credit.mode, step_credit.category) == ("none", "environment_invalid")


def test_benchmark_cache_completion_uses_ordered_trace_calls(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    from report_native_grpo import metrics

    task_id = "state_dl_then_cache"
    dev = tmp_path / "dev"
    dev.mkdir()
    row = {"task_id": task_id, "trial": 0, "tools": f"{DOWNLOAD},{CACHE}",
           "completed": True, "tool_accurate": True, "arg_applicable": False,
           "ref_applicable": False, "terminal_semantics_applicable": False, "termination": "FINISH"}
    (dev / "summary.json").write_text(json.dumps({"details": [row]}), encoding="utf-8")
    trace = {"task_id": task_id, "trial": 0,
             "history": [{"action": json.dumps({"name": name, "args": {"ref": 1}}),
                          "observation": "{}"} for name in (DOWNLOAD, CACHE)],
             "protocol": {"pending_submission": False, "tool_calls_executed": 2,
                          "unexpected_repeat_calls": 0, "tool_execution_errors": 0, "turns": []}}
    (dev / "traces.jsonl").write_text(json.dumps(trace) + "\n", encoding="utf-8")
    result = metrics(tmp_path, "dev", {(task_id, 0): True})
    assert result["cache", "n"] == result["cache", "d"] == 1


@pytest.mark.parametrize(("micro_batch", "experiment"), [(1,None),(4,None),*( (1,f"E{i}") for i in range(1,5))])
def test_pinned_trl_cpu_step_through_native_rollout(rollout, tmp_path, monkeypatch, micro_batch, experiment):
    """Exercise TRL masking, KL and a LoRA update on a tiny text policy."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from trl import GRPOConfig

    from rl.model_loading import assert_grpo_reference_adapter, trainable_parameter_fingerprint
    from rl.step_credit import StepCreditCollector
    from rl.step_credit_trainer import StepCreditGRPOTrainer

    monkeypatch.setenv("TRL_EXPERIMENTAL_SILENCE", "1")
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = {token: i for i, token in enumerate(["<pad>", "<|im_end|>", *alphabet])}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="<|im_end|>")
    tokenizer.chat_template = (
        '{% if messages[-1].role == "tool" %}{{ "</tool_call><|im_end|>\n" + messages[-1].content + "\nassistant" }}'
        '{% else %}{{ messages[-1].content + "\nassistant" }}{% endif %}'
    )
    base = Qwen3_5ForCausalLM(Qwen3_5TextConfig(
        vocab_size=len(tokenizer), num_hidden_layers=1, hidden_size=24, intermediate_size=48,
        num_attention_heads=2, num_key_value_heads=1, head_dim=12, layer_types=["full_attention"],
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        rope_parameters={"rope_theta": 10000.0, "partial_rotary_factor": 0.5},
    ))
    policy = get_peft_model(base, LoraConfig(r=2, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    spec = task([CACHE], [{"ref": 1}], expected_papers=[PAPER])
    key = spec["id"]
    collector = StepCreditCollector({key: spec}, 4)
    reward = make_grpo_reward_fn({key: spec}, step_credit=collector, native_group_size=4)

    def scripted_rollout(prompts, _trainer):
        def turns(contexts, _budget):
            if len(contexts) == 4:
                call = tokenizer.encode(xml(CACHE, ref=1), add_special_tokens=False)
                return [(call, False)] * 2 + [(answer, True)] * 2
            return [(answer, True)] * len(contexts)

        answer = tokenizer.encode(FINISH + "<|im_end|>", add_special_tokens=False)
        return rollout.run([spec] * len(prompts), tokenizer, turns,
                           max_turns=2, max_completion_length=512, max_new_tokens=256)

    resume_test = experiment == "E4"
    if resume_test:
        from copy import deepcopy
        from transformers import TrainerCallback
        initial_policy = deepcopy(policy)
        class StopAtFirstCheckpoint(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 1:
                    control.should_training_stop = True

    config = GRPOConfig(
        output_dir=str(tmp_path / "train"), use_cpu=True, bf16=False,
        gradient_checkpointing=False, max_steps=2 if resume_test else 1, per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=4 // micro_batch,
        num_generations=4, max_completion_length=512, beta=0.04, learning_rate=5e-6,
        save_strategy="steps" if resume_test else "no", save_steps=1, shuffle_dataset=False,
        report_to="none", logging_steps=1, disable_tqdm=True,
    )
    plan = [{"prompt": key, "task_id": key, "group_id": i} for i in range(config.max_steps)]
    common = dict(model=policy, processing_class=tokenizer, args=config,
                  train_dataset=Dataset.from_list(plan), rollout_func=scripted_rollout)
    if experiment:
        from rl.credit_v2_trainer import StrictGroupReward, CreditV2Trainer
        rollout.audit_tokenizer = tokenizer
        if resume_test:
            common["callbacks"] = [StopAtFirstCheckpoint()]
        strict = StrictGroupReward({key: spec}, rollout, experiment, plan, tmp_path)
        trainer = CreditV2Trainer(**common, credit_reward=strict)
    else:
        trainer = StepCreditGRPOTrainer(**common, reward_funcs=reward, step_credit_collector=collector)
    assert assert_grpo_reference_adapter(trainer.model, beta=config.beta)["reference_frozen"]
    before = trainable_parameter_fingerprint(trainer.model)
    trained = trainer.train()
    assert trained.global_step == 1 and torch.isfinite(torch.tensor(trained.training_loss))
    assert trainable_parameter_fingerprint(trainer.model) != before
    if experiment:
        captured = json.loads((tmp_path / "groups/group_000.json").read_text())
        assert captured["strict_outcomes"] == [1,1,0,0]
        assert all(row["decisions"] for row in captured["rows"])
        if resume_test:
            common.update(model=initial_policy, callbacks=[])
            fresh_reward = StrictGroupReward({key: spec}, rollout, experiment, plan, tmp_path)
            resumed = CreditV2Trainer(**common, credit_reward=fresh_reward)
            assert resumed.train(resume_from_checkpoint=str(tmp_path / "train/checkpoint-1")).global_step == 2
            assert json.loads((tmp_path / "groups/group_000.json").read_text()) == captured
            assert json.loads((tmp_path / "groups/group_001.json").read_text())["strict_outcomes"] == [1,1,0,0]
    else:
        assert trainer.step_credit_counts["reward"] == 1
        assert trainer.step_credit_max_conservation_error < 1e-5
    assert "kl" in trainer.state.log_history[0]
