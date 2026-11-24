"""Offline Qwen3.5 Native rollouts with a token-exact policy loss mask."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from pydantic import ValidationError

from agents.native_tool_agent import NativeToolCallingAgent
from agents.side_effects import LocalSideEffectManager
from models.store_memory import MemoryStore
from rl.env import MockArxivEnv
from tools.bootstrap import require_all_tools
from tools.tool_registry import registry
from utils.llm_client import _QWEN_CALL_END, _QWEN_CHAT_END, _decode_qwen_tool_call, parse_tool_response


_TOOL_SLOT = "__NATIVE_GRPO_TOOL_RESULT_4d9d522c__"
_FORGED_ROLE = ("<tool_response", "</tool_response", "<|im_start|>")
_PENDING_CLAIM = re.compile(
    r"(?:翻译|译文)[^。；，！？\n]{0,12}(?<!未)(?<!尚)已(?:经)?(?:翻译|生成|完成|就绪|可读)"
    r"|(?:translation|translated (?:text|pdf))\s+(?:is |has been )?(?:ready|complete|available)\b",
    re.IGNORECASE,
)


def _call_choice(text: str, tools: list[dict], turn: int) -> dict:
    message = {"role": "assistant", "content": text}
    if not any(marker in text for marker in _FORGED_ROLE):
        content, call = _decode_qwen_tool_call(text, tools)
        if call is not None and not text.split(_QWEN_CALL_END, 1)[1].strip():
            call["id"] = f"call_{turn + 1}"
            message.update(content=content, tool_calls=[call])
    return {"message": message}


def native_history(messages: list[dict]) -> list[dict]:
    """Convert recorded assistant turns using exact, unique tool call IDs."""
    by_id = {}
    for message in messages:
        if message["role"] == "tool":
            call_id = message.get("tool_call_id")
            if not call_id or call_id in by_id:
                raise ValueError("missing or duplicate tool response ID")
            by_id[call_id] = message

    history = []
    outstanding_pending = set()
    for message in messages:
        if message["role"] != "assistant":
            continue
        text = message.get("content") or ""
        step = {"thought": text, "observation": ""}
        if message.get("truncated"):
            step["action"] = "FORCE_STOP"
        else:
            try:
                if any(marker in text for marker in _FORGED_ROLE):
                    raise ValueError("assistant_generated_tool_observation")
                text, call, action = parse_tool_response({"message": message})
            except ValueError as exc:
                step.update(action="ERROR", observation=str(exc), parse_failed=True)
            else:
                if message.get("schema_error"):
                    step.update(action="ERROR", observation=message["schema_error"], tool_issue="schema_error")
                elif action is None:
                    step.update(action="FINISH", observation=text, final_answer=text)
                    if outstanding_pending and _PENDING_CLAIM.search(text):
                        step["tool_issue"] = "pending_claimed_ready"
                else:
                    call_id = call["id"]
                    tool = by_id.pop(call_id, None)
                    if tool is None:
                        step.update(action="ERROR", observation="missing tool response", parse_failed=True)
                    else:
                        observation = tool["content"]
                        step.update(action=json.dumps(action, ensure_ascii=False),
                                    observation=observation, tool_call_id=call_id)
                        if tool.get("tool_issue"):
                            step["tool_issue"] = tool["tool_issue"]
                        if "translation_status" in tool:
                            paper_id, status = tool["translation_status"]
                            if status == "PENDING":
                                outstanding_pending.add(paper_id)
                            elif status == "READY":
                                outstanding_pending.discard(paper_id)
        history.append(step)
    if by_id:
        raise ValueError(f"tool response IDs have no matching valid call: {sorted(by_id)}")
    return history


@dataclass
class NativeEpisode:
    """Keep generation, environment state and audit data local to one sample."""

    task_id: str
    agent: NativeToolCallingAgent
    prompt: list[dict]
    setup_receipts: list[dict]
    tools: list[dict]
    prompt_ids: list[int] = field(default_factory=list)
    completion_ids: list[int] = field(default_factory=list)
    mask: list[int] = field(default_factory=list)
    spans: list[tuple[int, int]] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    raw_turns: list[str] = field(default_factory=list)
    stop: str = ""
    audit_alignment: bool = False

    def add_text(self, text: str) -> bool:
        """Validate a single policy turn and append real tool feedback if needed."""
        self.raw_turns.append(text)
        choice = _call_choice(text, self.tools, len(self.raw_turns) - 1)
        message = choice["message"]
        self.messages.append(message)
        try:
            if any(marker in text for marker in _FORGED_ROLE):
                raise ValueError("assistant_generated_tool_observation")
            _, call, action = parse_tool_response(choice)
        except ValueError:
            self.stop = "parse_error"
            return False
        if action is None:
            self.stop = "finish"
            return False
        schema = registry.model_tool(action["name"])
        action["args"] = registry.normalize_arguments(action["name"], action["args"])
        valid, reason = self.agent._validate_routed_action(action, schema) if schema else (False, "unknown_tool")
        if not valid:
            message["schema_error"] = reason
            self.stop = "schema_error"
            return False

        self.agent._last_tool_exception = None
        try:
            observation = self.agent._execute_native_tool(action)
            issue = self.agent._last_tool_issue
        except Exception as exc:
            self.agent._last_tool_exception = exc
            observation, issue = f"{type(exc).__name__}: {exc}", "environment_execution_error"
        response = {"role": "tool", "tool_call_id": call["id"], "content": observation}
        if issue:
            response["tool_issue"] = issue
        exc = self.agent._last_tool_exception
        if issue in {"snapshot_coverage_error", "environment_execution_error"} or (
            exc is not None and (not isinstance(exc, ValueError) or isinstance(exc, ValidationError))
        ):
            self.stop = "environment_invalid"
        result = self.agent._last_tool_result
        if not issue and isinstance(result, dict):
            status = (result.get("status") if action["name"] == "translate_arxiv_pdf"
                      else (result.get("translate") or {}).get("status") if action["name"] == "get_paper_cache_status"
                      else None)
            if status and result.get("paper_id"):
                response["translation_status"] = (result["paper_id"], status)
        self.messages.append(response)
        return not self.stop

    def append_tokens(self, ids: list[int], *, assistant: bool) -> None:
        if assistant:
            self.spans.append((len(self.completion_ids), len(self.completion_ids) + len(ids)))
        self.completion_ids.extend(ids)
        self.mask.extend([int(assistant)] * len(ids))

    def result(self) -> dict:
        invalid = {}
        try:
            history = native_history(self.messages)
        except ValueError as exc:
            if not self.audit_alignment:
                raise
            history = []
            invalid["alignment_error"] = str(exc)
        return {
            **invalid,
            "task_id": self.task_id, "history": history, "iteration_count": len(history),
            "native_messages": [*self.prompt, *self.messages], "setup_receipts": self.setup_receipts,
            "raw_assistant_turns": self.raw_turns, "turn_token_spans": self.spans,
            "tool_mask": self.mask, "completion_ids": self.completion_ids,
            "assistant_tokens": sum(self.mask), "termination_reason": self.stop,
            "clipped": self.stop == "token_budget", "reached_max_turns": self.stop == "max_turns",
            "environment_invalid": self.stop in {"environment_invalid", "setup_failure"},
        }


class NativeRollout:
    """Run one turn at a time against the same snapshots and schemas as Native SFT."""

    def __init__(self, snapshot: Path, *, audit_alignment=False):
        self.audit_alignment = audit_alignment
        require_all_tools("Native GRPO")
        if not snapshot.is_file():
            raise FileNotFoundError(snapshot)
        self.snapshot = json.loads(snapshot.read_text(encoding="utf-8"))

    def episode(self, task: dict, pdf_root: Path) -> NativeEpisode:
        store = MemoryStore()
        env = MockArxivEnv(mode="replay", session_store=store, pdf_root=pdf_root)
        env.snapshot = self.snapshot
        agent = NativeToolCallingAgent(None, env=env, side_effect_mgr=LocalSideEffectManager(store))
        try:
            receipts = agent.apply_setup(task.get("setup") or [], "native_grpo")
        except Exception as exc:
            receipts = [{"tool_issue": "setup_failure", "observation": f"{type(exc).__name__}: {exc}"}]
        agent._conversation = []
        valid = not any(receipt["tool_issue"] for receipt in receipts)
        prompt, options = agent.build_messages(task["task"], "", agent.setup_context(receipts) if valid else "")
        stop = "" if valid else "setup_failure"
        return NativeEpisode(task["id"], agent, prompt, receipts, options["tools"], stop=stop,
                             audit_alignment=self.audit_alignment)

    @staticmethod
    def prompt_ids(tokenizer, episode: NativeEpisode) -> list[int]:
        text = tokenizer.apply_chat_template(
            episode.prompt, tools=episode.tools, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        return tokenizer.encode(text, add_special_tokens=False)

    @staticmethod
    def feedback_ids(tokenizer, episode: NativeEpisode) -> list[int]:
        # Render the official suffix, keeping the previously sampled IDs byte-for-byte.
        # Retemplating the whole history would normalize whitespace and XML arguments
        # and change the context used to score the sampled tokens.
        assistant, tool = episode.messages[-2:]
        call = assistant["tool_calls"][0]
        function = call["function"]
        render_call = {**call, "function": {**function, "arguments": json.loads(function["arguments"])}}
        rendered = tokenizer.apply_chat_template(
            [*episode.prompt, {**assistant, "tool_calls": [render_call]}, {**tool, "content": _TOOL_SLOT}],
            tools=episode.tools, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        _, boundary, suffix = rendered.rpartition(_QWEN_CALL_END)
        if not boundary or not suffix.startswith(_QWEN_CHAT_END) or suffix.count(_TOOL_SLOT) != 1:
            raise RuntimeError("Qwen tool feedback template boundary changed")
        return tokenizer.encode(suffix.replace(_TOOL_SLOT, tool["content"].strip()), add_special_tokens=False)

    def run(
        self, tasks: list[dict], tokenizer, generate: Callable, *,
        max_turns: int, max_completion_length: int, max_new_tokens: int,
    ) -> dict[str, Any]:
        """Run one independent trajectory per input row. TRL expands G upstream."""
        with TemporaryDirectory(prefix="native-grpo-") as temp:
            episodes = [self.episode(task, Path(temp) / str(i)) for i, task in enumerate(tasks)]
            for episode in episodes:
                episode.prompt_ids = self.prompt_ids(tokenizer, episode)
            for turn in range(max_turns):
                batches = defaultdict(list)
                for episode in episodes:
                    if not episode.stop:
                        remaining = max_completion_length - len(episode.completion_ids)
                        batches[min(remaining, max_new_tokens)].append(episode)
                for budget, batch in batches.items():
                    contexts = [e.prompt_ids + e.completion_ids for e in batch]
                    for episode, (ids, eos) in zip(batch, generate(contexts, budget), strict=True):
                        if not ids or len(ids) > budget:
                            raise RuntimeError("invalid native generation token count")
                        episode.append_tokens(ids, assistant=True)
                        text = tokenizer.decode(ids[:-1] if eos else ids, skip_special_tokens=False,
                                                clean_up_tokenization_spaces=False)
                        if not eos and _QWEN_CALL_END not in text and not any(marker in text for marker in _FORGED_ROLE):
                            episode.raw_turns.append(text)
                            episode.messages.append({"role": "assistant", "content": text, "truncated": True})
                            episode.stop = "token_budget"
                        elif episode.add_text(text):
                            if turn + 1 == max_turns:
                                episode.stop = "max_turns"
                            else:
                                feedback = self.feedback_ids(tokenizer, episode)
                                if len(episode.completion_ids) + len(feedback) >= max_completion_length:
                                    episode.stop = "token_budget"
                                else:
                                    episode.append_tokens(feedback, assistant=False)
            return {
                "prompt_ids": [e.prompt_ids for e in episodes],
                "completion_ids": [e.completion_ids for e in episodes],
                "logprobs": None,
                "env_mask": [e.mask for e in episodes],
                "trajectory_results": [e.result() for e in episodes],
            }


def generate_native_turn(model, tokenizer, contexts: list[list[int]], budget: int, temperature: float):
    """Generate left-padded turns and remove padding added after a string stop."""
    import torch
    from transformers import StopStringCriteria

    stops = [_QWEN_CALL_END, "<tool_response", "<|im_start|>"]
    chat_end = tokenizer.convert_tokens_to_ids(_QWEN_CHAT_END)
    eos = model.generation_config.eos_token_id
    eos_ids = list(dict.fromkeys([chat_end, *([eos] if isinstance(eos, int) else eos or [])]))
    width = max(map(len, contexts))
    pad = tokenizer.pad_token_id
    device = next(model.parameters()).device
    input_ids = torch.tensor([[pad] * (width - len(row)) + row for row in contexts], device=device)
    attention_mask = torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in contexts], device=device)
    training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=budget, use_cache=True,
                pad_token_id=pad, eos_token_id=eos_ids, tokenizer=tokenizer, stop_strings=stops,
                do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
            )
    finally:
        model.train(training)
    criterion = StopStringCriteria(tokenizer, stops)
    turns = []
    for row in output[:, width:].tolist():
        eos_at = next((i for i, tok in enumerate(row) if tok in eos_ids), len(row))
        tokens = row[:eos_at]
        # With stop_strings, generate pads rows that finish before the rest of
        # the batch. The Qwen tokenizer's pad can also be a valid EOS.
        if tokens and any(s in tokenizer.decode(tokens, skip_special_tokens=False) for s in stops):
            while tokens and tokens[-1] == pad:
                tokens.pop()
            if not criterion(torch.tensor([tokens]), None).item():
                raise RuntimeError("tokens continued past the native stop boundary")
            turns.append((tokens, False))
        else:
            ended = eos_at < len(row)
            turns.append((row[:eos_at + int(ended)], ended))
    return turns
