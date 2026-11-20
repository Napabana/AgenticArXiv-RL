import json

import pytest

from agents.native_tool_agent import NativeToolCallingAgent
from agents.side_effects import LocalSideEffectManager
from benchmark.metrics import is_strict_success
from benchmark.protocol_audit import summarize_protocol_results
from benchmark.runner import BenchmarkRunner
from tools.bootstrap import require_all_tools
from tools.tool_registry import registry
from utils.llm_client import parse_tool_response


def _reply(content=None, calls=None, reason="stop"):
    return {
        "model": "fixture-model",
        "choices": [{"finish_reason": reason, "message": {"content": content, "tool_calls": calls}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }


def _call(call_id, name, args):
    return {
        "id": call_id, "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


class _ScriptedClient:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def chat_completions(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.responses)


class _RecordingEnv:
    def __init__(self):
        self.calls = []

    def execute_tool(self, name, args):
        self.calls.append((name, args))
        return f"result for {args['query']}"


@pytest.fixture
def lookup(monkeypatch):
    require_all_tools("native tool test")
    monkeypatch.setattr(registry, "_tools", registry._tools.copy())
    registry.register_tool(
        "lookup", "lookup a paper",
        {
            "type": "object",
            "properties": {"query": {"type": "string"}, "session_id": {"type": "string"}},
            "required": ["query"],
        },
        lambda **_: pytest.fail("The tool should run through the injected environment"),
    )


def test_both_agents_run_and_score_two_tools_with_the_same_benchmark(lookup, monkeypatch):
    first_call = _call("one", "lookup", {"query": "first"})
    second_call = _call("two", "lookup", {"query": "second"})
    client = _ScriptedClient(
        _reply('Thought: first\nAction: {"name":"lookup","args":{"query":"first"}}'),
        _reply('Thought: second\nAction: {"name":"lookup","args":{"query":"second"}}'),
        _reply('Thought: done\n  Action: ```json\n{"name":"FINISH"}\n```'),
        _reply(calls=[first_call], reason="tool_calls"),
        _reply(calls=[second_call], reason="tool_calls"),
        _reply("已完成两次检索。"),
    )
    env = _RecordingEnv()
    side_effects = LocalSideEffectManager()
    runner = BenchmarkRunner(agent_types=["regex", "native_tool_calling"], repeat=1)
    runner._llm_client = client
    monkeypatch.setattr(runner, "_tool_env", lambda: env)
    monkeypatch.setattr(runner, "_side_effects", lambda: side_effects)
    results = runner.run_all([{
        "id": "two_lookups", "task": "查询两篇论文", "expected_tools": ["lookup", "lookup"],
        "expected_tool_args": [{"query": "first"}, {"query": "second"}],
    }])
    assert all(result.metrics and is_strict_success(result.metrics) for result in results)
    assert results[0].raw_result["reward"] == results[1].raw_result["reward"]
    assert results[1].raw_result["final_observation"] == "已完成两次检索。"
    assert results[1].metrics.total_tokens == 36
    assert [args["query"] for _, args in env.calls] == ["first", "second", "first", "second"]
    assert all(args["session_id"] == result.session_id
               for result, calls in zip(results, (env.calls[:2], env.calls[2:]))
               for _, args in calls)
    assert client.calls[0]["extra"]["stop"] == ["Observation:"]
    assert client.calls[3]["extra"]["tool_choice"] == "auto"
    assert "stop" not in client.calls[3]["extra"]
    assert len(client.calls[3]["messages"]) == 2
    native_history = client.calls[5]["messages"][2:]
    assert native_history[0]["tool_calls"] == [first_call]
    assert native_history[1] == {"role": "tool", "tool_call_id": "one", "content": "result for first"}
    assert native_history[2]["tool_calls"] == [second_call]
    assert native_history[3] == {"role": "tool", "tool_call_id": "two", "content": "result for second"}
    report = summarize_protocol_results(results)
    assert report["paired_strict"] == {"total": 1, "fail_to_pass": 0, "pass_to_fail": 0}
    assert report["agents"]["native_tool_calling"]["schema_compliance"]["count"] == 2


@pytest.mark.parametrize(("response", "error"), [
    (_reply("<tool_call>"), "missing_structured_tool_call"),
    (_reply(reason="tool_calls"), "missing_structured_tool_call"),
    (_reply(calls=[_call("1", "lookup", [])]), "arguments_not_object"),
    (_reply(calls=[_call("1", "lookup", {}), _call("2", "lookup", {})]), "multiple_tool_calls"),
    (_reply("partial response", reason="length"), "finish_reason:length"),
    (_reply(""), "empty_assistant_response"),
])
def test_api_parser_rejects_incomplete_or_ambiguous_turns(response, error):
    with pytest.raises(ValueError, match=error):
        parse_tool_response(response["choices"][0])


@pytest.mark.parametrize(("name", "args", "error"), [
    ("lookup", {"query": 42}, "invalid_value:query"),
    ("unregistered_tool", {}, "未知工具"),
])
def test_invalid_call_does_not_dispatch_and_is_not_finish(lookup, name, args, error):
    env = _RecordingEnv()
    client = _ScriptedClient(_reply(calls=[_call("one", name, args)]))
    result = NativeToolCallingAgent(client, env=env).run("查询")
    assert not env.calls
    assert result["history"][-1]["action"] == "ERROR"
    assert error in result["final_observation"]
    assert result["token_usage"]["total_tokens"] == 12


def test_comparison_reclassifies_legacy_false_finish_only_in_its_scoring_trace():
    runner = BenchmarkRunner(agent_types=["regex", "native_tool_calling"], repeat=1)
    runner._llm_client = _ScriptedClient(_reply("没有动作"), _reply("缺少工具能力，无法处理。"))
    results = runner.run_all([{"id": "noop", "task": "不调用工具", "expected_tools": []}])
    react, native = results
    assert react.raw_result["runtime_terminal_step"]["action"] == "FINISH"
    assert react.metrics.termination_type == "ERROR"
    assert react.metrics.parse_failures == 1
    assert native.metrics.termination_type == "FINISH"
    assert summarize_protocol_results(results)["paired_strict"]["fail_to_pass"] == 1


def test_execution_failure_is_returned_to_the_model(lookup):
    class BrokenEnv:
        def execute_tool(self, name, args):
            raise ValueError("fixture unavailable")

    client = _ScriptedClient(
        _reply(calls=[_call("one", "lookup", {"query": "paper"})]),
        _reply("无法查询，数据暂时不可用。"),
    )
    result = NativeToolCallingAgent(client, env=BrokenEnv()).run("查询")
    assert result["history"][1]["action"] == "FINISH"
    tool_feedback = client.calls[1]["messages"][-1]
    assert tool_feedback["tool_call_id"] == "one"
    assert "fixture unavailable" in tool_feedback["content"]
