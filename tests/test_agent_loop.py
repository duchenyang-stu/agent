import contextlib
import io
import json
import unittest
from unittest.mock import patch

from agent.agent_loop import run
from agent.providers import ProviderError
from agent.tools import ToolRegistry


class FakeProvider:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append({"messages": list(messages), "tools": tools})
        return next(self.responses)


class AgentLoopTest(unittest.TestCase):
    def run_quietly(self, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return run(*args, **kwargs)

    def test_registered_tool_is_exposed_and_executed_without_loop_changes(self):
        registry = ToolRegistry()

        @registry.tool(
            description="Double an integer.",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
        )
        def double(value):
            return value * 2

        provider = FakeProvider([
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_ok", "name": "double", "arguments": {"value": 4}},
                    {"id": "call_error", "name": "double", "arguments": {}},
                ],
            },
            {"role": "assistant", "content": "The first call returned 8."},
        ])

        result = self.run_quietly("double 4", provider=provider, registry=registry)

        self.assertTrue(result.ok)
        self.assertEqual(result.stop_reason, "final_answer")
        self.assertEqual(result.answer, "The first call returned 8.")
        self.assertEqual(result.steps, 2)
        self.assertEqual(provider.requests[0]["tools"], registry.schemas())
        tool_messages = provider.requests[1]["messages"][-2:]
        self.assertEqual(tool_messages[0]["tool_call_id"], "call_ok")
        self.assertEqual(json.loads(tool_messages[0]["content"]), {"ok": True, "result": 8})
        self.assertEqual(tool_messages[1]["tool_call_id"], "call_error")
        failure = json.loads(tool_messages[1]["content"])
        self.assertFalse(failure["ok"])
        self.assertEqual(failure["error"]["type"], "invalid_arguments")

    def test_normalized_tool_call_stays_internal_across_rounds(self):
        registry = ToolRegistry()

        @registry.tool(
            description="Calculate a test expression.",
            parameters={
                "type": "object",
                "properties": {"expr": {"type": "string"}},
                "required": ["expr"],
            },
        )
        def calc(expr):
            return 5

        provider = FakeProvider([
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "name": "calc", "arguments": {"expr": "2+3"}},
                ],
            },
            {"role": "assistant", "content": "The result is 5."},
        ])

        result = self.run_quietly(
            "calculate",
            provider=provider,
            registry=registry,
        )

        self.assertEqual(result.answer, "The result is 5.")
        second_messages = provider.requests[1]["messages"]
        self.assertEqual(second_messages[2]["tool_calls"][0]["name"], "calc")
        self.assertNotIn("function", second_messages[2]["tool_calls"][0])
        self.assertEqual(second_messages[3]["role"], "tool")
        self.assertEqual(second_messages[3]["tool_call_id"], "call_1")
        self.assertEqual(json.loads(second_messages[3]["content"]), {"ok": True, "result": 5})

    def test_tool_error_is_returned_to_model(self):
        registry = ToolRegistry()

        @registry.tool(
            description="Calculate a test expression.",
            parameters={
                "type": "object",
                "properties": {"expr": {"type": "string"}},
                "required": ["expr"],
            },
        )
        def calc(expr):
            raise ValueError("unsupported expression")

        provider = FakeProvider([
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "name": "calc", "arguments": {"expr": "sqrt(9)"}},
                ],
            },
            {"role": "assistant", "content": "I could not use that expression."},
        ])

        result = self.run_quietly("calculate", provider=provider, registry=registry)

        self.assertEqual(result.answer, "I could not use that expression.")
        self.assertEqual(result.tool_errors, 1)
        tool_result = json.loads(provider.requests[1]["messages"][-1]["content"])
        self.assertFalse(tool_result["ok"])
        self.assertEqual(tool_result["error"]["type"], "tool_execution_error")
        self.assertIn("unsupported expression", tool_result["error"]["message"])

    def test_max_steps_has_a_clear_stop_message(self):
        registry = ToolRegistry()

        @registry.tool(
            description="Do nothing.",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        def noop():
            return None

        provider = FakeProvider([
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "name": "noop", "arguments": {}},
                ],
            },
        ])

        with patch("agent.agent_loop.config.MAX_STEPS", 1):
            result = self.run_quietly(
                "keep going",
                provider=provider,
                registry=registry,
            )

        self.assertEqual(result.stop_reason, "max_steps")
        self.assertEqual(result.steps, 1)
        self.assertEqual(str(result), "Stopped after MAX_STEPS=1.")

    def test_retryable_api_error_recovers_with_exponential_backoff(self):
        provider = FakeProvider([])
        responses = iter([
            ProviderError("http_error", "rate limited", retryable=True, status_code=429),
            ProviderError("http_error", "unavailable", retryable=True, status_code=503),
            {"role": "assistant", "content": "recovered"},
        ])

        def complete(messages, tools):
            provider.requests.append({"messages": list(messages), "tools": tools})
            value = next(responses)
            if isinstance(value, Exception):
                raise value
            return value

        provider.complete = complete
        with (
            patch("agent.agent_loop.config.MAX_API_RETRIES", 2),
            patch("agent.agent_loop.config.RETRY_BASE_DELAY", 0.25),
            patch("agent.agent_loop.time.sleep") as sleep,
        ):
            result = self.run_quietly("retry", provider=provider, registry=ToolRegistry())

        self.assertEqual(result.stop_reason, "final_answer")
        self.assertEqual(result.answer, "recovered")
        self.assertEqual(len(provider.requests), 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [0.25, 0.5])

    def test_non_retryable_api_error_stops_immediately(self):
        error = ProviderError("http_error", "bad request", status_code=400)
        provider = FakeProvider([])
        provider.complete = lambda messages, tools: (_ for _ in ()).throw(error)

        with (
            patch("agent.agent_loop.config.MAX_API_RETRIES", 2),
            patch("agent.agent_loop.time.sleep") as sleep,
        ):
            result = self.run_quietly("fail", provider=provider, registry=ToolRegistry())

        self.assertFalse(result.ok)
        self.assertEqual(result.stop_reason, "api_error")
        self.assertEqual(result.error, {
            "type": "http_error",
            "message": "bad request",
            "retryable": False,
            "status_code": 400,
            "attempts": 1,
        })
        sleep.assert_not_called()

    def test_retry_exhaustion_returns_api_error(self):
        provider = FakeProvider([])
        attempts = []

        def fail(messages, tools):
            attempts.append(1)
            raise ProviderError("network_error", "timeout", retryable=True)

        provider.complete = fail
        with (
            patch("agent.agent_loop.config.MAX_API_RETRIES", 2),
            patch("agent.agent_loop.config.RETRY_BASE_DELAY", 0),
            patch("agent.agent_loop.time.sleep"),
        ):
            result = self.run_quietly("fail", provider=provider, registry=ToolRegistry())

        self.assertEqual(result.stop_reason, "api_error")
        self.assertEqual(result.error["attempts"], 3)
        self.assertEqual(len(attempts), 3)

    def test_tool_error_limit_stops_repeated_failures(self):
        provider = FakeProvider([{
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "name": "missing", "arguments": {}},
                {"id": "call_2", "name": "missing", "arguments": {}},
            ],
        }])

        with patch("agent.agent_loop.config.MAX_TOOL_ERRORS", 2):
            result = self.run_quietly("fail tools", provider=provider, registry=ToolRegistry())

        self.assertEqual(result.stop_reason, "tool_error_limit")
        self.assertEqual(result.tool_errors, 2)
        self.assertEqual(result.error["type"], "unknown_tool")
        self.assertEqual(len(result.messages), 5)


if __name__ == "__main__":
    unittest.main()
