import contextlib
import io
import json
import unittest
from unittest.mock import patch

from agent.agent_loop import run
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

        answer = self.run_quietly("double 4", provider=provider, registry=registry)

        self.assertEqual(answer, "The first call returned 8.")
        self.assertEqual(provider.requests[0]["tools"], registry.schemas())
        tool_messages = provider.requests[1]["messages"][-2:]
        self.assertEqual(tool_messages[0]["tool_call_id"], "call_ok")
        self.assertEqual(json.loads(tool_messages[0]["content"]), {"ok": True, "result": 8})
        self.assertEqual(tool_messages[1]["tool_call_id"], "call_error")
        failure = json.loads(tool_messages[1]["content"])
        self.assertFalse(failure["ok"])
        self.assertIn("invalid arguments", failure["error"])

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

        answer = self.run_quietly(
            "calculate",
            provider=provider,
            registry=registry,
        )

        self.assertEqual(answer, "The result is 5.")
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

        answer = self.run_quietly("calculate", provider=provider, registry=registry)

        self.assertEqual(answer, "I could not use that expression.")
        tool_result = json.loads(provider.requests[1]["messages"][-1]["content"])
        self.assertFalse(tool_result["ok"])
        self.assertIn("unsupported expression", tool_result["error"])

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
            answer = self.run_quietly(
                "keep going",
                provider=provider,
                registry=registry,
            )

        self.assertEqual(answer, "Stopped after MAX_STEPS=1.")


if __name__ == "__main__":
    unittest.main()
