import contextlib
import io
import json
import unittest
from unittest.mock import patch

from agent.agent_loop import run


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

    def test_normalized_tool_call_stays_internal_across_rounds(self):
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
            tools=[{"name": "calc"}],
            tool_runner=lambda call: 5,
        )

        self.assertEqual(answer, "The result is 5.")
        second_messages = provider.requests[1]["messages"]
        self.assertEqual(second_messages[2]["tool_calls"][0]["name"], "calc")
        self.assertNotIn("function", second_messages[2]["tool_calls"][0])
        self.assertEqual(second_messages[3]["role"], "tool")
        self.assertEqual(second_messages[3]["tool_call_id"], "call_1")
        self.assertEqual(json.loads(second_messages[3]["content"]), {"ok": True, "result": 5})

    def test_tool_error_is_returned_to_model(self):
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

        def fail(_call):
            raise ValueError("unsupported expression")

        answer = self.run_quietly("calculate", provider=provider, tools=[], tool_runner=fail)

        self.assertEqual(answer, "I could not use that expression.")
        tool_result = json.loads(provider.requests[1]["messages"][-1]["content"])
        self.assertFalse(tool_result["ok"])
        self.assertIn("unsupported expression", tool_result["error"])

    def test_max_steps_has_a_clear_stop_message(self):
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
                tools=[],
                tool_runner=lambda _call: None,
            )

        self.assertEqual(answer, "Stopped after MAX_STEPS=1.")


if __name__ == "__main__":
    unittest.main()
