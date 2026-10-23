import contextlib
import io
import json
import unittest

from agent.providers import OpenAICompatibleProvider, ProviderError


def make_provider(show_meta=False):
    return OpenAICompatibleProvider(
        api_key="test-key",
        model="test-model",
        chat_url="https://example.invalid/chat/completions",
        show_meta=show_meta,
    )


class MessageNormalizationTest(unittest.TestCase):
    def test_internal_tool_call_is_converted_only_at_provider_boundary(self):
        provider = make_provider()
        internal = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "name": "calc", "arguments": {"expr": "2+3"}},
            ],
        }

        external = provider.to_provider_message(internal)

        self.assertEqual(internal["tool_calls"][0]["name"], "calc")
        self.assertNotIn("function", internal["tool_calls"][0])
        self.assertEqual(external["tool_calls"][0]["type"], "function")
        self.assertEqual(external["tool_calls"][0]["function"]["name"], "calc")
        self.assertEqual(
            json.loads(external["tool_calls"][0]["function"]["arguments"]),
            {"expr": "2+3"},
        )

    def test_response_drops_provider_private_fields(self):
        provider = make_provider()
        message = provider.from_provider_response({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "done",
                    "reasoning_content": "private reasoning",
                    "vendor_field": {"value": 1},
                },
            }],
        })

        self.assertEqual(message, {"role": "assistant", "content": "done"})

    def test_request_drops_internal_private_fields(self):
        provider = make_provider()
        messages = provider.to_provider_messages([
            {"role": "user", "content": "hello", "trace_id": "secret"},
            {
                "role": "assistant",
                "content": "",
                "trace_id": "secret",
                "tool_calls": [{"id": "call_1", "name": "calc", "arguments": {}}],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"ok":true}',
                "trace_id": "secret",
            },
        ])

        self.assertEqual(messages[0], {"role": "user", "content": "hello"})
        self.assertNotIn("trace_id", messages[1])
        self.assertNotIn("trace_id", messages[2])
        self.assertEqual(messages[2]["tool_call_id"], "call_1")

    def test_provider_meta_can_be_logged_without_entering_internal_message(self):
        provider = make_provider(show_meta=True)
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            message = provider.from_provider_response({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "done",
                        "reasoning_content": "private reasoning",
                    },
                }],
            })

        self.assertIn("reasoning_content", output.getvalue())
        self.assertNotIn("reasoning_content", message)

    def test_none_content_with_tool_calls_becomes_empty_string(self):
        provider = make_provider()
        message = provider.from_provider_assistant({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "calc", "arguments": '{"expr":"2+3"}'},
            }],
        })

        self.assertEqual(message, {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "name": "calc", "arguments": {"expr": "2+3"}},
            ],
        })

    def test_empty_assistant_message_is_rejected(self):
        provider = make_provider()

        with self.assertRaisesRegex(ValueError, "neither content nor tool_calls"):
            provider.from_provider_assistant({"role": "assistant", "content": None})

    def test_invalid_tool_arguments_are_rejected(self):
        provider = make_provider()

        with self.assertRaisesRegex(ProviderError, "arguments must be JSON") as raised:
            provider.from_provider_tool_call({
                "id": "call_1",
                "function": {"name": "calc", "arguments": "not-json"},
            })
        self.assertEqual(raised.exception.error_type, "invalid_tool_arguments")

    def test_internal_non_object_tool_arguments_are_rejected(self):
        provider = make_provider()

        with self.assertRaisesRegex(ValueError, "arguments must be an object"):
            provider.to_provider_tool_call({
                "id": "call_1",
                "name": "calc",
                "arguments": [],
            })

    def test_non_object_tool_arguments_are_rejected(self):
        provider = make_provider()

        with self.assertRaisesRegex(ProviderError, "decode to object") as raised:
            provider.from_provider_tool_call({
                "id": "call_1",
                "function": {"name": "calc", "arguments": "[]"},
            })
        self.assertEqual(raised.exception.error_type, "invalid_tool_arguments")

    def test_malformed_provider_response_is_rejected_clearly(self):
        provider = make_provider()

        with self.assertRaisesRegex(ValueError, "no choices"):
            provider.from_provider_response({"choices": []})
        with self.assertRaisesRegex(ValueError, "no message"):
            provider.from_provider_response({"choices": [{}]})


if __name__ == "__main__":
    unittest.main()
