import io
import socket
import unittest
import urllib.error
from unittest.mock import patch

from agent.providers import OpenAICompatibleProvider, ProviderError


def make_provider(timeout=60):
    return OpenAICompatibleProvider(
        api_key="test-key",
        model="test-model",
        chat_url="https://example.invalid/chat/completions",
        show_meta=False,
        timeout=timeout,
    )


class ProviderErrorTest(unittest.TestCase):
    def test_retryable_http_status_is_classified(self):
        error = urllib.error.HTTPError(
            "https://example.invalid", 429, "Too Many Requests", {}, io.BytesIO(),
        )

        with patch("agent.providers.urllib.request.urlopen", side_effect=error):
            with self.assertRaises(ProviderError) as raised:
                make_provider().complete([], [])

        self.assertEqual(raised.exception.error_type, "http_error")
        self.assertEqual(raised.exception.status_code, 429)
        self.assertTrue(raised.exception.retryable)

    def test_non_retryable_http_status_is_classified(self):
        error = urllib.error.HTTPError(
            "https://example.invalid", 401, "Unauthorized", {}, io.BytesIO(),
        )

        with patch("agent.providers.urllib.request.urlopen", side_effect=error):
            with self.assertRaises(ProviderError) as raised:
                make_provider().complete([], [])

        self.assertEqual(raised.exception.status_code, 401)
        self.assertFalse(raised.exception.retryable)

    def test_network_timeout_is_retryable(self):
        with patch(
            "agent.providers.urllib.request.urlopen",
            side_effect=urllib.error.URLError(socket.timeout("timed out")),
        ):
            with self.assertRaises(ProviderError) as raised:
                make_provider(timeout=3).complete([], [])

        self.assertEqual(raised.exception.error_type, "network_error")
        self.assertTrue(raised.exception.retryable)

    def test_invalid_json_response_is_not_retryable(self):
        with patch(
            "agent.providers.urllib.request.urlopen",
            return_value=io.BytesIO(b"not-json"),
        ):
            with self.assertRaises(ProviderError) as raised:
                make_provider().complete([], [])

        self.assertEqual(raised.exception.error_type, "invalid_response")
        self.assertFalse(raised.exception.retryable)

    def test_malformed_message_is_wrapped_as_invalid_response(self):
        response = io.BytesIO(b'{"choices": []}')
        with patch("agent.providers.urllib.request.urlopen", return_value=response):
            with self.assertRaises(ProviderError) as raised:
                make_provider().complete([], [])

        self.assertEqual(raised.exception.error_type, "invalid_response")
        self.assertIn("no choices", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
