import json
import socket
import urllib.error
import urllib.request

try:
    from . import config
except ImportError:
    import config


class Provider:
    def complete(self, messages, tools):
        raise NotImplementedError


class ProviderError(RuntimeError):
    def __init__(self, error_type, message, *, retryable=False, status_code=None):
        super().__init__(message)
        self.error_type = error_type
        self.retryable = retryable
        self.status_code = status_code
        self.attempts = 1


class OpenAICompatibleProvider(Provider):
    RETRYABLE_STATUS_CODES = {408, 409, 429}

    def __init__(self, api_key, model, chat_url, show_meta=True, timeout=60):
        if not api_key:
            raise RuntimeError("Missing LLM_API_KEY. Put it in agent/.env or export it.")
        self.api_key = api_key
        self.model = model
        self.chat_url = chat_url
        self.show_meta = show_meta
        self.timeout = timeout

    def complete(self, messages, tools):
        body = json.dumps({
            "model": self.model,
            "messages": self.to_provider_messages(messages),
            "tools": self.to_provider_tools(tools),
        }).encode()
        req = urllib.request.Request(self.chat_url, data=body, headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                try:
                    data = json.load(r)
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    raise ProviderError(
                        "invalid_response", "provider response is not valid JSON",
                    ) from e
        except urllib.error.HTTPError as e:
            retryable = e.code in self.RETRYABLE_STATUS_CODES or 500 <= e.code < 600
            raise ProviderError(
                "http_error",
                f"provider HTTP {e.code}: {e.reason}",
                retryable=retryable,
                status_code=e.code,
            ) from e
        except (urllib.error.URLError, TimeoutError, socket.timeout) as e:
            raise ProviderError(
                "network_error", f"provider network error: {e}", retryable=True,
            ) from e
        try:
            return self.from_provider_response(data)
        except ProviderError:
            raise
        except ValueError as e:
            raise ProviderError("invalid_response", str(e)) from e

    def to_provider_tools(self, tools):
        return [{"type": "function", "function": tool} for tool in tools]

    def to_provider_messages(self, messages):
        return [self.to_provider_message(m) for m in messages]

    def to_provider_message(self, msg):
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            return {
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": [self.to_provider_tool_call(c) for c in msg["tool_calls"]],
            }
        if role == "tool":
            if not msg.get("tool_call_id"):
                raise ValueError(f"tool message missing tool_call_id: {msg}")
            return {
                "role": "tool",
                "tool_call_id": msg["tool_call_id"],
                "content": msg.get("content") or "",
            }
        if role in {"system", "user", "assistant"}:
            return {"role": role, "content": msg.get("content") or ""}
        raise ValueError(f"unsupported internal message role: {role!r}")

    def to_provider_tool_call(self, call):
        if not isinstance(call, dict):
            raise ValueError(f"tool call must be an object: {call!r}")
        if not call.get("id"):
            raise ValueError(f"tool call missing id: {call}")
        if not call.get("name"):
            raise ValueError(f"tool call missing name: {call}")
        arguments = call.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError(f"tool call arguments must be an object: {arguments!r}")
        return {
            "id": call["id"],
            "type": "function",
            "function": {
                "name": call["name"],
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        }

    def from_provider_response(self, data):
        if not isinstance(data, dict):
            raise ValueError(f"provider response must be an object: {data!r}")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"provider response has no choices: {data!r}")
        raw = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(raw, dict):
            raise ValueError(f"provider choice has no message: {choices[0]!r}")
        if self.show_meta:
            meta = {k: v for k, v in raw.items() if k not in {"role", "content", "tool_calls"}}
            if meta:
                print("\n--- provider meta, not sent back ---")
                print(json.dumps(meta, ensure_ascii=False, indent=2))
        return self.from_provider_assistant(raw)

    def from_provider_assistant(self, raw):
        content = raw.get("content") or ""
        tool_calls = raw.get("tool_calls") or []
        if not content and not tool_calls:
            raise ValueError(f"assistant message has neither content nor tool_calls: {raw}")
        msg = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = [self.from_provider_tool_call(c) for c in tool_calls]
        return msg

    def from_provider_tool_call(self, call):
        if not isinstance(call, dict):
            raise ValueError(f"tool call must be an object: {call!r}")
        fn = call.get("function") or {}
        if not call.get("id"):
            raise ValueError(f"tool call missing id: {call}")
        if not fn.get("name"):
            raise ValueError(f"tool call missing function.name: {call}")
        args_text = fn.get("arguments") or "{}"
        try:
            args = json.loads(args_text) if isinstance(args_text, str) else args_text
        except json.JSONDecodeError as e:
            raise ProviderError(
                "invalid_tool_arguments",
                f"tool call arguments must be JSON: {args_text}",
            ) from e
        if not isinstance(args, dict):
            raise ProviderError(
                "invalid_tool_arguments",
                f"tool call arguments must decode to object: {args}",
            )
        return {"id": call["id"], "name": fn["name"], "arguments": args}


def build_provider():
    if config.LLM_PROVIDER == "openai-compatible":
        return OpenAICompatibleProvider(
            config.API_KEY,
            config.MODEL,
            config.CHAT_COMPLETIONS_URL,
            config.SHOW_PROVIDER_META,
            config.API_TIMEOUT,
        )
    raise RuntimeError(f"Unsupported LLM_PROVIDER={config.LLM_PROVIDER!r}; add a Provider adapter for it.")
