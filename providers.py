import json
import urllib.request

try:
    from . import config
except ImportError:
    import config


class Provider:
    def complete(self, messages, tools):
        raise NotImplementedError


class OpenAICompatibleProvider(Provider):
    def __init__(self, api_key, model, chat_url, show_meta=True):
        if not api_key:
            raise RuntimeError("Missing LLM_API_KEY. Put it in agent/.env or export it.")
        self.api_key = api_key
        self.model = model
        self.chat_url = chat_url
        self.show_meta = show_meta

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
        with urllib.request.urlopen(req, timeout=60) as r:
            return self.from_provider_response(json.load(r))

    def to_provider_tools(self, tools):
        return [{"type": "function", "function": tool} for tool in tools]

    def to_provider_messages(self, messages):
        return [self.to_provider_message(m) for m in messages]

    def to_provider_message(self, msg):
        if msg["role"] != "assistant" or not msg.get("tool_calls"):
            return msg
        return {
            "role": "assistant",
            "content": msg.get("content") or "",
            "tool_calls": [self.to_provider_tool_call(c) for c in msg["tool_calls"]],
        }

    def to_provider_tool_call(self, call):
        return {
            "id": call["id"],
            "type": "function",
            "function": {
                "name": call["name"],
                "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False),
            },
        }

    def from_provider_response(self, data):
        raw = data["choices"][0]["message"]
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
        fn = call.get("function") or {}
        if not call.get("id"):
            raise ValueError(f"tool call missing id: {call}")
        if not fn.get("name"):
            raise ValueError(f"tool call missing function.name: {call}")
        args_text = fn.get("arguments") or "{}"
        try:
            args = json.loads(args_text) if isinstance(args_text, str) else args_text
        except json.JSONDecodeError as e:
            raise ValueError(f"tool call arguments must be JSON: {args_text}") from e
        if not isinstance(args, dict):
            raise ValueError(f"tool call arguments must decode to object: {args}")
        return {"id": call["id"], "name": fn["name"], "arguments": args}


def build_provider():
    if config.LLM_PROVIDER == "openai-compatible":
        return OpenAICompatibleProvider(
            config.API_KEY,
            config.MODEL,
            config.CHAT_COMPLETIONS_URL,
            config.SHOW_PROVIDER_META,
        )
    raise RuntimeError(f"Unsupported LLM_PROVIDER={config.LLM_PROVIDER!r}; add a Provider adapter for it.")
