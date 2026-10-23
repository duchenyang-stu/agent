import json
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    from . import config
    from .providers import ProviderError, build_provider
    from .tools import DEFAULT_REGISTRY, ToolError
except ImportError:
    import config
    from providers import ProviderError, build_provider
    from tools import DEFAULT_REGISTRY, ToolError


@dataclass
class RunResult:
    stop_reason: str
    answer: str = ""
    steps: int = 0
    messages: list = field(default_factory=list)
    tool_errors: int = 0
    error: Optional[dict] = None

    @property
    def ok(self):
        return self.stop_reason == "final_answer"

    def __str__(self):
        if self.answer:
            return self.answer
        if self.stop_reason == "max_steps":
            return f"Stopped after MAX_STEPS={self.steps}."
        if self.stop_reason == "tool_error_limit":
            return f"Stopped after {self.tool_errors} tool errors."
        if self.error:
            return f"Stopped because of {self.stop_reason}: {self.error['message']}"
        return f"Stopped because of {self.stop_reason}."


def dump(step, messages):
    print(f"\n--- step {step}: internal messages ---")
    print(json.dumps(messages, ensure_ascii=False, indent=2))


def complete_with_retries(provider, messages, tools):
    for retry in range(config.MAX_API_RETRIES + 1):
        try:
            return provider.complete(messages, tools)
        except ProviderError as e:
            e.attempts = retry + 1
            if not e.retryable or retry == config.MAX_API_RETRIES:
                raise
            delay = config.RETRY_BASE_DELAY * (2 ** retry)
            print(
                f"\n--- provider retry {retry + 1}/{config.MAX_API_RETRIES} "
                f"after {delay:g}s: {e} ---"
            )
            time.sleep(delay)


def provider_error_data(error):
    return {
        "type": error.error_type,
        "message": str(error),
        "retryable": error.retryable,
        "status_code": error.status_code,
        "attempts": error.attempts,
    }


def tool_error_data(error):
    if isinstance(error, ToolError):
        return {"ok": False, "error": {"type": error.code, "message": str(error)}}
    return {
        "ok": False,
        "error": {"type": "tool_execution_error", "message": str(error)},
    }


def run(task, provider=None, registry=None):
    provider = provider or build_provider()
    registry = DEFAULT_REGISTRY if registry is None else registry
    tools = registry.schemas()
    messages = [
        {"role": "system", "content": "You are a tiny agent. Use tools when useful."},
        {"role": "user", "content": task},
    ]
    tool_errors = 0
    for step in range(1, config.MAX_STEPS + 1):
        dump(step, messages)
        try:
            msg = complete_with_retries(provider, messages, tools)
        except ProviderError as e:
            return RunResult(
                stop_reason="api_error",
                steps=step,
                messages=messages,
                tool_errors=tool_errors,
                error=provider_error_data(e),
            )
        messages.append(msg)
        if not msg.get("tool_calls"):
            dump("final", messages)
            return RunResult(
                stop_reason="final_answer",
                answer=msg.get("content", ""),
                steps=step,
                messages=messages,
                tool_errors=tool_errors,
            )
        for call in msg["tool_calls"]:
            try:
                out = {"ok": True, "result": registry.run_tool(call)}
            except Exception as e:
                tool_errors += 1
                out = tool_error_data(e)
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(out, ensure_ascii=False),
            })
            if not out["ok"] and tool_errors >= config.MAX_TOOL_ERRORS:
                return RunResult(
                    stop_reason="tool_error_limit",
                    steps=step,
                    messages=messages,
                    tool_errors=tool_errors,
                    error=out["error"],
                )
    return RunResult(
        stop_reason="max_steps",
        steps=config.MAX_STEPS,
        messages=messages,
        tool_errors=tool_errors,
    )
