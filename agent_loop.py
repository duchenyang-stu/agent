import json

try:
    from . import config
    from .providers import build_provider
    from .tools import TOOLS, run_tool
except ImportError:
    import config
    from providers import build_provider
    from tools import TOOLS, run_tool


def dump(step, messages):
    print(f"\n--- step {step}: internal messages ---")
    print(json.dumps(messages, ensure_ascii=False, indent=2))


def run(task, provider=None, tools=None, tool_runner=None):
    provider = provider or build_provider()
    tools = TOOLS if tools is None else tools
    tool_runner = tool_runner or run_tool
    messages = [
        {"role": "system", "content": "You are a tiny agent. Use tools when useful."},
        {"role": "user", "content": task},
    ]
    for step in range(1, config.MAX_STEPS + 1):
        dump(step, messages)
        msg = provider.complete(messages, tools)
        messages.append(msg)
        if not msg.get("tool_calls"):
            dump("final", messages)
            return msg.get("content", "")
        for call in msg["tool_calls"]:
            try:
                out = {"ok": True, "result": tool_runner(call)}
            except Exception as e:
                out = {"ok": False, "error": repr(e)}
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(out, ensure_ascii=False),
            })
    return f"Stopped after MAX_STEPS={config.MAX_STEPS}."
