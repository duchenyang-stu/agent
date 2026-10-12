import ast, json, operator, os, sys, urllib.request
from datetime import datetime
from pathlib import Path

def load_env(path=Path(__file__).with_name(".env")):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

load_env()
MODEL = os.getenv("OPENAI_MODEL", "deepseek-flash")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com").rstrip("/")
CHAT_COMPLETIONS_URL = f"{BASE_URL}/chat/completions"
MAX_STEPS = int(os.getenv("MAX_STEPS", "6"))

def now(city="local"):
    return f"{city}: {datetime.now().isoformat(timespec='seconds')}"

def count_words(text):
    return {"words": len(text.split()), "chars": len(text)}

def calc(expr):
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}
    def ev(n):
        if isinstance(n, ast.Expression): return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in ops: return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in ops: return ops[type(n.op)](ev(n.operand))
        raise ValueError(f"unsupported expression: {expr}")
    return ev(ast.parse(expr, mode="eval"))

FUNCS = {"now": now, "count_words": count_words, "calc": calc}

def tool(name, desc, props, required=()):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(required)}}}

TOOLS = [
    tool("now", "Get current local time.", {"city": {"type": "string"}}),
    tool("count_words", "Count words and chars.", {"text": {"type": "string"}}, ["text"]),
    tool("calc", "Calculate arithmetic like 2*(3+4).", {"expr": {"type": "string"}}, ["expr"]),
]

def llm(messages):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY. Put it in agent/.env or export it.")
    body = json.dumps({"model": MODEL, "messages": messages, "tools": TOOLS}).encode()
    req = urllib.request.Request(CHAT_COMPLETIONS_URL, data=body, headers={
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "Content-Type": "application/json","reasoning_effort":"low"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["choices"][0]["message"]

def dump(step, messages):
    print(f"\n--- step {step}: messages ---")
    print(json.dumps(messages, ensure_ascii=False, indent=2))

def run(task):
    messages = [{"role": "system", "content": "You are a tiny agent. Use tools when useful."},
                {"role": "user", "content": task}]
    for step in range(1, MAX_STEPS + 1):
        dump(step, messages)
        msg = llm(messages)
        messages.append(msg)
        if not msg.get("tool_calls"):
            dump("final", messages)
            return msg.get("content", "")
        for call in msg["tool_calls"]:
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
                out = {"ok": True, "result": FUNCS[call["function"]["name"]](**args)}
            except Exception as e:
                out = {"ok": False, "error": repr(e)}
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": json.dumps(out, ensure_ascii=False)})
    return f"Stopped after MAX_STEPS={MAX_STEPS}."


if __name__ == "__main__":
    default = "What is (12+8)-3, and how many words are in 'hello tiny agent'?"
    print(run(" ".join(sys.argv[1:]) or default))
