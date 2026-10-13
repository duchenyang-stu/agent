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

def getenv(name, *fallbacks, default=None):
    for key in (name, *fallbacks):
        value = os.getenv(key)
        if value:
            return value
    return default

LLM_PROVIDER = getenv("LLM_PROVIDER", default="openai-compatible")
API_KEY = getenv("LLM_API_KEY", "OPENAI_API_KEY")
MODEL = getenv("LLM_MODEL", "OPENAI_MODEL", default="deepseek-flash")
BASE_URL = getenv("LLM_BASE_URL", "OPENAI_BASE_URL", default="https://api.deepseek.com").rstrip("/")
CHAT_COMPLETIONS_URL = getenv("LLM_CHAT_COMPLETIONS_URL", default=f"{BASE_URL}/chat/completions")
MAX_STEPS = int(os.getenv("MAX_STEPS", "6"))

class Provider:
    def complete(self, messages, tools):
        raise NotImplementedError

class OpenAICompatibleProvider(Provider):
    def __init__(self, api_key, model, chat_url):
        if not api_key:
            raise RuntimeError("Missing LLM_API_KEY. Put it in agent/.env or export it.")
        self.api_key = api_key
        self.model = model
        self.chat_url = chat_url

    def complete(self, messages, tools):
        body = json.dumps({"model": self.model, "messages": messages, "tools": tools}).encode()
        req = urllib.request.Request(self.chat_url, data=body, headers={
            "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return self.parse_response(json.load(r))

    def parse_response(self, data):
        return data["choices"][0]["message"]

def build_provider():
    if LLM_PROVIDER == "openai-compatible":
        return OpenAICompatibleProvider(API_KEY, MODEL, CHAT_COMPLETIONS_URL)
    raise RuntimeError(f"Unsupported LLM_PROVIDER={LLM_PROVIDER!r}; add a Provider adapter for it.")

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

def dump(step, messages):
    print(f"\n--- step {step}: messages ---")
    print(json.dumps(messages, ensure_ascii=False, indent=2))

def run(task):
    provider = build_provider()
    messages = [{"role": "system", "content": "You are a tiny agent. Use tools when useful."},
                {"role": "user", "content": task}]
    for step in range(1, MAX_STEPS + 1):
        dump(step, messages)
        msg = provider.complete(messages, TOOLS)
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
