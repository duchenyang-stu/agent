import ast
import operator
from datetime import datetime


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
    return {
        "name": name,
        "description": desc,
        "parameters": {"type": "object", "properties": props, "required": list(required)},
    }


TOOLS = [
    tool("now", "Get current local time.", {"city": {"type": "string"}}),
    tool("count_words", "Count words and chars.", {"text": {"type": "string"}}, ["text"]),
    tool("calc", "Calculate arithmetic like 2*(3+4).", {"expr": {"type": "string"}}, ["expr"]),
]


def run_tool(call):
    return FUNCS[call["name"]](**call.get("arguments", {}))
