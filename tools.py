import ast
import inspect
import operator
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Callable


class ToolError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class UnknownToolError(ToolError):
    def __init__(self, name):
        super().__init__("unknown_tool", f"unknown tool: {name!r}")


class InvalidToolArgumentsError(ToolError):
    def __init__(self, name, message):
        super().__init__("invalid_arguments", f"invalid arguments for tool {name!r}: {message}")


class ToolExecutionError(ToolError):
    def __init__(self, name, message):
        super().__init__("tool_execution_error", f"tool {name!r} failed: {message}")


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable

    def schema(self):
        # Keep handlers local and prevent callers from mutating the registry's schema.
        return {
            "name": self.name,
            "description": self.description,
            "parameters": deepcopy(self.parameters),
        }


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, tool):
        if not isinstance(tool, Tool):
            raise TypeError("register expects a Tool")
        if not isinstance(tool.name, str) or not tool.name.strip():
            raise ValueError("tool name must be a non-empty string")
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        if not callable(tool.handler):
            raise TypeError(f"tool handler must be callable: {tool.name}")
        self._tools[tool.name] = Tool(
            tool.name, tool.description, deepcopy(tool.parameters), tool.handler,
        )
        return tool

    def tool(self, *, description, parameters, name=None):
        def decorate(handler):
            self.register(Tool(name or handler.__name__, description, parameters, handler))
            return handler

        return decorate

    def schemas(self):
        return [tool.schema() for tool in self._tools.values()]

    def run_tool(self, call):
        if not isinstance(call, dict):
            raise ToolError("invalid_arguments", "tool call must be an object")
        name = call.get("name")
        if not isinstance(name, str) or name not in self._tools:
            raise UnknownToolError(name)
        arguments = call.get("arguments", {})
        if not isinstance(arguments, dict):
            raise InvalidToolArgumentsError(name, "arguments must be an object")
        handler = self._tools[name].handler
        try:
            inspect.signature(handler).bind(**arguments)
        except TypeError as e:
            raise InvalidToolArgumentsError(name, str(e)) from e
        try:
            return handler(**arguments)
        except Exception as e:
            raise ToolExecutionError(name, str(e)) from e


DEFAULT_REGISTRY = ToolRegistry()


@DEFAULT_REGISTRY.tool(
    description="Get current local time.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": [],
    },
)
def now(city="local"):
    return f"{city}: {datetime.now().isoformat(timespec='seconds')}"


@DEFAULT_REGISTRY.tool(
    description="Count words and chars.",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)
def count_words(text):
    return {"words": len(text.split()), "chars": len(text)}


@DEFAULT_REGISTRY.tool(
    description="Calculate arithmetic like 2*(3+4).",
    parameters={
        "type": "object",
        "properties": {"expr": {"type": "string"}},
        "required": ["expr"],
    },
)
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


def run_tool(call, registry=None):
    registry = DEFAULT_REGISTRY if registry is None else registry
    return registry.run_tool(call)
