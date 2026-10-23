import unittest

from agent.tools import (
    DEFAULT_REGISTRY,
    InvalidToolArgumentsError,
    Tool,
    ToolError,
    ToolExecutionError,
    ToolRegistry,
    UnknownToolError,
    run_tool,
)


class ToolRegistryTest(unittest.TestCase):
    def test_decorator_registers_schema_and_handler_together(self):
        registry = ToolRegistry()

        @registry.tool(
            name="greet",
            description="Greet someone.",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )
        def greeting(name):
            return f"Hello, {name}!"

        self.assertEqual(registry.schemas(), [{
            "name": "greet",
            "description": "Greet someone.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        }])
        self.assertEqual(greeting("Ada"), "Hello, Ada!")
        self.assertEqual(run_tool({
            "id": "call_1", "name": "greet", "arguments": {"name": "Ada"},
        }, registry=registry), "Hello, Ada!")

    def test_duplicate_registration_does_not_replace_handler(self):
        registry = ToolRegistry()
        registry.register(Tool("answer", "Original", {"type": "object"}, lambda: 42))

        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(Tool("answer", "Replacement", {"type": "object"}, lambda: 0))

        self.assertEqual(registry.run_tool({"name": "answer"}), 42)

    def test_schema_mutation_does_not_change_registry(self):
        registry = ToolRegistry()
        parameters = {"type": "object", "properties": {"text": {"type": "string"}}}
        registry.register(Tool("echo", "Echo", parameters, lambda text: text))

        parameters["properties"]["text"]["type"] = "number"
        exported = registry.schemas()
        exported[0]["parameters"]["properties"]["text"]["type"] = "boolean"

        self.assertEqual(
            registry.schemas()[0]["parameters"]["properties"]["text"]["type"], "string",
        )

    def test_unknown_tool_has_a_clear_error(self):
        with self.assertRaisesRegex(UnknownToolError, "unknown tool: 'missing'"):
            ToolRegistry().run_tool({"name": "missing", "arguments": {}})

    def test_non_object_tool_call_is_an_argument_error(self):
        with self.assertRaisesRegex(ToolError, "tool call must be an object") as raised:
            ToolRegistry().run_tool([])
        self.assertEqual(raised.exception.code, "invalid_arguments")

    def test_invalid_arguments_are_rejected_before_handler_execution(self):
        registry = ToolRegistry()
        executed = []

        @registry.tool(
            description="Remember text.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )
        def remember(text):
            executed.append(text)

        for arguments, error in [
            ([], "must be an object"),
            (None, "must be an object"),
            ({}, "invalid arguments"),
            ({"text": "hello", "unexpected": True}, "invalid arguments"),
        ]:
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(InvalidToolArgumentsError, error):
                    registry.run_tool({"name": "remember", "arguments": arguments})

        self.assertEqual(executed, [])

    def test_existing_tools_use_the_default_registry(self):
        self.assertEqual(
            [schema["name"] for schema in DEFAULT_REGISTRY.schemas()],
            ["now", "count_words", "calc"],
        )
        self.assertEqual(run_tool({"name": "calc", "arguments": {"expr": "2*(3+4)"}}), 14)
        self.assertEqual(
            run_tool({"name": "count_words", "arguments": {"text": "hello tiny agent"}}),
            {"words": 3, "chars": 16},
        )
        self.assertTrue(run_tool({"name": "now"}).startswith("local: "))

    def test_handler_error_is_not_mislabeled_as_an_argument_error(self):
        registry = ToolRegistry()

        def broken():
            raise TypeError("handler failed internally")

        registry.register(Tool("broken", "Fail", {"type": "object"}, broken))
        with self.assertRaisesRegex(ToolExecutionError, "handler failed internally"):
            registry.run_tool({"name": "broken"})

    def test_registries_are_isolated(self):
        registry = ToolRegistry()
        registry.register(Tool("custom", "Custom", {"type": "object"}, lambda: "ok"))

        self.assertEqual(ToolRegistry().schemas(), [])
        self.assertNotIn("custom", [schema["name"] for schema in DEFAULT_REGISTRY.schemas()])


if __name__ == "__main__":
    unittest.main()
