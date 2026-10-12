# Minimal Agent Loop

这个目录先放一个不依赖 Agent 框架的最小版本：

```bash
cp agent/.env.example agent/.env
# edit agent/.env and put your real key in OPENAI_API_KEY
python3 agent/minimal_agent.py
python3 agent/minimal_agent.py "现在几点？再算一下 18*7"
```

`agent/.env` 放运行时配置，例如：

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-flash
MAX_STEPS=6
```

也可以在命令行临时覆盖 `.env` 里的值：

```bash
OPENAI_BASE_URL="https://your-provider.example/v1" \
OPENAI_API_KEY="..." \
OPENAI_MODEL="your-model" \
python3 agent/minimal_agent.py
```

核心结构就是：

```text
messages = [system, user]
while step < MAX_STEPS:
    assistant = LLM(messages, tools)
    messages.append(assistant)
    if assistant 没有 tool_calls:
        return assistant.content
    for tool_call in assistant.tool_calls:
        result = run_local_tool(tool_call)
        messages.append(tool_result_or_error)
```

[minimal_agent.py](minimal_agent.py) 里做了三件关键事：

1. `MAX_STEPS = 6` 限制最多循环几步，防止模型一直调用工具。
2. 工具执行放在 `try/except` 里，报错也会作为 `{"ok": false, "error": ...}` 放回 `messages`。
3. 每一轮都打印完整 `messages`，所以你能看到“LLM 输出工具调用 -> 代码执行工具 -> 工具结果喂回 LLM”的全过程。

这就是 smolagents、LangGraph 等框架的底层骨架。框架主要是在这个循环外面继续加工程能力：工具注册、状态管理、暂停恢复、分支、人类审批、日志、追踪和持久化。
