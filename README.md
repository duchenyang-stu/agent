# Agent From Scratch

这个目录的目标不是直接用成熟框架，而是从最小 while 循环开始，分 10 次迭代自己实现一遍 Agent。路线的核心判断是：

```text
Agent = while 循环 + LLM + 工具 + 不断增长的 messages / state
```

当前版本在 [minimal_agent.py](minimal_agent.py)：

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

当前教学版已经拆成几个小文件：

- [minimal_agent.py](minimal_agent.py)：命令行入口。
- [config.py](config.py)：读取 `.env` 和运行配置。
- [providers.py](providers.py)：provider adapter，负责内部格式和外部 API 格式的双向转换。
- [tools.py](tools.py)：本地工具和工具 schema。
- [agent_loop.py](agent_loop.py)：Agent while 循环。

Agent 内部使用自己的 tool call 格式：

```json
{
  "id": "call_123",
  "name": "calc",
  "arguments": {
    "expr": "2+3"
  }
}
```

OpenAI-compatible 的外部格式只存在于 [providers.py](providers.py) 里：

```json
{
  "id": "call_123",
  "type": "function",
  "function": {
    "name": "calc",
    "arguments": "{\"expr\":\"2+3\"}"
  }
}
```

这个边界很重要：Agent loop 不应该知道某家 provider 的 `function.arguments` 字段长什么样。

先跑起来：

```bash
cp agent/.env.example agent/.env
# edit agent/.env and put your real key in LLM_API_KEY
python3 agent/minimal_agent.py
python3 agent/minimal_agent.py "现在几点？再算一下 18*7"
```

`agent/.env` 放运行时配置，例如：

```env
LLM_PROVIDER=openai-compatible
LLM_API_KEY=sk-your-key-here
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-flash
MAX_STEPS=6
SHOW_PROVIDER_META=1
```

命令行环境变量优先级高于 `.env`，所以可以临时覆盖：

```bash
LLM_MODEL=another-model python3 agent/minimal_agent.py
```

## 10 次迭代计划

### Iteration 1: 最小 Agent 循环

目标：彻底看懂一个 Agent 最核心的运行机制。

要实现：

- 手写 `messages` 列表。
- 调一次 LLM，判断有没有 `tool_calls`。
- 有工具调用就执行本地函数。
- 把工具结果作为 `role=tool` append 回 `messages`。
- 没有工具调用就返回最终答案。

要理解的问题：

- LLM 不执行工具，只输出工具调用请求。
- 真正执行工具的是你的 Python 代码。
- `messages` 就是当前最简单的上下文和短期记忆。
- `tool_call_id` 是把工具请求和工具结果对齐的协议字段。

完成标准：

- 能解释 `system -> user -> assistant.tool_calls -> tool -> assistant final` 的完整链路。
- 能看懂 `dump()` 打印出来的每一步 messages。

### Iteration 2: 配置和 Provider 兼容

目标：把容易变化的配置从代码里移出去。

要实现：

- `.env` 文件读取。
- `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`、`MAX_STEPS` 配置化。
- 命令行环境变量优先于 `.env`。
- 不把真实 key 提交，只提交 `.env.example`。
- 定义 `Provider.complete(messages, tools)` 接口。
- 实现第一版 `OpenAICompatibleProvider`。
- Agent 主循环只调用 provider，不直接拼 HTTP 请求。

要理解的问题：

- 当前代码支持的是 `openai-compatible` Chat Completions 协议，不是任意 provider 协议。
- OpenAI-compatible API 只是协议兼容，不代表所有字段完全一致。
- 不同服务可能返回额外字段，比如 `reasoning_content`、`index`。
- 有些 provider 支持 `tools`，有些只支持普通 chat，有些 function calling 格式略有差异。
- URL、模型名、鉴权方式都不应该散落在业务代码中。

完成标准：

- 换 provider 时不改 agent 循环，只改 `.env`。
- 新增非 OpenAI-compatible provider 时，只增加新的 adapter 类和 `build_provider()` 分支。
- 缺少 API key 时给出清楚错误。

### Iteration 3: 消息规范化和协议边界

目标：不要把 provider 私有字段无脑塞回下一轮请求。

要实现：

- 在 provider adapter 里把 provider 原始响应转换成内部 assistant message。
- Agent 内部只保留标准字段：`role`、`content`、`tool_calls`。
- 对 `content=None`、`content=""` 做兼容处理。
- 对 provider 返回的 `reasoning_content` 做日志记录，但不一定放回请求。
- 用 `SHOW_PROVIDER_META=0` 可以关闭 provider 私有字段日志。
- OpenAI-compatible 的 `function.name` / `function.arguments` 只存在于 adapter 内部。

要理解的问题：

- API response 和下一轮 request message 不是永远同一个 schema。
- 某些服务允许额外字段，某些服务会因为额外字段报错。
- 工程化 Agent 要在“模型返回格式”和“内部状态格式”之间做边界转换。
- 标准化不应该是脱离 provider 的全局函数，而应该是每个 provider adapter 的职责。

完成标准：

- 当前 DeepSeek/OpenAI-compatible 输出能正常跑。
- 换到更严格的 OpenAI-compatible 服务时，不因为多余字段失败。

### Iteration 4: 工具注册系统

目标：从手写 `FUNCS` 和 `TOOLS` 进化到可维护的工具注册。

要实现：

- 定义 `Tool` 数据结构，包含 `name`、`description`、`parameters`、`handler`。
- 用装饰器或注册函数添加工具。
- 启动时自动生成 LLM 需要的 tool schema。
- 工具执行统一走 `run_tool(call)`。

要理解的问题：

- 工具有两面：给 LLM 看的 schema，给 Python 执行的 handler。
- 参数 schema 越清楚，模型越容易正确调用工具。
- 工具名是协议的一部分，改名会影响模型调用。
- 工具结果最好统一格式，例如 `{"ok": true, "result": ...}`。

完成标准：

- 新增一个工具只需要写函数和注册，不需要改 agent 主循环。

### Iteration 5: 错误处理、重试和终止条件

目标：让 Agent 不会因为一次工具错误或 API 错误直接崩掉，也不会无限循环。

要实现：

- 区分工具错误、参数 JSON 解析错误、未知工具错误、LLM API 错误。
- 工具错误回传给 LLM，让模型有机会修正。
- API 错误可重试，设置最大重试次数。
- 增加终止原因：`final_answer`、`max_steps`、`api_error`、`tool_error_limit`。

要理解的问题：

- `MAX_STEPS` 限制的是 LLM 决策轮数，不是工具调用次数。
- 一次 LLM response 可能包含多个 tool calls。
- 错误是否回传给 LLM，是 Agent 行为设计的一部分。
- 无限工具调用是 Agent 最常见的问题之一。

完成标准：

- 故意让 `calc("sqrt(9)")` 报错时，模型能看到错误并尝试修正。
- 超过最大步数时能清楚打印为什么停止。

### Iteration 6: 上下文管理和 Memory

目标：解决 `messages` 无限增长的问题。

要实现：

- 估算 messages 长度，至少按字符数或粗略 token 数统计。
- 保留 system message、最近 N 轮消息。
- 对旧消息做 summary。
- 截断过长的工具结果。
- 把大结果存到外部文件，只在 messages 里放摘要或引用。

要理解的问题：

- 模型没有隐藏记忆，每次请求都要重新发送上下文。
- 上下文太长会增加成本、延迟，并可能超出窗口限制。
- 工具返回内容经常比用户问题大得多，必须治理。
- Memory 不是简单“全都塞进去”，而是选择什么进入上下文。

完成标准：

- 连续多轮工具调用后，messages 不会无限变大。
- 打印日志能看出哪些内容被保留、哪些被摘要、哪些被截断。

### Iteration 7: 状态对象和可持久化 Run

目标：把 `messages` 从局部变量升级成明确的 `AgentState`。

要实现：

- 定义 `AgentState`，包含 `run_id`、`messages`、`step`、`status`、`created_at`、`updated_at`。
- 每一步结束后把 state 保存到 JSON 文件。
- 支持从已有 state 文件恢复运行。
- 保存工具调用记录、错误记录、最终答案。

要理解的问题：

- 真实 Agent 不只是 messages，还有运行状态。
- 持久化是暂停、恢复、审计、调试的基础。
- LangGraph 里的 state，本质上就是把 while 循环里的隐式变量显式化。

完成标准：

- 运行中断后，可以从 state 文件继续。
- 可以打开 run 文件复盘每一步发生了什么。

### Iteration 8: 暂停、恢复和人工介入

目标：实现 LangGraph 类框架最核心的价值之一：人可以插入循环。

要实现：

- 给工具加风险等级，例如 `safe`、`needs_approval`。
- 高风险工具调用前暂停，输出待审批信息。
- 人工批准后继续执行工具。
- 人工拒绝后把拒绝原因作为工具结果回传给 LLM。

要理解的问题：

- Agent 的循环不一定连续跑完，中间可能要等人。
- 暂停恢复需要明确的状态和 tool call id。
- 人工介入本质上也是一种消息或状态更新。
- 发邮件、删文件、下单、写数据库这类工具通常需要审批。

完成标准：

- 遇到需要审批的工具时，程序停止在 `paused` 状态。
- 修改 state 为批准后，可以继续从原位置运行。

### Iteration 9: 并发工具、队列和结果对齐

目标：理解为什么 `tool_call_id` 很重要。

要实现：

- 一次 assistant response 中多个 tool calls 并发执行。
- 支持工具超时。
- 工具完成顺序可以和请求顺序不同。
- 结果仍然通过 `tool_call_id` 对齐回原始调用。

要理解的问题：

- 只靠数组顺序对齐工具结果，在并发场景里很脆。
- 同一个工具可能被调用多次，但参数不同。
- 某些工具成功、某些工具失败、某些工具超时时，必须保持每个 call 的结果明确。
- `tool_call_id` 就是“题号”，tool result 是带题号的答案。

完成标准：

- 故意让慢工具和快工具乱序完成，最终答案仍然正确。
- 日志里能清楚看到每个 `call_id -> result` 的映射。

### Iteration 10: 观测、测试和小型框架化

目标：把学习版 Agent 整理成一个可继续扩展的小框架。

要实现：

- 结构拆分：`config.py`、`llm.py`、`tools.py`、`state.py`、`agent.py`。
- 日志从 `print` 升级成结构化 event log。
- 增加单元测试：工具执行、消息规范化、错误回传、max steps。
- 增加 mock LLM 测试，不依赖真实 API。
- 输出 run trace，方便复盘每一步。

要理解的问题：

- Agent 框架的复杂度主要来自工程边界，不是 while 循环本身。
- 测试 Agent 要把 LLM 调用 mock 掉，否则结果不稳定且成本高。
- 观测能力决定你能不能调试 Agent。
- smolagents 是把工具、模型、执行器工程化；LangGraph 是把循环改成状态图。

完成标准：

- 不联网也能跑核心单元测试。
- 真实运行时能得到完整 trace。
- 读 smolagents 和 LangGraph 源码时，能把它们的概念映射回自己的实现。

## 学习时重点观察

每次迭代都建议打印或保存这些内容：

- 发给 LLM 的 `messages`。
- LLM 返回的原始 message。
- 标准化后的 assistant message。
- 每个 `tool_call_id` 对应的工具名、参数、结果、耗时、错误。
- 当前 step、终止原因、state 状态。

真正要掌握的不是某个库，而是这些问题：

- 模型看到的上下文是什么？
- 工具 schema 是怎么影响模型行为的？
- 工具调用失败时，错误应该给谁看？
- messages 什么时候会太长？
- 为什么要有 state？
- 为什么暂停恢复需要持久化？
- 为什么并发工具必须靠 `tool_call_id` 对齐？
- 什么东西适合放进 prompt，什么东西应该放进代码？

走完这 10 步之后，再去读 smolagents，你会看到它是在工程化工具和执行循环；再去读 LangGraph，你会看到它是在把循环里的变量变成显式状态，把 `while` 拆成可暂停、可恢复、可分支的图。
