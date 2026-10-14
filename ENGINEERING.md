# Agent工程升级

本版在原Agent Service Toolkit与已有RAG/MCP改造上增加执行控制和可检查证据。保留原注册、Chroma、界面、checkpointer、invoke/stream、AG-UI与Supervisor示例。核心仍是一个RAG Agent，不宣称重新实现原框架。

## 调用流程

请求 → 取得Agent和thread配置 → RAG本次提问初始化 → 有效上下文窗口 → model → 工具预算检查 → knowledge_search → HTTP MCP → 独立检索服务 → ToolMessage（含artifact）→ model → 引用成员校验与结构化结果 → 接口消息＋运行摘要。

## 本版行为

- 默认保留本地Chroma；设RAG_RETRIEVAL_BACKEND=mcp接入独立服务。
- MCP模式默认首轮使用tool_choice=any强制取证据，避免模型直接复用旧轮文档；可设RAG_REQUIRE_FIRST_RETRIEVAL=false恢复自由选择。此模式定位产品知识问答，普通问候也可能触发检索；模型需支持强制工具选择，当前真实验证使用OpenAI兼容模型。
- MCP URL、工具名、Bearer token、超时与top_k配置化，默认127.0.0.1:8050/mcp/。v1响应严格校验版本、状态、标识和文档；旧8002服务必须显式选legacy且配置旧URL。
- 成功、空资料、需澄清、部分降级保留不同状态；SDK/网络错误不给用户暴露远端body、认证头或密钥。
- 每次新HumanMessage初始化预算，同thread的新问题重新获得预算。工具轮数与工具执行尝试数分别计算；拒绝执行的调用不占尝试数，故障尝试占预算。一轮多个调用不能绕过数量上限。
- 本次RAG模型/工具调用都有时限，invoke与stream另有总等待时限。超时停止本地等待，不保证下游计算或计费立即停止。
- 保留完整checkpoint消息；只构建较短的模型输入窗口，删除整个旧用户轮次，不截断当前工具调用/结果配对。工具文本按字符长度预算截断并保留ID与截断标记。当前版本没有增加LLM历史摘要、长期记忆服务或精确token窗口控制。
- 模型使用[doc:完整文档id]标记引用。程序整理结构化结果并检查ID/URL来自允许的MCP证据；无效引用替换为明确无法核对的消息，缺引用标为unverified。此检查不证明文档支持结论或事实正确，不新增一次格式化模型调用。
- 旧Chroma文本结果无v1文档ID时保持原回答，标注unstructured_legacy_evidence，不能宣称同样完成引用校验。
- 最终ChatMessage仍有content；custom_data.agent_result提供状态、引用、检索执行ID、轮数、尝试数、模型调用与可用usage。流式tokens是校验前内容，finalize后的message是最终检查结果，原SSE类型与[DONE]保持。

## 配置与运行

复制.env.example到私密.env，按原运行方式填写模型配置，并设置：

```dotenv
RAG_RETRIEVAL_BACKEND=mcp
RETRIEVAL_MCP_URL=http://127.0.0.1:8050/mcp/
RETRIEVAL_MCP_CONTRACT=v1
RETRIEVAL_MCP_TOKEN=
MAX_TOOL_ROUNDS=3
MAX_TOOL_CALLS=5
AGENT_REQUEST_TIMEOUT_SECONDS=180
AGENT_MODEL_TIMEOUT_SECONDS=60
RETRIEVAL_MCP_TIMEOUT_SECONDS=120
RUN_RECORDS_ENABLED=true
RUN_RECORDS_DB=data/agent_runs.sqlite3
```

项目二只需要MCP SDK和轻量消费方契约，不安装项目一的模型/数据库服务依赖。独立检索服务另行运行，真实模式需要它已有的数据库、模型及私密配置；demo不代表真实质量。

沿用原Python运行或Docker方式。若复用其他目录的解释器，PYTHONPATH必须指向本版src；可先打印agents.retrieval_mcp.__file__验证实际导入位置。使用本项目uv.lock，不为本次升级改动整套依赖。

也可把专用配置放在忽略提交的.env.engineering，运行python scripts/run_engineering.py；它定位本版src并使用原服务入口。python scripts/run_engineering.py --ui使用相同配置启动原Streamlit，UI默认8502。专用配置的HOST/PORT、AGENT_URL与AUTH_SECRET要一致，环境变量优先于该文件。

## 运行记录

GET /runs/{run_id}查询本地摘要，沿用原路由的共享Bearer验证；不是多用户隔离的审计系统。建议部署前设置AUTH_SECRET，记录默认只含诊断元数据，不存问题、答案、完整文档或密钥。SQLite文件独立于checkpointer，保留条数有上限，写记录失败不阻断已成功回答。

保存模型时长、工具时长/批次时长、来源ID、成功响应的检索request_id/execution_id及阶段摘要。远端故障未返回标识时为null，不编造关联；不保存远端错误body。模型未提供usage时为null，不填0，不提供虚构账单金额。统计仅覆盖本RAG模型循环；可选Safeguard的供应商用量不包含在该聚合中。

invoke：未知Agent404、未加载503、检索/模型不可用503、超时504、当前轮上下文过大413；无资料正常200；达到工具预算返回200且agent_result.status=budget_exceeded。stream已开始后使用error事件报告故障，不能更改此前HTTP200。诊断接口继承现有共享认证，不代表thread/user已实现可信归属。

## 评测与验证

```powershell
python -m evals.run_engineering_eval --mode simulated --report evals/engineering-simulated-report.json
python -m evals.run_engineering_eval --mode live --report evals/engineering-live-report.json
```

专用产品集与原员工手册cases.yaml分开，单题失败后继续其余题。评测通过ASGI TestClient调用真实FastAPI路由和RAG图；live进一步调用真实HTTP MCP与模型，API本身未通过独立TCP服务进程压测。两问使用同thread，工具预算均设为1，用于验证新请求预算不受历史污染。测试用checkpoint是InMemorySaver，正式服务仍沿用已有SQLite/Postgres/Mongo初始化方式。

simulated替换模型与检索依赖，覆盖正常、空结果、澄清、部分降级、故障、错误引用与干扰文档。它验证状态/调用路径，不能证明真实检索质量或提示注入防御。live仅使用RS-12电池与温度两题；关键词与引用成员校验不等于完整事实评判，没有通用准确率或成本下降结论。

本版没有默认添加备用模型、自动检索重试、复杂摘要、队列/Redis、多租户身份隔离、同thread并发锁或新的多Agent业务。AG-UI原接口保留，新增运行记录及服务超时主要覆盖invoke/stream；RAG图中的工具预算与局部调用时限也适用于直接执行该图。框架弃用/序列化警告不为升级主线临时迁移版本。

## 2026-10-14实测记录

- 服务层与相关Agent/评测指标回归：110 passed、18条既有框架警告，4.45秒；不是110次真实模型业务调用。
- simulated报告8/8场景符合预期，依赖和模型为替换实现，不能用于宣称真实准确率或抵抗注入。
- 首次live同会话两题只有1/2通过：第二题复用旧轮资料，当前引用无法核对；保留engineering-live-before-policy-report.json。随后补强首轮强制检索。
- 修正后live同thread两题均200、各1次工具执行并有本次有效引用，22.627秒和20.838秒；见engineering-live-report.json。开启默认缓存，不作为无缓存延迟基准或新一轮消融实验。
- 另在真实TCP服务与SQLite checkpoint上验证一题：API认证拒绝、info、实际invoke、引用及运行记录查询7项检查均通过，总22.485秒。它使用真实模型及真实HTTP MCP，不是ASGI替换依赖；详细收据保存在本地工程交付记录。

结果只覆盖这些具体场景和小样本，不代表Supervisor全部业务已实测，也不证明一般质量、成本或吞吐改善。
