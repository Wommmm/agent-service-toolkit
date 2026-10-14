# RAG Agent接入项目一MCP检索

`src/agents/retrieval_mcp.py`中的LangChain工具`knowledge_search`作为MCP Client调用项目一的`search_knowledge`。它提取文档列表并返回JSON字符串，随后由ToolNode生成工具消息，供模型回答。

## 配置与运行

先启动项目一的MCP Server（命令及检索依赖见其`docs/mcp-retrieval.md`）。当前工具连接地址为`http://127.0.0.1:8002/mcp`，暂未配置化。

在本项目本地`.env`中设置：

```dotenv
RAG_RETRIEVAL_BACKEND=mcp
```

`chroma`为默认值，使用原员工手册检索；`mcp`选择产品知识库工具及对应提示词。配置在模块加载时选择，修改后需重新启动进程。两种知识库的问题与评测用例不可混为同一批。

项目二沿用现有服务启动方式。若在终端检查工具或运行测试，需要让Python识别src目录：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
.\.venv\Scripts\python.exe -m pytest tests\agents\test_retrieval_mcp.py tests\agents\test_rag_assistant_tool_rounds.py tests\test_eval_metrics.py -q
```

## 返回与失败处理

- `isError=True`或缺少结构化数据时，工具抛出RuntimeError，不把错误文本返回为参考资料。
- 成功但文档列表为空时，返回JSON字符串`[]`，区别于执行失败。
- 返回的文档字符串不是最终答案；ToolNode把它加入消息历史，再由模型生成答案或决定继续调用工具。

RAG Agent的ToolNode配置了`handle_retrieval_error`：工具抛出RuntimeError时，转换为`status="error"`的ToolMessage，内容为“知识库检索暂时不可用，请稍后重试”。直接调用`knowledge_search.ainvoke()`仍会抛异常；上述转换只发生在配置了该处理函数的ToolNode中。其他类型的异常仍按当前框架处理机制传播。

2026-10-11新增最小图测试，使用真实工具与错误处理函数、替身MCP会话，验证错误消息状态、内容、调用ID、MCP参数和会话关闭。它只验证工具节点错误转换，不证明真实连接失败已全面覆盖，也不验证后续模型回答。运行MCP测试和轮数测试得到`6 passed, 4 warnings in 8.70s`。

4项MCP工具边界检查使用替身会话，不证明真实网络或数据库连接；与原有轮数及指标测试一起执行为9项通过。真实接入证据单独保存在`evals/mcp_integration_report.json`。

## 单题真实验证（2026-10-08）

问题：RS-12如何进行电池测试？模型为deepseek-v4-flash，执行3次knowledge_search，每次返回3条文档，工具消息均success，最终返回测试步骤和阈值；耗时82.09秒。

这是单题调用链验证，尚无通用准确率结论。检索混入其他产品、耗时较高仍待改进。Safeguard因未配置GROQ_API_KEY而跳过，本报告不能作为输入安全检查证据。
