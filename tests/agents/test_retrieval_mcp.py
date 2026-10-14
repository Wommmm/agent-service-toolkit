import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from langchain_core.messages import AIMessage
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from agents.rag_assistant import handle_retrieval_error
import pytest

from agents import retrieval_mcp

@pytest.mark.asyncio
async def test_tool_node_converts_retrieval_error(mcp_reply):
    mcp_reply["reply"] = SimpleNamespace(
        isError=True, structuredContent=None, content=["database unavailable"]
    )
    graph = StateGraph(MessagesState)
    tool_node = ToolNode(
        [retrieval_mcp.knowledge_search],
        handle_tool_errors = handle_retrieval_error,
    )
    graph.add_node("tools", tool_node)
    graph.set_entry_point("tools")
    graph.set_finish_point("tools")
    app = graph.compile()
    message = AIMessage(
        content="",
        tool_calls=[{
            "name": "knowledge_search",
            "args": {"query": "RS-12电池"},
            "id": "call-1",
        }],
    )
    result = await app.ainvoke({"messages":[message]})
    tool_message = result["messages"][-1]
    assert tool_message.type == "tool"
    assert tool_message.status == "error"
    assert tool_message.content == "知识库检索暂时不可用，请稍后重试"
    assert tool_message.tool_call_id == "call-1"
    assert mcp_reply["calls"] == [("search_knowledge", {"query": "RS-12电池"})]
    assert mcp_reply["closed"]
@pytest.fixture
def mcp_reply(monkeypatch):
    state = {"reply": None, "initialized": False, "closed": False, "calls": []}

    @asynccontextmanager
    async def transport(*, url):
        assert url == "http://127.0.0.1:8002/mcp"
        yield "read", "write", None

    class Session:
        def __init__(self, read, write):
            assert (read, write) == ("read", "write")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            state["closed"] = True

        async def initialize(self):
            state["initialized"] = True

        async def call_tool(self, name, arguments):
            assert state["initialized"]
            state["calls"].append((name, arguments))
            return state["reply"]

    monkeypatch.setattr(retrieval_mcp, "streamable_http_client", transport)
    monkeypatch.setattr(retrieval_mcp, "ClientSession", Session)
    return state


@pytest.mark.asyncio
@pytest.mark.parametrize("documents", [[{"content": "电池资料", "source": "local"}], []])
async def test_returns_document_json_and_preserves_empty_results(mcp_reply, documents):
    mcp_reply["reply"] = SimpleNamespace(
        isError=False, structuredContent={"result": documents}, content=[]
    )
    query = "RS-12电池"
    result = await retrieval_mcp.knowledge_search.ainvoke({"query": query})
    assert json.loads(result) == documents
    assert mcp_reply["calls"] == [("search_knowledge", {"query": query})]
    assert mcp_reply["closed"]


@pytest.mark.asyncio
async def test_tool_error_is_not_returned_as_reference_documents(mcp_reply):
    mcp_reply["reply"] = SimpleNamespace(
        isError=True, structuredContent=None, content=["database unavailable"]
    )
    with pytest.raises(RuntimeError, match="database unavailable"):
        await retrieval_mcp.knowledge_search.ainvoke({"query": "用户问题"})
    assert mcp_reply["closed"]


@pytest.mark.asyncio
async def test_missing_structured_data_is_not_treated_as_empty_search(mcp_reply):
    mcp_reply["reply"] = SimpleNamespace(
        isError=False, structuredContent=None, content=[]
    )
    with pytest.raises(RuntimeError, match="结构化数据"):
        await retrieval_mcp.knowledge_search.ainvoke({"query": "用户问题"})
    assert mcp_reply["closed"]
