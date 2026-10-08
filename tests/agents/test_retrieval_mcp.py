import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from agents import retrieval_mcp


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
    result = await retrieval_mcp.knowledge_search.ainvoke({"query": "用户问题"})
    assert json.loads(result) == documents
    assert mcp_reply["calls"] == [("search_knowledge", {"query": "用户问题"})]
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
