import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
import httpx
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState, StateGraph
from pydantic import SecretStr

from agents import retrieval_mcp
from agents.retrieval_contracts import parse_response, model_payload


def packet(status="ok", documents=None):
    if documents is None:
        documents = [{"id": "battery-1", "content": "RS-12使用9V电池", "source": "local", "title": "说明书"}] if status in ("ok", "partial") else []
    return {"schema_version": "1.0", "request_id": "request-1", "execution_id": "execution-1",
            "backend": "live", "status": status, "documents": documents,
            "warnings": ["web_unavailable"] if status == "partial" else [], "message": "请提供型号" if status == "clarification" else ""}


@pytest.fixture
def transport(monkeypatch):
    state = {"payload": packet(), "error": False, "delay": 0, "closed": False, "calls": []}
    monkeypatch.setattr(retrieval_mcp.settings, "RETRIEVAL_MCP_CONTRACT", "v1")
    monkeypatch.setattr(retrieval_mcp.settings, "RETRIEVAL_MCP_URL", "http://127.0.0.1:8050/mcp/")
    monkeypatch.setattr(retrieval_mcp.settings, "RETRIEVAL_MCP_TOKEN", SecretStr("test-token"))

    @asynccontextmanager
    async def client(*, url, http_client):
        assert url == retrieval_mcp.settings.RETRIEVAL_MCP_URL
        assert http_client.headers["Authorization"] == "Bearer test-token"
        yield "read", "write", None

    class Session:
        def __init__(self, *args):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            state["closed"] = True
        async def initialize(self):
            pass
        async def call_tool(self, name, arguments):
            state["calls"].append((name, arguments))
            await asyncio.sleep(state["delay"])
            return SimpleNamespace(isError=state["error"], structuredContent=state["payload"], content=["private remote body test-token"])

    monkeypatch.setattr(retrieval_mcp, "streamable_http_client", client)
    monkeypatch.setattr(retrieval_mcp, "ClientSession", Session)
    return state


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["ok", "empty", "partial", "clarification"])
async def test_status_and_execution_identifiers_survive(transport, status):
    transport["payload"] = packet(status)
    result = json.loads(await retrieval_mcp.knowledge_search.ainvoke({"query": "RS-12"}))
    assert result["status"] == status
    assert result["request_id"] == "request-1"
    assert result["execution_id"] == "execution-1"
    assert result["warnings"] == transport["payload"]["warnings"]
    assert transport["calls"] == [("search_knowledge", {"query": "RS-12", "top_k": 5})]
    assert transport["closed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [None, {"result": []}, {**packet(), "schema_version": "2.0"}, packet("empty", [{"id": "bad", "content": "x", "source": "local"}])])
async def test_invalid_contract_is_not_successful_empty(transport, payload):
    transport["payload"] = payload
    with pytest.raises(retrieval_mcp.RetrievalFailure):
        await retrieval_mcp.fetch_retrieval("RS-12")
    assert transport["closed"]


@pytest.mark.asyncio
async def test_remote_error_does_not_expose_body_or_token(transport):
    transport["error"] = True
    with pytest.raises(retrieval_mcp.RetrievalFailure) as error:
        await retrieval_mcp.fetch_retrieval("RS-12")
    assert str(error.value) == "retrieval_unavailable"
    assert "test-token" not in str(error.value)


@pytest.mark.asyncio
async def test_timeout_closes_session(transport, monkeypatch):
    transport["delay"] = .1
    monkeypatch.setattr(retrieval_mcp.settings, "RETRIEVAL_MCP_TIMEOUT_SECONDS", .01)
    with pytest.raises(retrieval_mcp.RetrievalFailure, match="retrieval_timeout"):
        await retrieval_mcp.fetch_retrieval("RS-12")
    assert transport["closed"]


@pytest.mark.asyncio
async def test_tool_node_has_machine_readable_artifact(transport):
    node = ToolNode([retrieval_mcp.knowledge_search])
    graph = StateGraph(MessagesState)
    graph.add_node("tools", node)
    graph.set_entry_point("tools")
    graph.set_finish_point("tools")
    response = await graph.compile().ainvoke({"messages": [AIMessage(content="", tool_calls=[{"id": "call-1", "name": "knowledge_search", "args": {"query": "RS-12"}}])]})
    tool = response["messages"][-1]
    assert tool.artifact["documents"][0]["id"] == "battery-1"
    assert tool.tool_call_id == "call-1"


def test_document_budget_keeps_ids_and_reports_truncation():
    envelope = parse_response(packet(documents=[{"id": "long", "content": "x" * 6000, "source": "local"}]), "v1")
    payload = model_payload(envelope, 1000)
    assert len(payload["documents"][0]["content"]) == 1000
    assert payload["documents"][0]["id"] == "long"
    assert payload["warnings"] == ["document_text_truncated"]


def test_nested_transport_errors_have_sanitized_codes():
    request = httpx.Request("POST", "http://127.0.0.1/mcp")
    unauthorized = httpx.HTTPStatusError("private body", request=request, response=httpx.Response(401, request=request))
    assert retrieval_mcp.transport_failure_code(ExceptionGroup("transport", [unauthorized])) == "retrieval_auth_failed"
    assert retrieval_mcp.transport_failure_code(ExceptionGroup("transport", [httpx.ReadTimeout("private body")])) == "retrieval_timeout"
