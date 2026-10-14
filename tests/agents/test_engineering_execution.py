import asyncio
import importlib
import json
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode

from agents.execution_policy import ContextBudgetExceeded, current_turn, finalize_result, prepare_messages
from agents.retrieval_mcp import knowledge_search, RetrievalFailure

rag = importlib.import_module("agents.rag_assistant")
retrieval = importlib.import_module("agents.retrieval_mcp")


class ScriptedModel(RunnableLambda):
    def __init__(self, always_call=False, parallel=1, delay=0):
        self.seen = []
        self.bound = 0
        async def respond(messages):
            self.seen.append(messages)
            await asyncio.sleep(delay)
            recent = current_turn(messages)
            if always_call or not any(isinstance(msg, ToolMessage) for msg in recent):
                return AIMessage(content="", tool_calls=[{"name": "knowledge_search", "args": {"query": "RS-12电池"}, "id": str(uuid4())} for _ in range(parallel)])
            tool = next(msg for msg in reversed(recent) if isinstance(msg, ToolMessage))
            if tool.status == "error":
                return AIMessage(content="检索暂时不可用。")
            packet = json.loads(tool.content)
            if packet["status"] == "clarification":
                return AIMessage(content="请提供型号。")
            if not packet["documents"]:
                return AIMessage(content="没有找到资料。")
            return AIMessage(content="RS-12使用9V电池。[doc:battery-1]", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
        super().__init__(respond)
    def bind_tools(self, tools, **kwargs):
        self.bound += 1
        self.last_tool_choice = kwargs.get("tool_choice")
        return self


@pytest.fixture
def graph_setup(monkeypatch):
    model = ScriptedModel()
    state = {"status": "ok", "failure": None, "delay": 0}
    async def fetch(query):
        await asyncio.sleep(state["delay"])
        if state["failure"]:
            raise RetrievalFailure(state["failure"])
        return {"schema_version": "1.0", "request_id": "r1", "execution_id": "e1", "backend": "live",
                "status": state["status"], "warnings": [], "message": "",
                "documents": [{"id": "battery-1", "content": "9V", "source": "local"}] if state["status"] in ("ok", "partial") else []}
    monkeypatch.setattr(retrieval, "fetch_retrieval", fetch)
    monkeypatch.setattr(rag, "tool_node", ToolNode([knowledge_search], handle_tool_errors=rag.handle_retrieval_error))
    monkeypatch.setattr(rag, "get_model", lambda name: model)
    monkeypatch.setattr(rag.settings, "GROQ_API_KEY", None)
    monkeypatch.setattr(rag.settings, "RAG_RETRIEVAL_BACKEND", "mcp")
    monkeypatch.setattr(rag.rag_assistant, "checkpointer", InMemorySaver())
    return model, state


def config(thread=None):
    return {"configurable": {"thread_id": thread or str(uuid4())}}


@pytest.mark.asyncio
async def test_full_graph_artifact_citation_and_new_turn_budget(graph_setup, monkeypatch):
    monkeypatch.setattr(rag.settings, "MAX_TOOL_ROUNDS", 1)
    monkeypatch.setattr(rag.settings, "MAX_TOOL_CALLS", 1)
    thread = config()
    first = await rag.rag_assistant.ainvoke({"messages": [HumanMessage(content="RS-12电池?")]}, thread)
    second = await rag.rag_assistant.ainvoke({"messages": [HumanMessage(content="再次问电池?")]}, thread)
    for response in [first, second]:
        result = response["execution_result"]
        assert result["tool_calls"] == 1
        assert result["tool_rounds"] == 1
        assert result["answer_status"] == "answered"
        assert result["citations"][0]["id"] == "battery-1"
        assert result["tool_events"][0]["execution_id"] == "e1"
    assert first["execution_id"] != second["execution_id"]


@pytest.mark.asyncio
async def test_one_ai_round_cannot_bypass_call_budget(graph_setup, monkeypatch):
    model, _ = graph_setup
    model.__init__(parallel=3)
    monkeypatch.setattr(rag.settings, "MAX_TOOL_CALLS", 1)
    response = await rag.rag_assistant.ainvoke({"messages": [HumanMessage(content="电池?")]}, config())
    assert response["tool_calls_used"] == 1
    assert response["execution_result"]["status"] == "budget_exceeded"
    assert len([msg for msg in current_turn(response["messages"]) if isinstance(msg, ToolMessage)]) == 3


@pytest.mark.asyncio
async def test_model_ignoring_disabled_tools_stops(graph_setup, monkeypatch):
    model, _ = graph_setup
    model.__init__(always_call=True)
    monkeypatch.setattr(rag.settings, "MAX_TOOL_ROUNDS", 1)
    response = await rag.rag_assistant.ainvoke({"messages": [HumanMessage(content="电池?")]}, config())
    assert response["execution_result"]["status"] == "budget_exceeded"
    assert response["execution_result"]["tool_calls"] == 1
    assert len(model.seen) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("status,expected", [("empty", "no_evidence"), ("clarification", "clarification"), ("partial", "answered")])
async def test_successful_nonstandard_retrieval_states(graph_setup, status, expected):
    _, state = graph_setup
    state["status"] = status
    response = await rag.rag_assistant.ainvoke({"messages": [HumanMessage(content="电池?")]}, config())
    assert response["execution_result"]["answer_status"] == expected


@pytest.mark.asyncio
async def test_model_timeout_is_explicit_and_unknown_usage(graph_setup, monkeypatch):
    model, _ = graph_setup
    model.__init__(delay=.1)
    monkeypatch.setattr(rag.settings, "AGENT_MODEL_TIMEOUT_SECONDS", .01)
    response = await rag.rag_assistant.ainvoke({"messages": [HumanMessage(content="电池?")]}, config())
    assert response["execution_result"]["status"] == "timeout"
    assert response["execution_result"]["usage"] is None


@pytest.mark.asyncio
async def test_tool_timeout_finishes_with_all_call_ids(graph_setup, monkeypatch):
    _, state = graph_setup
    state["delay"] = .1
    monkeypatch.setattr(rag.settings, "RETRIEVAL_MCP_TIMEOUT_SECONDS", .01)
    response = await rag.rag_assistant.ainvoke({"messages": [HumanMessage(content="电池?")]}, config())
    assert response["execution_result"]["status"] == "timeout"
    assert response["tool_calls_used"] == 1
    assert response["model_events"][-1]["status"] == "ok"


def test_context_window_removes_complete_old_turns_only():
    old = HumanMessage(content="x" * 8000)
    active = [HumanMessage(content="电池?"), AIMessage(content="", tool_calls=[{"name": "knowledge_search", "args": {"query": "电池"}, "id": "c1"}]), ToolMessage(content="9V", tool_call_id="c1")]
    prepared = prepare_messages([old, AIMessage(content="旧答案"), *active], "系统规则", 2000)
    assert prepared[1:] == active
    assert prepared[0].type == "system"
    with pytest.raises(ContextBudgetExceeded):
        prepare_messages([HumanMessage(content="x" * 8000)], "系统规则", 2000)


@pytest.mark.parametrize("content", ["9V [doc:nonexistent]", "9V https://invalid.example/key"])
def test_invalid_references_do_not_pass_through(content):
    tool = ToolMessage(content="data", tool_call_id="c1", artifact={"schema_version": "1.0", "documents": [{"id": "battery-1", "source": "local", "content": "9V"}]})
    result = finalize_result({"messages": [HumanMessage(content="电池"), tool, AIMessage(content=content)]})
    assert result["execution_result"]["answer_status"] == "unverified"
    assert "nonexistent" not in result["messages"][-1].content
    assert "invalid.example" not in result["messages"][-1].content


def test_chroma_text_evidence_preserves_original_links_but_is_unverified():
    tool = ToolMessage(content="See https://example.org/handbook", tool_call_id="c1")
    answer = "15 days https://example.org/handbook"
    result = finalize_result({"messages": [HumanMessage(content="PTO"), tool, AIMessage(content=answer)]})
    assert result["messages"][-1].content == answer
    assert result["execution_result"]["citation_validation"] == "not_available"
    assert result["execution_result"]["answer_status"] == "unverified"


@pytest.mark.asyncio
async def test_guarded_request_is_recorded_as_blocked_without_model_or_tool(graph_setup, monkeypatch):
    from agents.safeguard import SafeguardOutput, SafetyAssessment
    class BlockedGuard:
        async def ainvoke(self, messages):
            return SafeguardOutput(safety_assessment=SafetyAssessment.UNSAFE, unsafe_categories=["test-category"])
    monkeypatch.setattr(rag, "Safeguard", BlockedGuard)
    model, _ = graph_setup
    response = await rag.rag_assistant.ainvoke({"messages": [HumanMessage(content="测试拦截")]}, config())
    assert response["execution_result"]["status"] == "blocked"
    assert response["execution_result"]["tool_calls"] == 0
    assert model.seen == []
