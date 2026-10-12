"""Real RAG graph and ToolNode; only external providers/guard are substituted."""

import asyncio
import importlib.util
import sys
import types
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool


@pytest.fixture
def load_rag(monkeypatch):
    def load(responses, tool_error=False, max_rounds=2):
        root = Path(__file__).resolve().parents[2] / "src" / "agents"
        package = types.ModuleType("agents")
        package.__path__ = [str(root)]
        monkeypatch.setitem(sys.modules, "agents", package)
        calls = []

        @tool("Database_Search")
        def search(query: str) -> str:
            """Search the test knowledge base."""
            calls.append(query)
            if tool_error:
                raise RuntimeError("private-provider-token")
            return "test evidence"

        class SafetyAssessment(Enum):
            SAFE = "safe"
            UNSAFE = "unsafe"

        class Safeguard:
            async def ainvoke(self, messages):
                return SimpleNamespace(safety_assessment=SafetyAssessment.SAFE)

        guard = types.ModuleType("agents.safeguard")
        guard.Safeguard = Safeguard
        guard.SafeguardOutput = SimpleNamespace
        guard.SafetyAssessment = SafetyAssessment
        external_tools = types.ModuleType("agents.tools")
        external_tools.database_search = search
        outputs = iter(responses)
        model = RunnableLambda(lambda messages: next(outputs))
        model.bind_tools = lambda tools: model
        core = types.ModuleType("core")
        core.get_model = lambda name: model
        core.settings = SimpleNamespace(MAX_TOOL_ROUNDS=max_rounds, DEFAULT_MODEL="fake")
        for name, module in [
            ("agents.safeguard", guard),
            ("agents.tools", external_tools),
            ("core", core),
        ]:
            monkeypatch.setitem(sys.modules, name, module)
        # Import the actual production graph without importing the provider registry.
        name = "rag_engineering_under_test"
        spec = importlib.util.spec_from_file_location(name, root / "rag_assistant.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module, calls

    return load


def tool_call(call_id="call-1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": "Database_Search", "args": {"query": "question"}, "id": call_id}],
    )


def test_round_budget_resets_at_new_user_turn(load_rag):
    module, _ = load_rag([])
    old = [HumanMessage(content="old"), tool_call(), AIMessage(content="old answer")]
    assert module.count_tool_rounds(old) == 1
    assert module.count_tool_rounds(old + [HumanMessage(content="new")]) == 0
    assert module.count_tool_rounds(old + [HumanMessage(content="new"), tool_call("new")]) == 1


def test_multiple_calls_in_one_message_are_one_round(load_rag):
    module, _ = load_rag([])
    message = tool_call()
    message.tool_calls.append(
        {"name": "Database_Search", "args": {"query": "other"}, "id": "call-2", "type": "tool_call"}
    )
    assert module.count_tool_rounds([HumanMessage(content="question"), message]) == 1


def test_uncooperative_model_cannot_exceed_tool_budget(load_rag):
    module, calls = load_rag([tool_call("1"), tool_call("2"), tool_call("3")])
    result = asyncio.run(
        module.rag_assistant.ainvoke(
            {"messages": [HumanMessage(content="question")]},
            {"configurable": {}, "recursion_limit": 20},
        )
    )
    assert len(calls) == 2
    assert result["messages"][-1].tool_calls == []
    assert "工具调用上限" in result["messages"][-1].content


def test_tool_failure_is_error_message_and_returns_to_model(load_rag):
    module, calls = load_rag(
        [tool_call(), AIMessage(content="temporarily unavailable")], tool_error=True
    )
    result = asyncio.run(
        module.rag_assistant.ainvoke(
            {"messages": [HumanMessage(content="question")]}, {"configurable": {}}
        )
    )
    errors = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert calls == ["question"]
    assert errors[0].status == "error"
    assert errors[0].tool_call_id == "call-1"
    assert "private-provider-token" not in errors[0].content
    assert result["messages"][-1].content == "temporarily unavailable"


def test_new_turn_with_old_history_can_still_search(load_rag):
    module, calls = load_rag([tool_call("new"), AIMessage(content="new answer")], max_rounds=1)
    history = [
        HumanMessage(content="old"),
        tool_call("old"),
        ToolMessage(content="old result", tool_call_id="old"),
        AIMessage(content="old answer"),
    ]
    result = asyncio.run(
        module.rag_assistant.ainvoke(
            {"messages": history + [HumanMessage(content="new")]}, {"configurable": {}}
        )
    )
    assert len(calls) == 1
    assert result["messages"][-1].content == "new answer"
