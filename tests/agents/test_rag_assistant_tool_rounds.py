from langchain_core.messages import AIMessage, HumanMessage

from agents.rag_assistant import count_tool_rounds


def test_count_tool_rounds_counts_only_ai_messages_with_tool_calls() -> None:
    messages = [
        HumanMessage(content="How many vacation days do I have?"),
        AIMessage(content="I need to search the handbook.", tool_calls=[]),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "Database_Search",
                    "args": {"query": "vacation days"},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        ),
    ]

    assert count_tool_rounds(messages) == 1
