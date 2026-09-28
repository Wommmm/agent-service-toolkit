from langchain_core.messages import AIMessage, HumanMessage

from evals.metrics import (
    calculate_completion_rate,
    count_tool_rounds,
    get_final_answer,
    is_within_round_limit,
    matches_expected_answer,
)


def test_extracts_answer_and_counts_tool_rounds() -> None:
    messages = [
        HumanMessage(content="How many PTO days?"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search",
                    "args": {},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(content="Employees receive 15 days."),
    ]

    assert get_final_answer(messages) == "Employees receive 15 days."
    assert count_tool_rounds(messages) == 1


def test_matches_whole_number() -> None:
    assert matches_expected_answer("Employees receive 5 days.", ["5"])
    assert not matches_expected_answer("Employees receive 15 days.", ["5"])


def test_calculates_completion_rate() -> None:
    assert calculate_completion_rate([True, False]) == 0.5
    assert calculate_completion_rate([]) == 0.0


def test_checks_tool_round_limit() -> None:
    assert is_within_round_limit(1, 3)
    assert not is_within_round_limit(4, 3)
