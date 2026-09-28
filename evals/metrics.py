import re

from langchain_core.messages import AIMessage, AnyMessage

# 测评流水线，


def get_final_answer(messages: list[AnyMessage]) -> str:
    """Returns the final AI message content"""
    for message in reversed(messages):  # 倒序查找
        if isinstance(message, AIMessage):  # 判断是否是AI消息
            return str(message.content)
    return ""


def count_tool_rounds(messages: list[AnyMessage]) -> int:
    """Counts the number of tool_calls in a list of messages"""
    tool_rounds = 0
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_rounds += 1
    return tool_rounds


def matches_expected_answer(answer: str, expected_any_of: list[str]) -> bool:
    normalized_answer = answer.casefold()

    for expected in expected_any_of:
        pattern = rf"\b{re.escape(expected.casefold())}\b"

        if re.search(pattern, normalized_answer):
            return True

    return False


def calculate_completion_rate(completed: list[bool]) -> float:
    """
    汇总多条任务的完成情况
    Summarize the completion status of multiple tasks.
    """
    if not completed:
        return 0.0
    return sum(completed) / len(completed)


def is_within_round_limit(tool_rounds: int, max_tool_rounds: int) -> bool:
    return tool_rounds <= max_tool_rounds
