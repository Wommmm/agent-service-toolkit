"""Tool limits are per user turn, not per checkpointed conversation."""

import logging

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

logger = logging.getLogger(__name__)


def count_tool_rounds(messages: list[AnyMessage]) -> int:
    rounds = 0
    for message in messages:
        if isinstance(message, HumanMessage):
            rounds = 0
        elif isinstance(message, AIMessage) and message.tool_calls:
            rounds += 1
    return rounds


def handle_retrieval_error(error: Exception) -> str:
    # Never include provider exceptions or credentials in model-visible errors.
    logger.warning("Retrieval tool failed", extra={"error_type": type(error).__name__})
    return "知识库检索暂时不可用，请稍后重试。"
