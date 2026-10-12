"""Keep one failed or slow case from aborting an evaluation report."""

import asyncio
from time import perf_counter

from langchain_core.messages import HumanMessage

from evals.metrics import (
    count_tool_rounds,
    get_final_answer,
    is_within_round_limit,
    matches_expected_answer,
)


async def evaluate_case(case: dict, resolve_agent, timeout_seconds: float = 60.0) -> dict:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    started = perf_counter()
    row = {
        "id": case["id"],
        "question": case["question"],
        "answer": "",
        "tool_rounds": 0,
        "answer_matched": False,
        "within_round_limit": False,
        "completed": False,
        "error_type": None,
    }
    try:
        agent = resolve_agent(case["agent_id"])
        result = await asyncio.wait_for(
            agent.ainvoke(
                {"messages": [HumanMessage(content=case["question"])]},
                config={
                    "configurable": {"thread_id": f"eval-{case['id']}", "user_id": "eval-user"}
                },
            ),
            timeout=timeout_seconds,
        )
        messages = result["messages"]
        answer = get_final_answer(messages)
        rounds = count_tool_rounds(messages)
        matched = matches_expected_answer(answer, case["expected_any_of"])
        within_limit = is_within_round_limit(rounds, case["max_tool_rounds"])
        row.update(
            answer=answer,
            tool_rounds=rounds,
            answer_matched=matched,
            within_round_limit=within_limit,
            completed=matched and within_limit,
        )
    except Exception as error:
        row["error_type"] = type(error).__name__
    row["latency_seconds"] = round(perf_counter() - started, 2)
    return row
