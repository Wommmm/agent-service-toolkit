import asyncio
import json
from time import perf_counter

import yaml
from langchain_core.messages import HumanMessage

from agents import get_agent
from evals.metrics import (
    calculate_completion_rate,
    count_tool_rounds,
    get_final_answer,
    is_within_round_limit,
    matches_expected_answer,
)


async def main() -> None:
    with open("evals/cases.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    completed_results = []
    case_results = []

    for case in data["cases"]:
        agent = get_agent(case["agent_id"])
        question = case["question"]

        start_time = perf_counter()
        result = await agent.ainvoke(
            {
                "messages": [
                    HumanMessage(content=question),
                ]
            },
            config={
                "configurable": {
                    "thread_id": f"eval-{case['id']}",
                    "user_id": "eval-user",
                }
            },
        )
        latency_seconds = perf_counter() - start_time
        messages = result["messages"]
        answer = get_final_answer(messages)
        tool_rounds = count_tool_rounds(messages)
        within_round_limit = is_within_round_limit(
            tool_rounds,
            case["max_tool_rounds"],
        )
        answer_matched = matches_expected_answer(
            answer,
            case["expected_any_of"],
        )

        completed = answer_matched and within_round_limit

        print(f"\nCase: {case['id']}")
        print(f"Answer: {answer}")
        print(f"Tool Rounds: {tool_rounds}")
        print(f"Answer Matched: {answer_matched}")
        print(f"Within Round Limit: {within_round_limit}")
        print(f"Completed: {completed}")
        print(f"Latency: {latency_seconds:.2f}s")

        completed_results.append(completed)
        case_results.append(
            {
                "id": case["id"],
                "question": question,
                "answer": answer,
                "tool_rounds": tool_rounds,
                "answer_matched": answer_matched,
                "within_round_limit": within_round_limit,
                "completed": completed,
                "latency_seconds": round(latency_seconds, 2),
            }
        )

    completion_rate = calculate_completion_rate(completed_results)
    report = {
        "completion_rate": completion_rate,
        "cases": case_results,
    }

    with open("evals/report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Report saved to evals/report.json")
    print(f"Completion Rate: {completion_rate:.2%}")


if __name__ == "__main__":
    asyncio.run(main())
