import asyncio
import json

import yaml

from agents import get_agent
from evals.case_runner import evaluate_case
from evals.metrics import calculate_completion_rate


async def main() -> None:
    with open("evals/cases.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    completed_results = []
    case_results = []

    for case in data["cases"]:
        row = await evaluate_case(case, get_agent)
        completed_results.append(row["completed"])
        case_results.append(row)
        print(f"Case: {row['id']} Completed: {row['completed']} Error: {row['error_type']}")

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
