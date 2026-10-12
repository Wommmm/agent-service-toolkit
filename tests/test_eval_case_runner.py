import asyncio

from langchain_core.messages import AIMessage

from evals.case_runner import evaluate_case

CASE = {
    "id": "one",
    "agent_id": "rag",
    "question": "days?",
    "expected_any_of": ["15"],
    "max_tool_rounds": 2,
}


def test_failed_case_does_not_prevent_later_case():
    class Broken:
        async def ainvoke(self, *args, **kwargs):
            raise RuntimeError("private-provider-token")

    class Success:
        async def ainvoke(self, *args, **kwargs):
            return {"messages": [AIMessage(content="15 days")]}

    async def run():
        return [
            await evaluate_case(CASE, lambda name: Broken()),
            await evaluate_case(CASE, lambda name: Success()),
        ]

    failed, success = asyncio.run(run())
    assert failed["completed"] is False
    assert failed["error_type"] == "RuntimeError"
    assert "private-provider-token" not in str(failed)
    assert success["completed"] is True


def test_timed_out_case_is_recorded_and_cancelled():
    cancelled = []

    class Slow:
        async def ainvoke(self, *args, **kwargs):
            try:
                await asyncio.sleep(60)
            finally:
                cancelled.append(True)

    result = asyncio.run(evaluate_case(CASE, lambda name: Slow(), timeout_seconds=0.01))
    assert result["error_type"] == "TimeoutError"
    assert result["completed"] is False
    assert cancelled == [True]


def test_missing_agent_becomes_failed_case():
    def resolve(name):
        raise KeyError(name)

    result = asyncio.run(evaluate_case(CASE, resolve))
    assert result["error_type"] == "KeyError"
    assert not result["completed"]
