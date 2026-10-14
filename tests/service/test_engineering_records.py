import asyncio
import json
from time import perf_counter
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from core import settings
from service.run_records import _read, _write, record_run, timed_stream
from service.service import app


def test_record_retention_and_sql_parameter_safety(tmp_path):
    path = tmp_path / "runs.sqlite3"
    for index in range(3):
        _write(path, {"run_id": str(index), "status": "completed"}, 2)
    assert _read(path, "0") is None
    assert _read(path, "2")["status"] == "completed"
    assert _read(path, "' OR 1=1 --") is None


@pytest.mark.asyncio
async def test_record_allowlist_excludes_raw_content(tmp_path, monkeypatch):
    path = tmp_path / "runs.sqlite3"
    monkeypatch.setattr(settings, "RUN_RECORDS_DB", str(path))
    await record_run("r1", "rag-assistant", "thread-1", perf_counter(), "completed", {"answer": "private", "query": "private", "usage": None, "tool_calls": 1})
    result = _read(path, "r1")
    assert "private" not in json.dumps(result)
    assert result["usage"] is None


@pytest.mark.asyncio
async def test_stream_timeout_closes_underlying_iterator():
    closed = []
    async def slow():
        try:
            await asyncio.sleep(.1)
            yield "x"
        finally:
            closed.append(True)
    with pytest.raises(TimeoutError):
        async for _ in timed_stream(slow(), .01):
            pass
    assert closed == [True]


def test_unknown_agent_returns_404():
    with patch("service.service.get_agent", side_effect=KeyError("unknown")):
        client = TestClient(app)
        assert client.post("/unknown/invoke", json={"message": "hi"}).status_code == 404
        assert client.post("/unknown/stream", json={"message": "hi"}).status_code == 404


def test_timeout_returns_504_and_records_failure(test_client, mock_agent, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "AGENT_REQUEST_TIMEOUT_SECONDS", .01)
    monkeypatch.setattr(settings, "RUN_RECORDS_DB", str(tmp_path / "runs.sqlite3"))
    async def slow(**kwargs):
        await asyncio.sleep(.1)
    mock_agent.ainvoke.side_effect = slow
    response = test_client.post("/rag-assistant/invoke", json={"message": "hi", "thread_id": "t1"})
    assert response.status_code == 504
    run_id = response.json()["detail"]["run_id"]
    assert test_client.get("/runs/" + run_id).json()["status"] == "timeout"


def test_final_result_survives_invoke(test_client, mock_agent, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "RUN_RECORDS_DB", str(tmp_path / "runs.sqlite3"))
    result = {"status": "completed", "answer_status": "answered", "tool_calls": 1, "usage": None}
    mock_agent.ainvoke.return_value = [("values", {"messages": [AIMessage(content="9V", additional_kwargs={"agent_result": result})]})]
    response = test_client.post("/rag-assistant/invoke", json={"message": "电池?", "thread_id": "t1"})
    assert response.status_code == 200
    assert response.json()["custom_data"]["agent_result"] == result
    assert test_client.get("/runs/" + response.json()["run_id"]).json()["tool_calls"] == 1


def test_stream_timeout_emits_error_and_done(test_client, mock_agent, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "AGENT_REQUEST_TIMEOUT_SECONDS", .01)
    monkeypatch.setattr(settings, "RUN_RECORDS_DB", str(tmp_path / "runs.sqlite3"))
    async def stream(**kwargs):
        yield ("updates", {"other": {"messages": [AIMessage(content="开始处理")]}})
        await asyncio.sleep(.1)
    mock_agent.astream = stream
    response = test_client.post("/rag-assistant/stream", json={"message": "hi"})
    assert response.status_code == 200
    assert '"code": "timeout"' in response.text
    assert response.text.count("[DONE]") == 1


def test_run_summary_inherits_bearer_auth(test_client, monkeypatch):
    from pydantic import SecretStr
    monkeypatch.setattr(settings, "AUTH_SECRET", SecretStr("test-secret"))
    assert test_client.get("/runs/missing").status_code == 401
    assert test_client.get("/runs/missing", headers={"Authorization": "Bearer test-secret"}).status_code == 404
