"""Bounded local diagnostic summaries, stored separately from graph checkpoints."""
import asyncio
import json
import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from time import perf_counter, time

from core import settings

logger = logging.getLogger(__name__)


def _write(path, record, max_rows):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(destination, timeout=5)) as database, database:
        database.execute("CREATE TABLE IF NOT EXISTS agent_runs (run_id TEXT PRIMARY KEY, created REAL NOT NULL, summary TEXT NOT NULL)")
        database.execute("INSERT OR REPLACE INTO agent_runs VALUES (?, ?, ?)",
                         (record["run_id"], time(), json.dumps(record, ensure_ascii=False)))
        database.execute("DELETE FROM agent_runs WHERE run_id NOT IN (SELECT run_id FROM agent_runs ORDER BY created DESC, run_id DESC LIMIT ?)", (max_rows,))


def _read(path, run_id):
    destination = Path(path)
    if not destination.exists():
        return None
    with closing(sqlite3.connect(f"{destination.resolve().as_uri()}?mode=ro", uri=True, timeout=5)) as database:
        row = database.execute("SELECT summary FROM agent_runs WHERE run_id=?", (run_id,)).fetchone()
    return json.loads(row[0]) if row else None


async def record_run(run_id, agent_id, thread_id, started, status, result=None):
    if not settings.RUN_RECORDS_ENABLED or agent_id != "rag-assistant":
        return
    # Explicit allowlist: no answer, question, raw tool content, credential or config dump.
    summary = {key: result.get(key) for key in (
        "answer_status", "execution_id", "tool_rounds", "tool_calls", "model_calls",
        "usage", "model_events", "tool_events", "retrievals", "warnings",
    )} if isinstance(result, dict) else {}
    summary.update(run_id=str(run_id), agent_id=agent_id, thread_id=thread_id,
                   status=status, duration_ms=round((perf_counter() - started) * 1000, 3))
    try:
        await asyncio.to_thread(_write, settings.RUN_RECORDS_DB, summary, settings.RUN_RECORDS_MAX_ROWS)
    except Exception:
        # Diagnostics must not turn a successful answer into a failed request.
        logger.warning("Run summary unavailable run_id=%s", run_id)


async def get_run(run_id):
    return await asyncio.to_thread(_read, settings.RUN_RECORDS_DB, run_id)


async def timed_stream(stream, timeout):
    deadline = perf_counter() + timeout
    try:
        while True:
            remaining = deadline - perf_counter()
            if remaining <= 0:
                raise TimeoutError
            try:
                yield await asyncio.wait_for(anext(stream), remaining)
            except StopAsyncIteration:
                return
    finally:
        close = getattr(stream, "aclose", None)
        if close:
            await close()
