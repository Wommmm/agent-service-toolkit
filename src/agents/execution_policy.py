"""Small deterministic policies for the existing RAG graph, not a new agent framework."""
import json
import re
from time import time
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


def current_turn(messages):
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return messages[index:]
    return messages


def start_turn(state, config, timeout):
    human = next((msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), None)
    turn_id = human.id if human else None
    if turn_id and state.get("turn_id") == turn_id:
        return {}
    return {
        "turn_id": turn_id, "execution_id": config.get("metadata", {}).get("engineering_run_id", str(uuid4())),
        "deadline": time() + timeout, "tool_calls_used": 0, "tool_events": [],
        "model_events": [], "outcome": "running", "execution_result": {},
    }


def remaining_seconds(state, limit):
    return max(0, min(limit, state.get("deadline", time() + limit) - time()))


def _message_size(message):
    return len(json.dumps({"content": message.content, "tool_calls": getattr(message, "tool_calls", [])}, ensure_ascii=False))


class ContextBudgetExceeded(RuntimeError):
    pass


def prepare_messages(messages, instructions, max_chars):
    """Drop only entire old human turns. Never cut a pending AI/tool sequence."""
    system = SystemMessage(content=instructions)
    blocks = []
    for message in messages:
        if isinstance(message, HumanMessage) or not blocks:
            blocks.append([])
        blocks[-1].append(message)
    used = _message_size(system)
    chosen = []
    for block in reversed(blocks):
        size = sum(_message_size(msg) for msg in block)
        if used + size > max_chars:
            if not chosen:
                raise ContextBudgetExceeded("current_turn_too_large")
            break
        chosen.insert(0, block)
        used += size
    return [system] + [message for block in chosen for message in block]


def error_tool_message(call, code):
    return ToolMessage(
        content=json.dumps({"status": "error", "code": code, "documents": []}, ensure_ascii=False),
        name=call["name"], tool_call_id=call["id"], status="error",
    )


def tool_event(message, duration_ms):
    artifact = message.artifact if isinstance(message.artifact, dict) else {}
    error_code = None
    if message.status == "error":
        try:
            candidate = json.loads(message.content).get("code")
            if candidate in {"budget_exceeded", "tool_timeout", "invalid_query", "retrieval_unavailable", "retrieval_timeout", "retrieval_auth_failed", "missing_structured_content", "invalid_retrieval_contract", "retrieval_connection_failed"}:
                error_code = candidate
        except (ValueError, TypeError):
            pass
    return {
        "tool": message.name, "call_id": message.tool_call_id,
        "status": "error" if message.status == "error" else artifact.get("status", "ok"),
        "duration_ms": artifact.get("client_duration_ms", duration_ms),
        "timing_scope": "client" if artifact.get("client_duration_ms") is not None else "tool_batch",
        "request_id": artifact.get("request_id"), "execution_id": artifact.get("execution_id"),
        "document_ids": [doc["id"] for doc in artifact.get("documents", [])],
        "error_code": error_code,
        "server_traces": [{key: trace.get(key) for key in ("stage", "status", "duration_ms", "documents")}
                          for trace in artifact.get("traces", []) if isinstance(trace, dict)],
    }


def finalize_result(state):
    """Validate reference membership, not factual entailment. No extra model call."""
    messages = current_turn(state["messages"])
    final = messages[-1]
    text = final.content if isinstance(final.content, str) else json.dumps(final.content, ensure_ascii=False)
    references = {}
    retrievals = []
    warnings = []
    errors = False
    had_tools = False
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        had_tools = True
        errors |= message.status == "error"
        if isinstance(message.artifact, dict) and "schema_version" in message.artifact:
            payload = message.artifact
            retrievals.append({key: payload.get(key) for key in ("request_id", "execution_id", "backend", "status")})
            warnings.extend(payload.get("warnings", []))
            for doc in payload.get("documents", []):
                references[doc["id"]] = {key: doc.get(key) for key in ("id", "source", "title", "chunk_id", "url")}

    requested = list(dict.fromkeys(re.findall(r"\[doc:([^\]\n]+)\]", text)))
    invalid = [identifier for identifier in requested if identifier not in references]
    allowed_urls = {doc["url"] for doc in references.values() if doc.get("url")}
    generated_urls = re.findall(r"https?://[^\s)\]>]+", text)
    invalid_urls = [url for url in generated_urls if url not in allowed_urls] if retrievals else []
    outcome = state.get("outcome", "running")
    if outcome == "running":
        outcome = "completed"
    if invalid or invalid_urls:
        text = "回答包含无法核对的引用，请重新提问或核对原始资料。"
        warnings.append("invalid_citation")
        answer_status = "unverified"
    elif references:
        answer_status = "answered" if requested else "unverified"
        if not requested:
            warnings.append("missing_citation")
    elif errors:
        answer_status = "error"
    elif retrievals:
        answer_status = "clarification" if any(item["status"] == "clarification" for item in retrievals) else "no_evidence"
    elif had_tools:
        answer_status = "unverified"
        warnings.append("unstructured_legacy_evidence")
    else:
        answer_status = "no_retrieval"
    if outcome != "completed":
        answer_status = outcome

    models = state.get("model_events", [])
    available_usage = [event.get("usage") for event in models]
    usage = None
    if models and all(isinstance(item, dict) and "input_tokens" in item and "output_tokens" in item for item in available_usage):
        usage = {key: sum(item.get(key, 0) for item in available_usage) for key in ("input_tokens", "output_tokens", "total_tokens")}
    result = {
        "schema_version": "1.0", "execution_id": state.get("execution_id"),
        "status": outcome, "answer_status": answer_status, "answer": text,
        "citations": [references[identifier] for identifier in requested if identifier in references],
        "warnings": list(dict.fromkeys(warnings)), "retrievals": retrievals,
        "tool_rounds": sum(isinstance(msg, AIMessage) and bool(msg.tool_calls) for msg in messages),
        "tool_calls": state.get("tool_calls_used", 0), "model_calls": len(models),
        "usage": usage, "model_events": models, "tool_events": state.get("tool_events", []),
        "citation_validation": "membership_only" if retrievals else "not_available",
    }
    response = final.model_copy(update={"content": text, "additional_kwargs": {**final.additional_kwargs, "agent_result": result}})
    return {"messages": [response], "execution_result": result}
