"""Consumer-owned v1 contract; does not install the retrieval server or its models."""
import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RetrievedDocument(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    id: str = Field(min_length=1)
    content: str
    source: Literal["local", "web", "demo", "legacy"]
    title: str = ""
    chunk_id: str | None = None
    url: str = ""


class RetrievalEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    schema_version: Literal["1.0"]
    request_id: str = Field(min_length=1)
    execution_id: str = Field(min_length=1)
    backend: Literal["demo", "live", "legacy"]
    status: Literal["ok", "empty", "partial", "clarification"]
    documents: list[RetrievedDocument]
    message: str = ""
    warnings: list[str] = Field(default_factory=list)
    traces: list[dict] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent(self):
        if self.status in ("empty", "clarification") and self.documents:
            raise ValueError("Non-document statuses must not contain documents")
        if self.status == "ok" and not self.documents:
            raise ValueError("ok requires documents")
        if len({doc.id for doc in self.documents}) != len(self.documents):
            raise ValueError("Duplicate document IDs")
        return self


def parse_response(payload: object, contract: str) -> RetrievalEnvelope:
    if contract == "v1":
        return RetrievalEnvelope.model_validate(payload)
    if contract != "legacy" or not isinstance(payload, dict) or not isinstance(payload.get("result"), list):
        raise ValueError("Invalid explicitly selected legacy response")
    documents = []
    for doc in payload["result"]:
        if not isinstance(doc, dict) or not isinstance(doc.get("content"), str):
            raise ValueError("Invalid legacy document")
        content = doc["content"]
        identifier = doc.get("chunk_id") or hashlib.sha256(content.encode()).hexdigest()[:20]
        documents.append(RetrievedDocument(
            id=str(identifier), content=content, source="legacy",
            title=str(doc.get("title", "")), url=str(doc.get("url", "")),
        ))
    return RetrievalEnvelope(
        schema_version="1.0", request_id="legacy-unknown", execution_id="legacy-unknown",
        backend="legacy", status="ok" if documents else "empty", documents=documents,
        warnings=["legacy service has no execution identifiers"],
    )


def model_payload(envelope: RetrievalEnvelope, max_chars: int) -> dict:
    """Clip document text, retain identifiers, and explicitly report truncation."""
    payload = envelope.model_dump()
    remaining = max_chars
    clipped = False
    for document in payload["documents"]:
        text = document["content"]
        allowance = min(remaining, 4000)
        document["content"] = text[:allowance]
        document["truncated"] = len(text) > allowance
        clipped |= document["truncated"]
        remaining -= len(document["content"])
    if clipped:
        payload["warnings"].append("document_text_truncated")
    return payload
