import asyncio
import json
from time import perf_counter

import httpx
from langchain_core.tools import tool
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from pydantic import ValidationError

from agents.retrieval_contracts import model_payload, parse_response
from core import settings

class RetrievalFailure(RuntimeError):
   """Sanitized error code; remote content and authentication never escape."""
   def __init__(self, code: str):
      self.code = code
      super().__init__(code)


def transport_failure_code(error: Exception) -> str:
   pending = [error]
   while pending:
      item = pending.pop()
      if isinstance(item, BaseExceptionGroup):
         pending.extend(item.exceptions)
      elif isinstance(item, (TimeoutError, httpx.TimeoutException)):
         return "retrieval_timeout"
      elif isinstance(item, httpx.HTTPStatusError) and item.response.status_code in (401, 403):
         return "retrieval_auth_failed"
   return "retrieval_connection_failed"


async def fetch_retrieval(query: str) -> dict:
   query = query.strip()
   if not query or len(query) > 2000:
      raise RetrievalFailure("invalid_query")
   token = settings.RETRIEVAL_MCP_TOKEN
   headers = {"Authorization": "Bearer " + token.get_secret_value()} if token else None
   started = perf_counter()
   try:
      async with asyncio.timeout(settings.RETRIEVAL_MCP_TIMEOUT_SECONDS):
         async with httpx.AsyncClient(headers=headers, timeout=settings.RETRIEVAL_MCP_TIMEOUT_SECONDS,
                                     trust_env=False) as client:
            async with streamable_http_client(url=settings.RETRIEVAL_MCP_URL, http_client=client) as (read_stream, write_stream, _):
               async with ClientSession(read_stream, write_stream) as session:
                  await session.initialize()
                  arguments = {"query": query}
                  if settings.RETRIEVAL_MCP_CONTRACT == "v1":
                     arguments["top_k"] = settings.RETRIEVAL_TOP_K
                  result = await session.call_tool(settings.RETRIEVAL_MCP_TOOL, arguments=arguments)
                  if result.isError:
                     raise RetrievalFailure("retrieval_unavailable")
                  if result.structuredContent is None:
                     raise RetrievalFailure("missing_structured_content")
                  envelope = parse_response(result.structuredContent, settings.RETRIEVAL_MCP_CONTRACT)
                  payload = model_payload(envelope, settings.RETRIEVAL_CONTEXT_MAX_CHARS)
                  payload["client_duration_ms"] = round((perf_counter() - started) * 1000, 3)
                  return payload
   except RetrievalFailure:
      raise
   except TimeoutError:
      raise RetrievalFailure("retrieval_timeout") from None
   except (ValidationError, ValueError, TypeError):
      raise RetrievalFailure("invalid_retrieval_contract") from None
   except Exception as error:
      # Includes transport exception groups. Do not expose their nested headers/body.
      raise RetrievalFailure(transport_failure_code(error)) from None


@tool(response_format="content_and_artifact")
async def knowledge_search(query: str) -> tuple[str, dict]:
   """检索产品知识库，获取与用户问题相关的参考文档。"""
   payload = await fetch_retrieval(query)
   visible = {key: value for key, value in payload.items() if key != "traces"}
   return json.dumps(visible, ensure_ascii=False), payload
           
         
      
