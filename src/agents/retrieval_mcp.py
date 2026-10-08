import json
from langchain_core.tools import tool
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

@tool
async def knowledge_search(query: str) -> str:
   """检索产品知识库，获取与用户问题相关的参考文档。"""
   url = "http://127.0.0.1:8002/mcp"
   async with streamable_http_client(url=url) as (read_stream, write_stream,_):
      async with ClientSession(read_stream, write_stream) as session:
         await session.initialize()
         result = await session.call_tool(
            "search_knowledge",
            arguments = {"query": query},
         )
         if result.isError:
            raise RuntimeError(f"MCP检索失败:{result.content}")
         payload = result.structuredContent
         if payload is None:
            raise RuntimeError("MCP工具没有返回结构化数据")
         documents = payload["result"]
         return json.dumps(documents,ensure_ascii = False)
           
         
      
