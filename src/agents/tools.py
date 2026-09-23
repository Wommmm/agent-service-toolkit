import math
import os
import re
import threading

import numexpr
from langchain_chroma import Chroma
from langchain_core.tools import BaseTool, tool



def calculator_func(expression: str) -> str:
    """使用numexpr计算数学表达式
    当你使用numexpr回答数学问题时，此函数非常有用。
    此工具仅用于数学问题，不适用其他任何情况，智能输入数学表达式
    参数expression(str)一个有效符合numexpr格式的表达式
    返回值也是str
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


calculator: BaseTool = tool(calculator_func)
calculator.name = "Calculator"


# Format retrieved documents
def format_contexts(docs):
    """
    拼检索结果给模型。带编号和来源，回答时能说清是哪条、出自哪个文件。
 source 用 .get 取：有的文档压根没这个字段，写 metadata["source"] 会 KeyError 直接挂。
   """
    parts = []
    for i,doc in enumerate(docs,1):
        source = doc.metadata.get("source","未知来源")
        parts.append(f"[{i}] 来源: {source}\n{doc.page_content}")
    return "\n\n".join(parts)


_lock = threading.Lock()
_retriever = None


def load_chroma_db():
    # 不传 embedding_function，Chroma 用自带的本地模型 all-MiniLM-L6-v2，不用 API Key
    # ⚠️ 建库脚本必须用同一个模型，否则向量空间对不上，检索结果全乱
    # 单例，只建一次。原来每次调用都 new 一个客户端，LangGraph 并行跑多个检索工具时
    # 几个线程会同时初始化 chromadb 的原生绑定，直接崩（报错还每次都不一样）。双检锁挡掉并发。
    global _retriever
    if _retriever is None:
        with _lock:
            if _retriever is None:
                chroma_db = Chroma(persist_directory="./chroma_db")
                top_k = int(os.getenv("RETRIEVAL_TOP_K", "5"))
                _retriever = chroma_db.as_retriever(search_kwargs={"k": top_k})
    return _retriever


def database_search_func(query: str) -> str:
    """Searches chroma_db for information in the company's handbook."""
    # Get the chroma retriever
    retriever = load_chroma_db()

    # Search the database for relevant documents
    documents = retriever.invoke(query)

    # Format the documents into a string
    context_str = format_contexts(documents)

    return context_str


database_search: BaseTool = tool(database_search_func)
database_search.name = "Database_Search"  # Update name with the purpose of your database
