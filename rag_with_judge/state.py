"""
state - 定义「Judge 节点」和「call new rag」节点所需要的状态

        因为 Send API 在使用时强调「下游 node 的输入状态应当不同于上游 node 的状态」
        Send API：https://langchain-ai.github.io/langgraph/how-tos/graph-api/#map-reduce-and-the-send-api

        RAG with Judge 的主状态和 Naive RAG 一致

Author - hahahajw
Date - 2025-06-27 
"""
from typing import TypedDict
from langchain_core.documents import Document


class JudgeState(TypedDict):
    old_query: str  # 分配给 Judge 的问题
    similar_chunk: Document  # 分配给 Judge 的检索到的文档


class CallNewRagState(TypedDict):
    query_for_next_rag: str  # 下次调用 RAG 所需的问题
    similar_chunk: Document  # 要使用下轮 RAG 的回答更新提出新问题的 chunk
