"""
workflow - 构建 RAG with Judge 的流程

Author - hahahajw
Date - 2025-06-27 
"""
from typing import Literal

from langchain_core.documents import Document
from langgraph.graph import START, StateGraph, END
from langgraph.types import Send

from naive_rag.state import NaiveRagState
import naive_rag.nodes as naive_rag_nodes
import rag_with_judge.nodes as judge_nodes


def retrieve_to_judge(state: NaiveRagState):
    """
    使用 Send API 并行执行 Judge 过程
    Args:
        state: state['query']，这轮 RAG 的问题，用于初始化 JudgeState 中的 old_query 通道
               state['similar_chunks']，retrieve node 检索到的相似文本块，其中的每一个元素将被分配给一个 Judge

    Returns:
        Send API 的结果
    """
    query = state['query']
    similar_chunks = state['similar_chunks']

    # Sena API 文档: https://langchain-ai.github.io/langgraph/how-tos/graph-api/#map-reduce-and-the-send-api
    return [
        Send(
            'llm_as_a_judge',  # 终止（目标）节点的名称
            {
                'old_query': query,
                'similar_chunk': similar_chunk
            }  # 要向目标节点传递那些通道及对应的值是什么
        )
        for similar_chunk in similar_chunks
    ]


def is_2_in_score(state: NaiveRagState) -> Literal['2 in scores', '2 not in scores']:
    """
    根据 Judge 的判断结果路由到「生成」或「get_ready_for_next_rag」节点
    Args:
        state: state['similar_chunks'] 中元素的 metadata 属性中的 score 键值对

    Returns:
        '2 in scores': 如果检索到的文档中有得分为 2 的，当前 RAG 的流程就不应当继续下去了，可以直接生成答案了
        '2 not in scores': 如果文档中没有得分为 2 的，就继续下去，可能需要递归调用 RAG
    """
    similar_chunks = state['similar_chunks']
    for sc in similar_chunks:
        if sc.metadata['score'] == 2:
            return '2 in scores'

    return '2 not in scores'


def get_ready_for_next_rag(state: NaiveRagState):
    """
    整理出递归调用所需的 query 和对应的 similar_chunk，用于之后并行递归调用 RAG
    Args:
        state: state['similar_chunks'] 中元素的 metadata 属性中的
               1. score 键值对，用于判断当前 chunk 是否有 next_query
               2. next_query 键值对
               3. 当前的 chunk 本身

    Returns:
        Send API 的结果
    """
    next_queries, chunks = [], []
    similar_chunks = state['similar_chunks']

    for sc in similar_chunks:
        if sc.metadata['score'] == 1:
            next_queries.append(sc.metadata['next_query'])
            chunks.append(sc)

    # 当前检索到的文档得分全为 0
    # 为当前添加一个虚假的问题和对应的文档，避免 Send API 出错
    if not next_queries:
        next_queries.append('0')
        chunks.append(Document(page_content=''))

    # Send API: https://langchain-ai.github.io/langgraph/how-tos/graph-api/#map-reduce-and-the-send-api
    return [
        Send(
            'call_new_rag',  # 终止（目标）节点的名称
            {
                'query_for_next_rag': next_query,
                'similar_chunk': similar_chunk
            }  # 要向目标节点传递那些通道及对应的值是什么
        )
        for next_query, similar_chunk in zip(next_queries, chunks)
    ]


def get_rag_with_judge_workflow() -> StateGraph:
    """返回 RAG with Judge 的 workflow"""
    wf = StateGraph(NaiveRagState)

    # 添加节点
    nodes = {
        'retriever': naive_rag_nodes.retriever_node,
        'llm_as_a_judge': judge_nodes.judge_node,
        'tmp0': judge_nodes.tmp0_node,
        'tmp': judge_nodes.tmp_node,
        'call_new_rag': judge_nodes.call_new_rag_node,
        'augmented': naive_rag_nodes.augmented_node,
        'generate': naive_rag_nodes.generate_node
    }
    for name, node in nodes.items():
        wf.add_node(name, node)

    # 添加边
    wf.add_edge(START, 'retriever')
    wf.add_conditional_edges(
        'retriever',  # 起始节点
        retrieve_to_judge,  # 路由函数
        ['llm_as_a_judge']  # 终止节点s
    )
    wf.add_edge('llm_as_a_judge', 'tmp0')
    wf.add_conditional_edges(
        'tmp0',
        is_2_in_score,
        {
            '2 in scores': 'augmented',
            '2 not in scores': 'tmp'
        }
        # 终止节点s 可以是 list，其中的元素是「所有的终止节点」或 dict，其中的键是路由函数的返回值，值是此时要去的节点名
    )
    wf.add_conditional_edges(
        'tmp',
        get_ready_for_next_rag,
        ['call_new_rag']
    )
    wf.add_edge('call_new_rag', 'augmented')
    wf.add_edge('augmented', 'generate')
    wf.add_edge('generate', END)

    return wf
