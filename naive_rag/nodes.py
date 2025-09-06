"""
nodes - 定义组成 NaiveRAG 所需的节点

Author - hahahajw
Date - 2025-05-26 
"""
from naive_rag.state import NaiveRagState
from modules.prompt import get_rag_sys_prompt, get_rag_with_judge_sys_prompt, get_input_prompt

from langchain_core.runnables import RunnableConfig


def retriever_node(state: NaiveRagState,
                   config: RunnableConfig) -> NaiveRagState:
    """
    根据用户问题检索到相应的文档

    Args:
        state: 用户的输入，state['query']
        config: 检索器，config['configurable']['retriever']

    Returns:
        NaiveRagState['similar_chunks']: 更新 NaiveRagState 下的 similar_chunks 通道
    """
    retriever = config['configurable']['retriever']
    similar_chunks_with_scores = retriever.get_similar_chunk_with_score(query=state['query'])

    return {'similar_chunks': [similar_chunk for similar_chunk, score in similar_chunks_with_scores]}  # type: ignore


def augmented_node(state: NaiveRagState) -> NaiveRagState:
    """
    将检索到的内容经过处理后填入到准备好的 prompt 中

    Args:
        state: 检索到的文档，state['similar_chunks']
               用户问题，state['query']

    Returns:
        NaiveRagState['messages']: 将处理好的信息添加到消息 list 中
    """
    docs = [f'knowledge {i}: ' + similar_chunk.page_content + '\n' for i, similar_chunk in enumerate(state['similar_chunks'])]
    input_meg = [get_input_prompt(query=state['query'], context=''.join(docs))]

    # 处理 system prompt
    if state.get('messages', None):
        # messages 中已有消息，证明已经有了 sys prompt 则不必重复更新
        pass
    else:
        # 对于不同的 RAG 系统使用正确的 sys prompt
        sys_prompt = get_rag_sys_prompt() if state['similar_chunks'][0].metadata.get('next_rag_thread_id') is None else get_rag_with_judge_sys_prompt()
        input_meg = [sys_prompt] + input_meg

    return {'messages': input_meg}  # type: ignore


def generate_node(state: NaiveRagState,
                  config: RunnableConfig) -> NaiveRagState:
    """
    回答最终的问题

    Args:
        state: 增强阶段更新的消息，state['messages']
        config: 要使用的 LLM，config['configurable']['llm']

    Returns:
        NaiveRagState['messages']: LLM 生成的完整消息
        NaiveRagState['answer']: LLM 生成的消息内容
    """
    llm = config['configurable']['llm']
    response = llm.invoke(state['messages'])

    return {'messages': [response], 'answer': response.content}  # type: ignore
