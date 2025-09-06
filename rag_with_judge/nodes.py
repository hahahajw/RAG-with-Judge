"""
nodes - RAG with Judge 所需的所有节点

        检索、增强、生成 节点来自于 Naive RAG

Author - hahahajw
Date - 2025-06-27 
"""
from loguru import logger as log
from pydantic import BaseModel, Field
from typing import Literal
from uuid import uuid4

from langchain_core.runnables import RunnableConfig

from naive_rag.state import NaiveRagState
from rag_with_judge.state import JudgeState, CallNewRagState
from modules.prompt import get_judge_prompt


def judge_node(state: JudgeState,
               config: RunnableConfig):
    """
    利用 LLM 判断当前 similar_chunk 对回答 query 的帮助程度
    Args:
        state: state['old_query']，当前 Judge 收到的问题
               state['similar_chunk']，当前 Judge 要判断的 chunk
        config: config['configurable']['llm']，要扮演 Judge 的 LLM
                config['configurable']['recursion_depth']，用于更新当前 chunk 位于的递归深度

    Returns:
        LangChain 中的 Document 是可变对象，这里通过引用更改，不需显示返回
        state['similar_chunk'].metadata: 1. metadata['recursion_depth']，值来自 config，表示当前的递归深度
                                         2. metadata['next_rag_thread_id']，值为 ''，表示下次调用 RAG 的 thread_id
                                            如果之后路由到了 call_new_rag 节点，会更新这个键值对。其他情况下（score 为 0、2）时则不会改变
                                            这是为了之后能够将整个 RAG 流程以搜索树的形式复现出来
                                         3. 有关这次 Judge 的一些信息，包括
                                            metadata['score']，值来自 judgement.score
                                            metadata['reason']，值来自 judgement.reason
                                            metadata['next_query']，值来自 judgement.next_query
    """
    # log.info(f'Judge')

    # LLM 结构化输出
    class JudgeInformation(BaseModel):
        score: Literal[0, 1, 2] = Field(
            description="score: 0=Fully Useful, 1=Partially Useful, 2=Completely Useless"
        )
        reason: str = Field(description="a brief explanation of the judgment")
        supplementary_question: str = Field(
            description="provide the generated new question only if score is 1; otherwise, must be an empty string """,
            default=''
        )

    query = state['old_query']
    similar_chunk = state['similar_chunk']
    prompt = [get_judge_prompt(query=query, context=similar_chunk.page_content)]

    llm = config['configurable']['llm']
    llm = llm.with_structured_output(JudgeInformation)

    # 提交给 LLM Judge 进行判断
    judgement = llm.invoke(prompt)
    # log.info(f'将 {query} 和 {similar_chunk.page_content} 交给 judge 判断，结果为 {judgement}')

    # 更新当前 similar_chunk 的 metadata
    # 为当前的 similar_chunk 的 metadata 添加两个键值对「当前的递归深度」和「下次调用 RAG 的 thread_id」
    similar_chunk.metadata['recursion_depth'] = config['configurable']['recursion_depth']
    similar_chunk.metadata['next_rag_thread_id'] = ''

    # 向 metadata 中添加这次 Judge 的信息
    similar_chunk.metadata['score'] = judgement.score
    # if judgement.score == 0:
    #     similar_chunk.page_content += '123'
    #     # similar_chunk.page_content = ''
    similar_chunk.metadata['reason'] = judgement.reason
    similar_chunk.metadata['next_query'] = judgement.supplementary_question

    return


def tmp0_node(state: NaiveRagState):
    """过渡节点，处理 Judge 出来后不能直接 接上一个条件边"""
    # 很奇怪，在 Learn.ipynb 里面有一个复现的例子
    # 问题可能出在我对 Send API 的了解不够深
    return


def tmp_node(state: NaiveRagState):
    """过渡节点，缓和 Judge 节点引出的条件边 和 之后 call_new_rag 所需的 Send API 并行调用"""
    return


def call_new_rag_node(state: CallNewRagState,
                      config: RunnableConfig):
    """
    根据获得的问题递归调用 RAG
    Args:
        state: state['query_for_next_rag']，递归调用 RAG 所需的问题
        config: config 的全部，需要重新定义一个 config 才可以保证重建搜索树

    Returns:
        state['similar_chunk'].page_content: 利用本次 RAG 的答案更新，使其是知识完备的
        state['similar_chunk'].metadata['next_rag_thread_id']: 递归调用 RAG 所需的 thread_id
    """
    next_query = state['query_for_next_rag']

    # 如果上游节点传递了「空内容」则说明检索到的文档的得分全为 0（虽然不大可能）则不需要继续
    # 如果当前的递归深度达到最大值，也不需要继续
    if (
        next_query == '0'
        or config['configurable']['recursion_depth'] + 1 > config['configurable']['max_rec_depth']
    ):
        return

    # 递归调用 RAG
    rag = config['configurable']['graph']

    next_thread_id = str(uuid4())

    # 重新定义运行时配置
    new_config = {
        'configurable': {
            'thread_id': next_thread_id,
            'retriever': config['configurable']['retriever'],
            'llm': config['configurable']['llm'],
            'graph': rag,
            'recursion_depth': config['configurable']['recursion_depth'] + 1,
            'max_rec_depth': config['configurable']['max_rec_depth']
        }
    }
    print(f'使用 {next_thread_id} 调用 RAG，问题为 {next_query}，当前递归深度为 {new_config["configurable"]["recursion_depth"]}')

    response = rag.invoke(
        input={'query': next_query},
        config=new_config
    )

    # 更新 similar_chunk 的 page_content 和 metadata['next_rag_thread_id']
    similar_chunk = state['similar_chunk']
    similar_chunk.page_content += f"""\n[question]: {response['query']}\n[answer]: {response['answer']}\n"""
    similar_chunk.metadata['next_rag_thread_id'] = next_thread_id

    return
