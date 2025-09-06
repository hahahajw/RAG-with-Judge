"""
run_qa - 调用不同的问答系统回答问题

Author - hahahajw
Date - 2025-07-31 
"""
from typing import (
    Union,
    Literal,
    Tuple,
    Dict,
    List,
    Optional
)
import json
from loguru import logger as log
from tqdm import tqdm
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph.state import CompiledStateGraph


def get_qa_pair(
        qa_system: Union[ChatOpenAI, CompiledStateGraph],
        config: dict,
        question_path: str,
        test_size: int,
        test_type: Literal['new', 'continue'],
        qa_res_path: str
):
    """
    调用问答系统回答问题
    问题的抽取是按照顺序来的，如果某个问题在回答时出错了，会将其答案设置为 ''
    Args:
        qa_system (Union[ChatOpenAI, CompiledStateGraph]): 问答系统的接口，可能是单个 LLM 或编译好的 LangGraph 图
        config: LangGraph 图的运行时配置。当问答系统为 LLM 时传递的是系统提示词
        question_path: 存放问题的路径
        test_size: 这次测试的问题数量
        test_type: new 表示这一轮的测试结果会覆盖之前的测试数据
                   continue 表示保留之前的测试结果，并将这轮 test_size 个问题的结果追加到原先结果的后面
        qa_res_path: 存放问答结果的路径

    Returns:
        None
    """
    # 检查 qa_system 是否合法
    if not (isinstance(qa_system, ChatOpenAI) or isinstance(qa_system, CompiledStateGraph)):
        raise TypeError(f'qa_system 必须是 ChatOpenAI 或 CompiledStateGraph 实例')

    # 提取这一轮要测试的问题
    all_questions, qa_pairs = extract_questions(
        question_path=question_path,
        test_size=test_size,
        test_type=test_type,
        qa_res_path=qa_res_path
    )
    # 在调用图进行问答时，需要为每个问题分配一个单独的 thread_id 方便后续根据 thread_id 重建搜索树
    # 这里分配的 thread_id 是问题在 all_question 中的索引
    thread_id = len(qa_pairs)

    # 调用 qa_system 进行测试（都是 LangChain 中的 Runnable）
    for question in tqdm(all_questions, desc="回答问题", total=len(all_questions)):
        cur_qa_pair = {
            'question': question['question'],
            'ground_truth': question['answer'],
            'prediction': ''
        }

        try:
            # 调用 LLM 回答
            if isinstance(qa_system, ChatOpenAI):
                response = qa_system.invoke([
                    config.get('SystemMessage', SystemMessage(
                        'You are a question-answering assistant. Respond directly and concisely based on your knowledge. Output only the final answer with no explanations, pleasantries, or extra information.')),
                    HumanMessage(question['question'])
                ])
                # 更新当前问答结果
                cur_qa_pair['prediction'] = response.content

            # 调用图回答
            else:
                # 设置调用当前问题所需的 thread_id
                config['configurable']['thread_id'] = cur_qa_pair['thread_id'] = str(thread_id)
                # 重新设置当前的递归深度（RAG with Judge 会用到）
                config['configurable']['recursion_depth'] = 0
                response = qa_system.invoke(input={'query': question['question']}, config=config)

                # 更新当前问答结果
                cur_qa_pair['prediction'] = response['answer']
                cur_qa_pair['rec_depth'] = config['configurable']['recursion_depth']

                # 提取黄金答案
                context = question['context']
                supporting_facts = []
                for title, i in question['supporting_facts']:
                    for j, c in enumerate(context):
                        if c[0] == title:
                            break
                    tmp = context[j][1][i] if context[j][0] == title else None
                    if tmp is None:
                        pass
                    else:
                        supporting_facts.append(tmp)
                # 提取检索到的内容
                similar_chunks = []
                for doc in response['similar_chunks']:
                    similar_chunks.append(doc.page_content)

                cur_qa_pair['similar_chunks'] = similar_chunks
                cur_qa_pair['supporting_facts'] = supporting_facts

        except Exception as e:
            log.error(f"{question['question']} 出现错误 {e}")

        thread_id += 1
        qa_pairs.append(cur_qa_pair)

    # 将问答结果写入文件
    os.makedirs(os.path.dirname(qa_res_path), exist_ok=True)
    with open(qa_res_path, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=4)

    return


def extract_questions(
        question_path: str,
        test_size: int,
        test_type: Literal['new', 'continue'],
        qa_res_path: str
) -> Tuple[List, List]:
    """
    按照 question_path 中的问题顺序，提取索引在 len(qa_res_path) : len(qa_res_path) + test_size 间的问题

    Returns:
        Tuple[List, List]: 提取的这次问答所需的问题,
                           先前的问答结果（首次问答则为 []）
    """
    # 打开问题文件，读取所有问题
    with open(question_path, 'r', encoding='utf-8') as f:
        all_questions = json.load(f)

    # 如果测试类型为 'new'，则从所有问题中取出前 test_size 个问题
    if test_type == 'new':
        all_questions = all_questions[:test_size]
        qa_pairs = []
        log.info(f'加载问题的索引为 [:{test_size}]，共 {len(all_questions)} 个')

    # 如果测试类型为 'continue'，则从 qa_res_path 文件中读取已经回答过的问题，并从所有问题中按照顺序取出 test_size 个问题
    elif test_type == 'continue':
        with open(qa_res_path, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)

        if len(qa_pairs) >= len(all_questions):
            raise ValueError(f'当前文件中的问题以及全部测试完成')

        all_questions = all_questions[len(qa_pairs):len(qa_pairs) + test_size]
        log.info(f'加载问题的索引为 [{len(qa_pairs)}:{len(qa_pairs) + test_size}]，共 {len(all_questions)} 个')

    # 如果测试类型不是 'new' 或 'continue'，则抛出异常
    else:
        raise ValueError(f'不支持的 test_type: {test_type}')

    # 返回提取的问题、先前已经回答过的问题
    return all_questions, qa_pairs


if __name__ == '__main__':
    import os
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv()

    llm = ChatOpenAI(
        api_key=os.getenv("BL_API_KEY"),
        base_url=os.getenv("BL_BASE_URL"),
        model='qwen-max-latest',
        temperature=0.0
    )

    get_qa_pair(
        qa_system=llm,
        config={},
        question_path='../data/hotpotqa_test_data_500.json',
        test_size=3,
        # test_type='continue',
        test_type='new',
        qa_res_path='./hotpotqa/llm_only/qa_result.json'
    )
