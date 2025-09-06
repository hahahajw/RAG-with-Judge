"""
prompt - 定义 RAG 所需的 prompt，包括 增强、生成 两部分

Author - hahahajw
Date - 2025-05-26 
"""
from langchain_core.messages import SystemMessage, HumanMessage


# TODO: 优化在生成答案时的引用部分。可以考虑优化一下 prompt，或使用 MCP，让 LLM 在每次调用外部知识时生成特定格式的引用
def get_rag_sys_prompt() -> SystemMessage:
    """获取 RAG 的系统提示"""

    # Qwen 帮忙写的：https://chat.qwen.ai/s/40b7ad62-33c4-44f4-bcdf-a47030c19ded?fev=0.0.169
    rag_sys_prompt = """You are a RAG Q&A assistant that works according to the following rules:
1. Answer questions using only the knowledge snippets provided, without using any external knowledge.
2. If the knowledge snippets contain no information that can answer the question, you must respond with "I cannot answer this question."
3. Answers must be concise and direct: output only the final answer, without explanation, references to the knowledge, or polite language.
4. If the knowledge snippets are irrelevant to the question, immediately trigger rule 2."""

    return SystemMessage(content=rag_sys_prompt)


def get_rag_with_judge_sys_prompt() -> SystemMessage:
    """定义 RAG with Judge 的 System Prompt"""
    # Qwen 帮忙写的：https://chat.qwen.ai/s/40b7ad62-33c4-44f4-bcdf-a47030c19ded?fev=0.0.169
#     rag_with_judge_sys_prompt = """You are a RAG Q&A assistant that strictly follows these rules:
# 1. Answer questions using only the knowledge provided by the user, including retrieved original knowledge snippets and added Q&A pairs (using only the answer part of each Q&A pair). Do not use any external knowledge.
# 2. If none of the knowledge (original snippets + Q&A answers) contains information that can answer the question, you must respond with "I cannot answer this question."
# 3. Responses must be concise and direct: output only the final answer, without explanations, references to the knowledge, or polite language.
# 4. If the knowledge is irrelevant to the question, immediately trigger rule 2."""
    rag_with_judge_sys_prompt = """You are a RAG Q&A assistant that operates strictly according to the following rules:
1. Use only the knowledge provided by the user to answer questions, including the retrieved raw knowledge snippets and the added Q&A pairs (the **answer part is considered supplementary knowledge**, and only the content of the answer should be used). Do not use any external knowledge.
2. If there is no information in all the knowledge (raw snippets + answers from Q&A pairs) that can answer the question, you must respond with "I cannot answer this question."
3. The response must be concise and direct: output only the final answer, without explanations, references to the knowledge, or polite language.
4. If the knowledge is irrelevant to the question, immediately trigger rule 2."""

    return SystemMessage(content=rag_with_judge_sys_prompt)


def get_input_prompt(query: str,
                     context: str) -> HumanMessage:
    """设定 RAG 在每一轮对话中输入的 prompt

    Args:
        query (str): 用户问题
        context (str): 整理好格式的知识

    Returns:
        HumanMessage: 包含用户提问和检索到的上下文的输入
    """
    # 在 Qwen 帮忙写系统提示词时，也建议了输入时的格式
    # https://chat.qwen.ai/s/40b7ad62-33c4-44f4-bcdf-a47030c19ded?fev=0.0.169
    rag_input_prompt = f"""[Knowledge]
{context}
[question]
{query}"""

    return HumanMessage(content=rag_input_prompt)


def get_judge_prompt(query: str,
                     context: str) -> SystemMessage:

    # 提示词部分是 DeepSeek 帮忙写的（虽然写到最后有点胡诌的意思了）
    # https://chat.deepseek.com/a/chat/s/3632669f-0426-41db-bb32-8045a172f115
    # https://chat.deepseek.com/a/chat/s/6789df81-2371-4c58-a43e-be7ecf370e39

    judge_prompt = f"""You are a professional Judger responsible for strictly evaluating the extent to which given knowledge helps answer a question. Please perform the task according to the following criteria:

### Evaluation Criteria (Must be strictly followed)
- **Fully Useful (2 points)**: The knowledge can independently and completely answer all parts of the question (no additional information needed).
- **Partially Useful (1 point)**: The knowledge can answer only some parts of the question, but not all (key information is missing).
- **Completely Useless (0 points)**: The knowledge is unrelated to the question, or cannot answer any part of it.

### Additional Rules
- If judged as **Partially Useful (1 point)**, you must generate a new question that satisfies the following:
  - Its answer can **directly supplement the missing part** in the current knowledge (e.g., if steps are missing, the new question should target the steps).
  - The answer provides the **maximum help** in answering the **original question** (prioritizing coverage of the missing core components, not peripheral details).
  - The new question must be concise, clear, and focused solely on the missing part (avoid being vague or repeating the original question).
- For other scores (0 or 2), do not generate a supplementary question.

### Output Requirements (Must be strictly followed)
- Output in **pure JSON format**, containing only the following three fields:
  - `"score"`: Integer (0, 1, or 2), indicating the helpfulness score.
  - `"reason"`: String, a brief explanation of the judgment (within 20 characters, directly referencing key points from the knowledge or question).
  - `"supplementary_question"`: String, provide the generated new question **only if `score` is 1**; otherwise, must be an empty string `""`.
- **Do not** include any extra fields, explanations, or text (e.g., "Okay, I will..."). The output must be directly parseable JSON.

Now, please make your judgment based on the input:
Knowledge: {context}
Question: {query}"""

    return SystemMessage(content=judge_prompt)
