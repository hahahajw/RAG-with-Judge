"""
rebuild_search_tree - 重建搜索树

Author - hahahajw
Date - 2025-08-12
"""
from typing import List, Optional
import os
import uuid
from datetime import datetime
import webbrowser

from langchain_core.documents import Document
from langgraph.graph.state import CompiledStateGraph


class QueryNode:
    def __init__(
            self,
            query: str,
            answer: str,
            thread_id: str
    ):
        self.query = query
        self.answer = answer
        self.thread_id = thread_id
        self.children: List[DocumentNode] = []


class DocumentNode:
    def __init__(
            self,
            similar_chunk: Document
    ):
        self.document = similar_chunk
        self.children: Optional[QueryNode] = None


def build_tree(
        graph: CompiledStateGraph,
        thread_id: str
):
    """
    递归重建搜索树
    Args:
        graph: 问答所使用的图
        thread_id: 问答使用的线程 ID

    Returns:
        搜索树的根节点
    """
    # 参考 API：
    # https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/time-travel/
    # https://langchain-ai.github.io/langgraph/reference/types/#langgraph.types.StateSnapshot

    # 获得图在当前 thread_id 下的最新状态
    state = list(graph.get_state_history(config={'configurable': {'thread_id': thread_id}}))[0]

    # 获得当前 thread_id 所处理的问题
    root = QueryNode(
        query=state.values['query'],
        answer=state.values['answer'],
        thread_id=thread_id
    )
    # 处理这个问题所检索到的文档
    similar_chunks = state.values['similar_chunks']
    for similar_chunk in similar_chunks:
        document_node = DocumentNode(
            similar_chunk=similar_chunk
        )
        # 递归处理由当前文档所产生的子问题
        next_thread_id = similar_chunk.metadata['next_rag_thread_id']
        document_node.children = build_tree(graph, next_thread_id) if next_thread_id != '' else None

        root.children.append(document_node)

    return root


def generate_unique_id() -> str:
    """生成唯一ID"""
    return str(uuid.uuid4())[:8]


def visualize_tree(root: QueryNode, output_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tmp")) -> str:
    """
    使用网页可视化搜索树（使用 Trae 开发）
    Args:
        root: 搜索树的根节点
        output_dir: 输出HTML文件的目录

    Returns:
        生成的HTML文件路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 生成唯一的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"tree_visualization_{timestamp}.html")

    # 生成HTML内容
    html_content = generate_tree_html(root)

    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"搜索树已可视化到: {output_file}")

    # 打开浏览器
    url = 'file://' + os.path.abspath(output_file).replace('\\', '/')
    print(f'正在打开浏览器: {url}')
    webbrowser.open(url)

    return output_file


def generate_tree_html(root: QueryNode) -> str:
    """生成搜索树的HTML表示"""
    # 递归生成树结构的HTML
    tree_html = generate_node_html(root)

    # 读取HTML模板文件
    template_path = os.path.join(os.path.dirname(__file__), 'templates', 'tree_template.html')
    with open(template_path, 'r', encoding='utf-8') as f:
        html_template = f.read()

    # 替换模板中的占位符
    html_template = html_template.replace('{{tree_html}}', tree_html)

    return html_template


def generate_node_html(node, level: int = 0) -> str:
    """递归生成节点的HTML表示"""
    if isinstance(node, QueryNode):
        return generate_query_node_html(node, level)
    elif isinstance(node, DocumentNode):
        return generate_document_node_html(node, level)
    return ""


def generate_query_node_html(node: QueryNode, level: int) -> str:
    """生成QueryNode的HTML表示"""
    node_id = generate_unique_id()

    # 生成子节点的HTML
    children_html = []
    for child in node.children:
        children_html.append(generate_node_html(child, level + 1))

    children_container = ""
    if children_html:
        children_container = f"""
        <div class="children-container level-{level + 1}">
            {''.join(children_html)}
        </div>
        """

    # 读取模板文件
    template_path = os.path.join(os.path.dirname(__file__), 'templates', 'query_node_template.html')
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()

    # 替换占位符
    template = template.replace('{{node_id}}', node_id)
    template = template.replace('{{thread_id}}', node.thread_id)
    template = template.replace('{{query}}', node.query)
    template = template.replace('{{answer}}', node.answer)
    template = template.replace('{{children_container}}', children_container)
    template = template.replace('{{level}}', str(level))

    return template


def generate_document_node_html(node: DocumentNode, level: int) -> str:
    """生成DocumentNode的HTML表示"""
    node_id = generate_unique_id()
    document = node.document

    # 获取文档元数据
    score = document.metadata.get('score', 'N/A')
    reason = document.metadata.get('reason', 'N/A')
    recursion_depth = document.metadata.get('recursion_depth', 'N/A')
    # 将metadata对象转换为格式化的字符串
    metadata = '\n'.join([f'{k}: {v}' for k, v in document.metadata.items()])

    # 生成子节点的HTML
    child_html = ""
    if node.children:
        child_html = f"""
        <div class="document-children level-{level + 1}">
            {generate_node_html(node.children, level + 1)}
        </div>
        """

    # 文档内容预览（前100个字符）
    content_preview = document.page_content[:100] + ('...' if len(document.page_content) > 100 else '')

    # 读取模板文件
    template_path = os.path.join(os.path.dirname(__file__), 'templates', 'document_node_template.html')
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()

    # 替换占位符
    template = template.replace('{{node_id}}', node_id)
    template = template.replace('{{content_preview}}', content_preview)
    template = template.replace('{{score}}', str(score))
    template = template.replace('{{recursion_depth}}', str(recursion_depth))
    template = template.replace('{{reason}}', reason)
    template = template.replace('{{document_content}}', document.page_content)
    template = template.replace('{{metadata}}', metadata)
    template = template.replace('{{child_html}}', child_html)
    template = template.replace('{{level}}', str(level))

    return template


if __name__ == '__main__':
    from dotenv import load_dotenv
    import os

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_milvus import Milvus, BM25BuiltInFunction
    from langgraph.checkpoint.memory import MemorySaver

    from rag_with_judge.workflow import get_rag_with_judge_workflow
    from modules.retriever import Retriever

    load_dotenv()

    llm = ChatOpenAI(
        api_key=os.getenv("BL_API_KEY"),
        base_url=os.getenv("BL_BASE_URL"),
        model='qwen-max-latest',
        # model='qwen-max-2025-01-25',
        # model='qwen2.5-7b-instruct',  # 小模型会有同样的效果吗  ->  不太好
        temperature=0.0
    )

    embed_model = OpenAIEmbeddings(
        api_key=os.getenv("BL_API_KEY"),
        base_url=os.getenv("BL_BASE_URL"),
        model="text-embedding-v3",
        dimensions=1024,
        check_embedding_ctx_length=False
    )

    vector_store = Milvus(
        collection_name='hybrid_hotpotqa500_hnsw',
        embedding_function=embed_model,
        builtin_function=BM25BuiltInFunction(output_field_names='sparse'),
        vector_field=['dense', 'sparse']
    )

    # 获得带有短期记忆的 RAG with Judge 图
    memory = MemorySaver()
    rag_with_judge_wf = get_rag_with_judge_workflow()
    rag_with_judge = rag_with_judge_wf.compile(checkpointer=memory)

    # 定义检索器
    rrf_retriever = Retriever(
        vector_store=vector_store,
        topk=3,
        search_fields=['dense', 'sparse'],
        search_type='hybrid',
        search_params=[
            {'metric_type': 'L2', 'params': {'ef': 21}},
            {'params': {'drop_ratio_search': 0.2}}
        ],
        ranker_type='rrf',
        ranker_params={'k': 60}
    )

    # 定义运行时配置
    config = {
        'configurable': {
            'retriever': rrf_retriever,
            'llm': llm,
            'thread_id': '1',
            'graph': rag_with_judge,
            'recursion_depth': 0,
            'max_rec_depth': 3
        }
    }

    query = 'Which suburb of Adelaide in the City of Norwood Payneham St Peters is included in the electoral district of Dunstan?'

    res = rag_with_judge.invoke(
        input={'query': query},
        config=config
    )

    r = build_tree(rag_with_judge, '1')
    visualize_tree(r)
