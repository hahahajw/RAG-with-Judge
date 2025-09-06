"""
test - 

Author - hahahajw
Date - 2025-05-23 
"""
# import os
# import sys
# from loguru import logger as log
# from typing import List, Set
#
#
# def get_all_file_path(data_path: str,
#                       file_type_wanted: Set) -> List[str]:
#     """
#     得到 data_path 下，文件后缀在 file_type_wanted 列表中的所有文件的路径，包括子目录下的文件
#
#     Args:
#         data_path: 文件所处文件夹的名称
#         file_type_wanted: 想要提取文件的类型
#
#     Returns:
#         List[str]: 所有想要文件的路径
#     """
#     if not os.path.exists(data_path):
#         log.error(f'{data_path} 不存在！')
#         raise ValueError(f'{data_path} 不存在！')
#
#     interested_file_path = []
#     not_interested_file_path = []
#     # root 是当前文件夹、dir 是当前文件夹下的子文件夹、file 是当前文件夹下的单个文件
#     for root, dirs, files in os.walk(data_path):
#         for file_name in files:
#             # 获取当前文件路径
#             cur_file_path = os.path.join(root, file_name)
#             # 判断当前文件是否是需要的
#             _, file_extension = os.path.splitext(file_name)
#             if file_extension in file_type_wanted:
#                 interested_file_path.append(cur_file_path)
#             else:
#                 not_interested_file_path.append(cur_file_path)
#                 log.info(f'现在暂不支持处理 {file_extension} 类型的文件 {cur_file_path}')
#
#     log.info(f'已获取 {len(interested_file_path)} 份文件，忽略 {len(not_interested_file_path)} 份文件')
#
#     return interested_file_path
#
#
# if __name__ == '__main__':
#     # get_all_file_path(r'../RAG rebuild', file_type_wanted={'.py'})
#     # a = {'.pdf', 'a'}
#     # print(type(a))
#     # for key in a:
#     #     print(key)
#     from langchain_milvus import Milvus
#     from langchain_openai import ChatOpenAI
#     from langchain_openai import OpenAIEmbeddings
#     # client = Milvus()
#
#     run_config = {
#         'configurable': {
#             'vector_store': Milvus,
#             'top_k': 3,
#             'llm': ChatOpenAI,
#             'thread_id': '1'
#         }
#     }
#     print('-' * 10 + ' 进入检索模块 ' + '-' * 10)


# 调试 langchain_milvus 在使用混合搜索并设置了较大的 topk 时，返回的结果数总是不够
# 小 bug，当把 fetch_k 设置为同需要的 topk 一样大时就不存在这样的问题了，估计是参数传递时的错误
import os
import pprint

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from langchain_milvus import Milvus, BM25BuiltInFunction

load_dotenv()

# 定义 嵌入模型、LLM
embed_model = OpenAIEmbeddings(
    api_key=os.getenv("BL_API_KEY"),
    base_url=os.getenv("BL_BASE_URL"),
    model="text-embedding-v3",
    dimensions=1024,
    check_embedding_ctx_length=False
)
# vs_hy_500 = Milvus(
#     collection_name='hybrid_hotpotqa_500',
#     embedding_function=embed_model,
#     # 定义一个可以进行混合搜索的 Milvus 实例
#     builtin_function=BM25BuiltInFunction(output_field_names='sparse'),
#     vector_field=['dense', 'sparse']
# )
# 尝试只加载 dense 字段是否可行
# vs_hy_500 = Milvus(
#     collection_name='hybrid_hotpotqa_500',
#     embedding_function=embed_model,
#     # 定义一个可以进行混合搜索的 Milvus 实例
#     # builtin_function=BM25BuiltInFunction(output_field_names='sparse'),
#     # vector_field=['dense', 'sparse']
#     vector_field='dense'
# )

# 尝试只加载 sparse 字段
vs_hy_500 = Milvus(
    collection_name='hybrid_hotpotqa_500',
    embedding_function=None,
    builtin_function=BM25BuiltInFunction(output_field_names='sparse'),
    # vector_field=['dense', 'sparse']
    vector_field='sparse'
)

query = 'How long did the career span of the actor who starred with Mickey Rooney and Marilyn Maxwell in Off Limits?'
res = vs_hy_500.similarity_search_with_score(
    query=query,
    # query='hello',
    k=10,
    # ranker_type="weighted",
    # ranker_params={"weights": [1, 0]},
    fetch_k=10
    # ranker_type="rrf",
    # ranker_params={"k": 100},
)

pprint.pprint(res)
print(len(res))


# # 测试如何创建 索引类型为 HNSW 的密集嵌入 向量数据库
#
# # 先看看一个默认类型的向量数据库是怎么建立的
# # 定义嵌入模型
# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv
# import os
#
# load_dotenv()
#
# embed_model = OpenAIEmbeddings(
#     api_key=os.getenv("BL_API_KEY"),
#     base_url=os.getenv("BL_BASE_URL"),
#     model="text-embedding-v3",
#     dimensions=1024,
#     check_embedding_ctx_length=False
# )
#
# # 加载向量数据库
# from langchain_milvus import Milvus, BM25BuiltInFunction
#
# hybrid_vs = Milvus(
#     collection_name='dsa1',
#     embedding_function=embed_model,
#     builtin_function=BM25BuiltInFunction(output_field_names='sparse'),
#     vector_field=['dense', 'sparse'],
#     index_params=[
#         {
#             "metric_type": "L2",
#             "index_type": "HNSW",
#             "params": {
#                 "M": 64,
#                 "efConstruction": 100
#             }
#         },
#         {
#             "metric_type": "BM25",
#             "index_type": "AUTOINDEX",
#             "params": {}
#         }
#     ]
# )
#
# from uuid import uuid4
# from langchain_core.documents import Document
#
# doc1 = Document(
#     page_content='sdasd',
#     metadata={
#         'id': 1
#     }
# )
#
# hybrid_vs.add_documents(documents=[doc1], ids=[str(uuid4())])
