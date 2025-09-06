"""
Index.py - 索引过程，将外部文件经过切片后存入向量数据库

Author - hahahajw
Date - 2025-05-23 
"""
from loguru import logger as log
from typing import (
    Optional,
    List,
    Set
)

from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus


class Index:
    """索引过程，将外部文本转换为嵌入向量并存储到向量数据库中"""

    def __init__(self,
                 embed_model: OpenAIEmbeddings,
                 data_path: str,
                 file_type_wanted: Set,
                 vector_store_name: str = 'IndexTest'):
        """

        Args:
            embed_model: 使用的嵌入模型
            data_path: 要加载文件所处文件夹的目录
            file_type_wanted: 现在可以处理的文件类型
            vector_store_name: 这次创建的向量数据库名称
        """
        self.embed_model: OpenAIEmbeddings = embed_model
        self.data_path: str = data_path
        self.file_type_wanted: Set = file_type_wanted
        self.vector_store_name: str = vector_store_name

        self.client: Optional[MilvusClient] = None
        self.vector_store: Optional[Milvus] = None

    @staticmethod
    def get_all_file_path(data_path, file_type_wanted: Set):
        """
        得到 data_path 下，文件后缀在 file_type_wanted 列表中的所有文件的路径，包括子目录下的文件

        Args:
            data_path: 文件所处文件夹的名称
            file_type_wanted: 想要提取文件的类型

        Returns:
            List[str]: 所有想要文件的路径，列表中的元素是某一个文件的路径
                       e.g ./data/a.pdf
        """
        if not os.path.exists(data_path):
            log.error(f'{data_path} 不存在！')
            raise ValueError(f'{data_path} 不存在！')

        interested_file_path = []
        not_interested_file_path = []
        # root 是当前文件夹、dir 是当前文件夹下的子文件夹、file 是当前文件夹下的单个文件
        for root, dirs, files in os.walk(data_path):
            for file_name in files:
                # 获取当前文件路径
                cur_file_path = os.path.join(root, file_name)
                # 判断当前文件是否是需要的
                _, file_extension = os.path.splitext(file_name)
                if file_extension in file_type_wanted:
                    interested_file_path.append(cur_file_path)
                else:
                    not_interested_file_path.append(cur_file_path)
                    log.info(f'现在暂不支持处理 {file_extension} 类型的文件 {cur_file_path}')

        log.info(f'已获取 {len(interested_file_path)} 份文件，忽略 {len(not_interested_file_path)} 份文件')

        return interested_file_path

    @staticmethod
    def file_to_documents(file_path) -> List[Document]:
        """
        将一个文件加载到 Document 列表中

        Args:
            file_path: 某一个文件的路径

        Returns:
            List[Document]: 被加载到 Document 对象中的 文件，PDF 的一页被加载到一个 Document 对象中
        """
        load = PyPDFLoader(file_path, extract_images=False)

        log.info(f'{file_path} 完成加载')

        return load.load()

    @staticmethod
    def documents_to_chunks(docs: List[Document],
                            chunk_size: int,
                            chunk_overlap: int) -> List[Document]:
        """
        将 context 较长的 Document 划分为较小的 chunk，可以直接存入向量数据库

        Args:
            docs: contex 较长的 Document
            chunk_size: chunk 中文本块的大致长度
            chunk_overlap: chunk 间重叠字符的数量

        Returns:
            List[Document]: 最终的 chunk 列表
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            length_function=len,  # 计算文本长度的方式，最后划分出的 chunk 不严谨满足这个长度。因为不是以长度为标准进行的划分
            separators=[
                "\n\n",
                "\n",
                "。",
                "，"
                ".",
                ",",
                " ",
                ""
            ]  # 划分文本所使用的分隔符。添加了中文的句号和逗号，以便更好的划分中文文本。此外，分隔符的顺序也是有意义的
        )

        return text_splitter.split_documents(docs)

    def get_index_done(self,
                       chunk_size: int = 700,
                       chunk_overlap: int = 50):
        """
        将 chunk 存入向量数据库中
        Args:
            chunk_size: chunk 中文本块的大致长度
            chunk_overlap: chunk 间重叠字符的数量

        Returns:
            None, 创建完成的 vector store 可以通过 Index 实例的 vector store 属性得到
        """
        from pymilvus.client.types import LoadState

        # 1. 看要创建的数据库是否已经存在，存在则直接返回向量数据库
        # 如果已经存在了一个重名的数据库但没被加载到内存中，则下面的代码会将其加载到内存中
        # 如果已经存在了一个重名的数据库并且已经被加载到内存中，则下面的代码只是为对应的向量数据库创建了一个新的引用
        vector_store = Milvus(
            embedding_function=self.embed_model,
            collection_name=self.vector_store_name
        )
        self.client = vector_store.client
        state = self.client.get_load_state(collection_name=self.vector_store_name)
        # 当前的向量数据库还没有被创建过
        if state['state'] == LoadState.NotExist:
            pass
        else:
            log.info(f'{self.vector_store_name} 向量数据库已存在，成功加载到内存中')
            self.vector_store = vector_store
            return

        # 2. 不存在则读取所有数据读、分块、嵌入，并返回向量数据库
        self.add_new_files(
            vector_store=vector_store,
            new_data_path=self.data_path,
            file_type_wanted=self.file_type_wanted,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        self.vector_store = vector_store

        return

    def add_new_files(self,
                      vector_store: Milvus,
                      new_data_path: str,
                      file_type_wanted: Set,
                      chunk_size: int = 700,
                      chunk_overlap: int = 50):
        """
        把 new_data_path 文件夹下所有类型在 file_type_wanted 中的文件加载到 vector_store 中

        Args:
            vector_store: 要存入数据的向量数据库
            new_data_path: 文件所处的文件夹
            file_type_wanted: 现在可以处理的文件类型
            chunk_size: chunk 中文本块的大致长度
            chunk_overlap: chunk 间重叠字符的数量

        Returns:
            None: 不返回任何东西
        """
        from uuid import uuid4
        from langchain_core.runnables import RunnableLambda

        # 获得文件夹下所有想要文件的路径（包括嵌套的子文件夹）
        all_files_path = self.get_all_file_path(
            data_path=new_data_path,
            file_type_wanted=file_type_wanted
        )

        # 批量处理文件
        # 返回值将是一个二维 list，列表中的元素是某个文件的所有 chunk
        all_file_chunks = (
            RunnableLambda(self.file_to_documents)
            | RunnableLambda(self.documents_to_chunks).bind(chunk_size=chunk_size, chunk_overlap=chunk_overlap)  # 使用 bind 方法为某个 runnable 绑定不在上个节点输出中的额外参数
        ).batch(all_files_path)

        # 将所有 chunk 依次加载到向量数据库中
        for file_chunks in all_file_chunks:
            # Qwen 的嵌入模型（text-embedding-v3）在接受字符数组作为输入时
            # 2025.03.10 更新：数组的最大长度为 10，每个元素最多 8192 token 长
            chunk_num = 0
            file_name = file_chunks[0].metadata['source']
            for i in range(0, len(file_chunks), 10):
                cur_file_chunks = file_chunks[i:i + 10]
                vector_store.add_documents(
                    documents=cur_file_chunks,
                    ids=[str(uuid4()) for _ in range(len(cur_file_chunks))]
                )
                chunk_num += len(cur_file_chunks)

            log.info(f'{file_name} 被划分为了 {chunk_num} 份 chunk，已存入 {self.vector_store_name}')

        return


if __name__ == '__main__':
    import os
    from dotenv import load_dotenv

    from langchain_milvus import Milvus
    from langchain_openai import OpenAIEmbeddings

    load_dotenv()

    e_m = OpenAIEmbeddings(
        api_key=os.getenv("BL_API_KEY"),
        base_url=os.getenv("BL_BASE_URL"),
        model="text-embedding-v3",
        dimensions=1024,
        check_embedding_ctx_length=False
    )

    # 创建了一个空 Index 实例
    vs_test1 = Index(
        embed_model=e_m,
        data_path='../data',
        file_type_wanted={'.pdf'},
        vector_store_name='vs_test1'
    )

    # 需要执行 get_index_done 函数算完成了嵌入
    vs_test1.get_index_done(chunk_size=700, chunk_overlap=50)
