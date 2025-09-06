# 项目具体实现

*2025.05.20*

[LangChain 中文文档](https://python.langchain.ac.cn/docs/introduction/)

[LangGraph 中文文档](https://github.langchain.ac.cn/langgraph/)

## 工程结构设计

### modules 模块

modules 模块用于定义、实现各种 RAG 的基础模块，包括 索引、检索、增强、生成

#### 索引

索引模块的目的是将外部文本知识经过「分块、向量化」后存入向量数据库，用于之后的基于语义相似度的检索

对于复杂的现实场景，要得到「文本形式的知识」就很已经困难了。之后的分块操作也需要考虑特定场景下知识的形式

<font color='blue'>索引是一个很重要但现在的项目中考虑不多的优化方面</font>

##### Milvus 向量数据库扫盲

###### 什么是 collection

*参考文章：*

1. *[集合说明 | Milvus 文档](https://milvus.io/docs/zh/manage-collections.md)*
2. *[模式解释 | Milvus 文档](https://milvus.io/docs/zh/schema.md)*

~~向量数据库中的 [collection](https://milvus.io/docs/zh/manage-collections.md#Collection) 是指「一些嵌入向量的集合」~~

~~在 Milvus 中，每个被存储的嵌入向量被称为[实体](https://milvus.io/docs/zh/manage-collections.md#Entity)。在 Milvus 中，实体会有一些属性来描述自己（比如一个人的实体会有身高、体重等属性），这些属性被称为实体的 [字段](https://milvus.io/docs/zh/manage-collections.md#Schema-and-Fields)，将这些字段按照顺序组合在一起就形成了描述一个实体的 [Schema](https://milvus.io/docs/zh/manage-collections.md)。你可以通过 [自己定义实体的 schema](https://milvus.io/docs/zh/schema.md) 来明确实体应当有什么样的具体属性~~

在 Milvus 中，可以将 [collection](https://milvus.io/docs/zh/manage-collections.md#Collection) 视为一个二维表格，该表格具有 *固定的列* 和 *变化的行*

- 固定的列对应到 Milvus 中为：[字段](https://milvus.io/docs/zh/manage-collections.md#Schema-and-Fields)，所有字段组合在一起构成了描述一个 collection 的 [Schema](https://milvus.io/docs/zh/manage-collections.md#Schema-and-Fields)，当然你可以通过 [自己定义实体的 schema](https://milvus.io/docs/zh/schema.md) 来明确当前 collection 中的实体应当有什么样的属性

  一个 collection 最多可以有 [4 个向量字段](https://milvus.io/docs/zh/schema.md#Overview)，一个[向量字段（Vector Field）](https://milvus.io/docs/zh/v2.5.x/index-vector-fields.md)也可以被称为**向量场**（有关在多向量场上进行混合搜索可以参考[这篇](https://milvus.io/docs/zh/v2.5.x/multi-vector-search.md#Multi-Vector-Hybrid-Search)文档）

  此外，在定义 schema 后，你可以选择为某些字段[建立索引](https://milvus.io/docs/zh/create-collection.md#Optional-Set-Index-Parameters)。在 Milvus 等向量数据库中，[索引](https://milvus.io/docs/zh/v2.5.x/index-explained.md#Index-Explained) 是建立在数据之上的 **附加结构**

- 变化的列对应到 Milvus 中为：[实体](https://milvus.io/docs/zh/manage-collections.md#Entity)，一个 collection 可以有无数个实体，每个实体都有相同的、符合 Schema 的字段

Milvus 是[面向列](https://milvus.io/docs/zh/v2.5.x/overview.md#What-Makes-Milvus-so-Fast)的向量数据库，[一般情况下](https://milvus.io/docs/zh/create-collection.md)，需要自己设计，定义 schema 以创建一个 collection。但是，LangChain 已经和 Milvus 集成，[封装](https://milvus.io/docs/zh/basic_usage_langchain.md)好了一些函数和功能，就不需要重复造轮子了

###### LangChain 中的 VectorStore 和 Milvus 中的 collection 间的关系

前面已经提到：

1. 在 Milvus 中，collection 是「一群嵌入向量的集合」
2. LangChain 中的 VectorStore 是经过一定封装后的 collection

所以，VectorStore 的底层仍然是 collection，因此可以通过 client 来管理 VectorStore。对于一些 LangChian 没有封装进 langchain_milvus 库中的东西，你可以自己通过 client，将 VectorStore 视为 collection 来实现

##### 将 Milvus 部署到本地

*:knife: 感觉被摆了一刀！Milvus Lite [不支持 windows 环境](https://milvus.io/docs/zh/operational_faq.md#MilvusClientmilvusdemodb-gives-an-error-ModuleNotFoundError-No-module-named-milvuslite-What-causes-this-and-how-can-it-be-solved)*

*不过好像人家确实[说了](https://milvus.io/docs/zh/milvus_lite.md#Prerequisites)，我也看到了，不知道为什么没重视。估计睡少了 \^_^*

###### 利用 Docker 部署 Milvus Standalone

*参考文档：https://milvus.io/docs/zh/install_standalone-windows.md#Run-Milvus-in-Docker*

具体过程为：

1. 以管理员身份打开 Docker Desktop

2. 以管理员身份打卡 Powershell，输入 cmd 以进入 CMD 模式来运行脚本

   *要不是 Qwen [提醒](https://chat.qwen.ai/s/69d0a438-d9a6-420a-a4f6-737e49c13a84?fev=0.0.97)，我怎么也不会想到是因为没有进入 cmd 模式才导致脚本无法运行*

3. 输入以下命令下载脚本 

   ```shell
   Invoke-WebRequest https://raw.githubusercontent.com/milvus-io/milvus/refs/heads/master/scripts/standalone_embed.bat -OutFile standalone.bat
   ```

   这个命令将会把脚本下载到执行命令的文件夹下，并命名为 standalone.bat

4. 之后运行 `standalone.bat start`

   这个命令将把 Milvus Standalone 作为容器安装到 Docker 中

在完成安装后，在 Docker 启动的情况下 Python SDK 就可以和 Docker 中的 Milvus 沟通了

###### attu 可视化

*参考文章：[attu/README_CN.md at main · zilliztech/attu](https://github.com/zilliztech/attu/blob/main/README_CN.md) 和 https://chat.qwen.ai/s/69d0a438-d9a6-420a-a4f6-737e49c13a84?fev=0.0.97*

[attu](https://github.com/zilliztech/attu/blob/main/README_CN.md) 是 Milvus 开发的一个可视化应用，安装命令如下：

```shell
docker run -p 8000:3000 -e HOST_URL=http://localhost:8000 -e MILVUS_URL=host.docker.internal:19530 zilliz/attu:v2.5.6
```

和上面部署 Milvus Standalone 的步骤一样：用管理员身份打卡 Docker Desktop 和 Powershell（Powershell 要进入 CMD 模式）、运行上面的命令

这样 attu 就被部署到 Docker 中了，访问 http://localhost:8000/ 就可以看到自己的向量数据库了

###### 管理 Milvus 中的数据

*参考资料：*

1. *https://milvus.io/docs/zh/install_standalone-windows.md#Run-Milvus-in-Docker*

在完成前面两步的操作后，你就可以通过 Milvus 提供的 Python SDK 利用代码来操作数据了，并且操作结果可以在 attu 中看到。之后你就可以通过 attu 来管理你的数据

需要注意的是，Milvus 中存储的数据在你下载 standalone.bat 同一文件夹下的 **volumes/milvus** 文件夹中

##### 索引过程具体实现

之前用的是 [Qdrant](https://qdrant.tech/documentation/)，有一些缺点，比如：使用 client 创建多个 collection 时会有一些问题、文档的支持不太好

这次决定使用 [Milvus](https://milvus.io/docs/zh)，它的文档支持的比较好、也可以很方便的使用[混合搜索](https://milvus.io/docs/zh/milvus_hybrid_search_retriever.md)、也可以创建[多 collection](https://milvus.io/docs/zh/basic_usage_langchain.md)，这对之后知识库的创建比较有帮助

具体代码详见 RAG rebuild\modules\Index.py

需要稍微提一下的是其使用方法，读取一个文件夹下的所有数据并存入向量数据库的过程为：

1. 先创建一个 Index 实例
2. 调取 Index 实例下的方法 `get_index_done ` 完成索引过程

之后所有和向量数据库有关的内容就都在这个实例下的属性中，具体代码如下

```python
# 创建了一个空 Index 实例
    vs_test1 = Index(
        embed_model=embed_model,
        data_path='../data',
        file_type_wanted={'.pdf'},
        vector_store_name='vs_test1'
    )

    # 需要执行 get_index_done 函数算完成了嵌入
    vs_test1.get_index_done(chunk_size=700, chunk_overlap=50)
```

## Naive RAG 实现

![image-20250527171419201](https://cdn.jsdelivr.net/gh/hahahajw/MyBlogImg@main/img/202509060947718.png)

Naive RAG 是按照上面的流程实现的

### 在混合嵌入数据库中实现混合检索

*参考文档：*

1. *[Using Full-Text Search with LangChain and Milvus | Milvus Documentation](https://milvus.io/docs/full_text_search_with_langchain.md)*
2. *[Dense embedding + Sparse embedding](https://milvus.io/docs/milvus_hybrid_search_retriever.md#Dense-embedding-+-Sparse-embedding)*
3. *[如何使用 ReRanker](https://milvus.io/docs/zh/milvus_hybrid_search_retriever.md#Define-multiple-arbitrary-vector-fields)*
4. *「混合检索」所代表的意思就包括了 要在多个向量场上进行检索，然后对多个向量场上的检索结果进行重排，所以如果在加上什么「多路召回」就显得有些多余了*

稀疏检索 是通过将「问题」和「知识」表示为「几乎全是 0 的高维向量」，之后根据「打分函数（BM25、TF-IDF 等）」召回分数高的文档。有关稀疏检索的详细内容见 [什么是是稀疏检索](# 什么是稀疏检索)

密集检索 是通过将「问题」和「知识」通过预训练的嵌入模型转化为嵌入空间的向量（向量的每一维上的数很少是 0，因此被称为密集向量），之后根据两个嵌入向量间的距离来判断两者语义上的相似度，其在理解含义和上下文方面有优势

ReRanker 是...。有关 ReRanker 的详细内容见 [什么是 ReRanker](# 什么是 ReRanker)

#### 创建混合嵌入向量数据库

在 LangChain 与 Milvus 的集成中引入了稀疏检索和密集检索，具体来说，只需要在创建一个向量数据库时

1. 通过 `embedding_function` 定义将文本转化为密集嵌入的方法

2. 通过 `builtin_function` 定义将文本转化为稀疏嵌入的方法

   你可以像这个参数传递一个 langchain_milvue 封装好的 `BM25BuiltInFunction` 实例 或 [一个自定义的稀疏嵌入（链接所指向的 option 2）](https://milvus.io/docs/milvus_hybrid_search_retriever.md#Dense-embedding-+-Sparse-embedding)

   在使用封装好的 `BM25BuiltInFunction` 实例时，还需要注意其中的一些[参数](https://milvus.io/docs/full_text_search_with_langchain.md#Initialization-with-BM25-Function)，包括：

   - `input_field_names` (str): 输入字段的名称，默认为 `text` 。它表示此函数作为输入读取的字段（MIlvus 默认存储文本的字段就是 `text`）
   - `output_field_names` (str): 输出字段的名称，默认为 `sparse` 。它表示此函数将计算结果输出到哪个字段

   在实践中，如果存在多个嵌入函数组合时，对每个函数清晰定义其输入字段和输出字段可以避免歧义。在当前的实践例子中，上面参数的默认值就已经能够区分了

3. 通过 `vector_field` 来指定前面定义的不同嵌入函数所输出的字段，其类型为 `List[str]`

   当然，输出字段的名称是可以随意定义的。不过，当你更改名称后需要同时修改前面嵌入函数的输出字段

通过向上面的参数传递对应的值，我们就创建了一个 <font color='blue'>包含稀疏嵌入和密集嵌入的一个 Milvus 存储实例</font>，其可以进行<font color='blue'>密集 + 稀疏的混合搜索</font>

下面是一个具体的例子：

```python
# 定义密集向量的嵌入函数
import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings

load_dotenv()

embed_model = OpenAIEmbeddings(
    api_key=os.getenv("BL_API_KEY"),
    base_url=os.getenv("BL_BASE_URL"),
    model="text-embedding-v3",
    dimensions=1024,
    check_embedding_ctx_length=False
)

# 定义一个可以实现 密集 + 稀疏 混合检索的 Milvue 向量存储实例
from langchain_milvus import Milvus, BM25BuiltInFunction

vs = Milvus(
    collection_name='hybrid_retrieval_test',
    embedding_function=embed_model,
    builtin_function=BM25BuiltInFunction(output_field_names='sparse'),
    vector_field=['dense', 'sparse']
)

# 定义一些文档
from langchain_core.documents import Document

docs = [
    Document(
        page_content="在阿瓦·莫雷诺的小说《窃语之墙》中，年轻记者索菲亚在一座古老庄园的斑驳墙壁内揭开了一个数十年之久的阴谋，而过去的低语正危及她自身的理智。",
        metadata={"category": "Mystery"},
    ),
    Document(
        page_content="在伊桑·布莱克伍德的小说《最后的庇护所》中，一群幸存者必须携手合作，逃离末日后的荒芜之地，而人类最后的残余正挣扎着用绝望的求生努力维系生命。",
        metadata={"category": "Post-Apocalyptic"},
    ),
    Document(
        page_content="在莉拉·罗斯的小说《记忆窃贼》中，一位魅力非凡的盗贼受雇于一位神秘客户，此人拥有窃取和操控记忆的能力，任务是一场胆大包天的盗窃行动。然而，他很快发现自己陷入了一张充满欺骗与背叛的罗网之中。",
        metadata={"category": "Heist/Thriller"},
    )
]

# 将这些文档加载到刚刚创建的向量数据库中
from uuid import uuid4

vs.add_documents(
    documents=docs,
    ids=[str(uuid4()) for _ in range(len(docs))]
)

vs.similarity_search_with_score(
    query='窃贼',
    k=1, ranker_type="rrf", ranker_params={"k": 100}
)
# 结果为：
[(Document(metadata={'pk': '66920f59-4a3e-470f-a0c1-2304f084ff33', 'category': 'Heist/Thriller'}, page_content='在莉拉·罗斯的小说《记忆窃贼》中，一位魅力非凡的盗贼受雇于一位神秘客户，此人拥有窃取和操控记忆的能力，任务是一场胆大包天的盗窃行动。然而，他很快发现自己陷入了一张充满欺骗与背叛的罗网之中。'),
  0.009900989942252636)]
```

对应的 Milvus 实例中的字段如下：

![image-20250701172433982](https://cdn.jsdelivr.net/gh/hahahajw/MyBlogImg@main/img/202507011724863.png)

此外，还可以创建一个[包含多个密集嵌入的 Milvus 实例](https://milvus.io/docs/zh/milvus_hybrid_search_retriever.md#Define-multiple-arbitrary-vector-fields)，以实现更自由的混合多路检索（比如为假设性问题也分配一个向量场，这样就可以同时进行在这三个向量场上进行检索，并对搜索结果进行重排序。当然重排序时可以选择「调用 ReRanker 模型的 API」或「自己定义 ReRanker 的逻辑（一个比较好的博客在[这里](https://mp.weixin.qq.com/s/X5k5xVC65u0puIvfYOjdfA)）」）

#### 使用混合嵌入向量数据集

~~具体来说，我们希望能够在混合嵌入向量数据库上进行「单稀疏检索」、「单密集检索」和「混合检索」~~

- ~~对于「混合检索」来说，在混合嵌入向量数据库中，可以直接使用 langchain_milvus 封装的 `similarity_search_with_score`，并向其中传入 ReRanker 的类型来实现混合检索 + 多路召回~~

  ~~只不过这时获得的分数是经过归一化后的，你无法获得任何一路的原始分数~~

- ~~在混合嵌入向量数据库中，有 3 种方式（这在 `Learning.ipynb` 中有详细的记录）可以进行「单稀疏检索」、「单密集检索」~~

~~但是，由于 langchain_milvus 在搜索方面实现的不是很好，有很多的小问题，上面的方法都不是很好。最好的方法是：先通过 langchain_milvus 创建向量数据库，之后直接使用 [client.search](https://milvus.io/api-reference/pymilvus/v2.5.x/ORM/Collection/search.md) 和 [client.hybrid_search](https://milvus.io/api-reference/pymilvus/v2.5.x/ORM/Collection/hybrid_search.md) 来分别实现在单个向量场上的检索 和 混合检索~~

~~这样做可以带来下面两个优点：~~

1. ~~在混合嵌入中进行的「单稀疏或密集检索」可以获得和「单稀疏或密集嵌入实例」一致的搜索结果~~

   ~~前提是[密集嵌入要使用 HNSW 索引](https://github.com/milvus-io/milvus/discussions/43264)。如何在密集嵌入上使用 HNSW 索引见[这里](# 对密集嵌入使用 HNSW 索引)~~

2. ~~在召回多路检索的结果时，ReRanker 的选择更自由~~

*2025.07.29 更新：*

具体来说，使用「混合嵌入向量数据库」是指在其上进行 单稀疏检索、单密集检索、混合检索

- 对于「混合检索」来说，在混合嵌入向量数据库中，直接使用 langchain_milvus 封装好的 `similarity_search_with_score` 即可

  - 如果你希望「仅在一些向量场上进行混合搜索」并对结果使用 Weighted ReRanker 进行重排，你可以使用 `similarity_search_with_score` 并控制对应向量场的权重为你设计的权重，其它向量场的权重为 0 来实现
  
    需要注意的是，由于 `similarity_search_with_score` 在实现的过程中，是按照 `vector_store.vector_fields` 中向量场的**顺序**为每个向量场分配权重的，所以你在传入 `weights` 参数时，需要按照对应的顺序设计权重参数
  
    例如，你希望仅在字段为 `sparse1` 和 `dense1` 的向量场上进行搜索，`vector_store.vector_fields` 的结果为 `['dense1', 'dense2', 'sparse1', 'sparse2']`
  
    那么 Weighted ReRanker 的 `weights` 参数应为：`[0.8, 0.0, 0.5, 0.0]` （每一个权重控制在 [0, 1] 之间即可）
  
  - 如果你希望「仅在一些向量场上进行混合搜索」并对结果使用 rrf 或其他算法进行重排，那么你就需要自己通过调用 [client.hybrid_search](https://milvus.io/api-reference/pymilvus/v2.5.x/ORM/Collection/hybrid_search.md)，分别为需要的向量场创建 AnnSearchRequest 请求来实现了
  
    这是因为 `similarity_search_with_score` 在实现 rrf 时，默认是在所有向量场进行搜索后重排的，没有办法「仅在一些向量场上进行搜素」
  
    你可以参考自己在 `retriever.py` 和 `similarity_search_with_score` 的实现、[这篇](https://milvus.io/docs/zh/multi-vector-search.md#Perform-Hybrid-Search)、[这篇](https://milvus.io/docs/zh/hybrid_search_with_milvus.md#Hybrid-Search-with-Milvus) 文档
  
- 对于「单稀疏检索」、「单密集检索」来说，在混合嵌入向量数据库中有这样 2 种实现方法：

  1. [在 similarity_search_with_score 中使用 Weighted ReRanker](https://milvus.io/docs/zh/milvus_hybrid_search_retriever.md#Define-multiple-arbitrary-vector-fields)，通过控制不同向量场的权重参数来实现仅在某一个向量场上进行搜索

     同样，就像上面说的，你需要为 `vector_store.vector_fields` 中的所有向量场按顺序传入对应的权重参数。你需要的向量场为 1，其余向量场为 0

  2. 使用 [client.search](https://milvus.io/api-reference/pymilvus/v2.5.x/ORM/Collection/search.md) 

     或者在加载混合嵌入数据库实例时，仅加载你想要搜索的向量场的字段。这样可行的原因是：`similarity_search_with_score` 在实现时，如果当前 Milvus 实例中仅有一个嵌入函数（你选择加载的那个），会路由到 `_collection_search` 函数中调用 cleint.search 进行搜素

     但是这种方法不如直接调用 client.search 灵活，在传递一些参数（比如 output_fields）时会受到一些限制

###### 对密集嵌入使用 HNSW 索引

*参考资料：*

1. *[HNSW Milvus v2.5.x 文档](https://milvus.io/docs/zh/v2.5.x/hnsw.md#HNSW)*
2. *`Learn.ipynb` 中有关「在 langchain_milvus 中为向量指定索引类型」部分的代码*
3. *[索引向量字段 Milvus v2.5.x 文档  ->  如何为一个向量场或字段建立索引](https://milvus.io/docs/zh/v2.5.x/index-vector-fields.md?tab=floating#Index-a-Collection)*
4. *[索引解释 Milvus v2.5.x 文档  ->  索引只是建立在数据上的附加结构](https://milvus.io/docs/zh/v2.5.x/index-explained.md#Index-Explained)*
5. *[内存索引 Milvus v2.5.x 文档  ->  Milvus 支持的索引类型](https://milvus.io/docs/zh/v2.5.x/index.md)*

具体来说，`Milvus` 函数接受一个 `index_parmas` 的参数，其类型为 `Optional[Union[Dict, List[Dict]]]` 

如果你想为向量数据库中的一些向量场指定索引类型，在 langchain_milvus 中你需要按照 `vector_filed` 中向量场出现的顺序，为每个向量场指定索引类型。指定索引类型的方式和 Milvus 中 [为某个向量场或字段添加 index_parmas 的方法相同](https://milvus.io/docs/zh/v2.5.x/index-vector-fields.md?tab=floating#Index-a-Collection)

此外，在 Milvus 等向量数据库中，[索引](https://milvus.io/docs/zh/v2.5.x/index-explained.md#Index-Explained) 是建立在数据之上的 **附加结构**

因此，只要有数据，你就可以更改索引，甚至在[这篇文档](https://milvus.io/docs/zh/v2.5.x/index-vector-fields.md?tab=floating#Drop-an-Index)中，你还可以删除一个索引。当然，之后如果有需要，你还可以对相同的字段 [重新建立索引](https://milvus.io/docs/zh/v2.5.x/index-vector-fields.md#Index-a-Collection)，毕竟「索引只是建立在数据上的 **附加结构**」

##### 什么是稀疏检索

*参考资料：*

1. *[什么是稀疏检索-秘塔AI搜索](https://metaso.cn/search/8636172467221897216?q=什么是稀疏检索)*

   *这里面有个写的比较清楚的 RAG 综述，在[这里](https://metaso.cn/s/kqAkjh9)*

2. *[RGA检索大全入门-CSDN博客](https://blog.csdn.net/qq_52491868/article/details/149274607#t31)*

3. *[Kimi K2 对稀疏检索的解释](https://www.kimi.com/share/d208o6ro7orehb873pug)*

4. *[dorianbrown/rank_bm25: A Collection of BM25 Algorithms in Python](https://github.com/dorianbrown/rank_bm25)*

   *这个 Github 仓库里面实现的 BM25 算法来自[这篇论文](https://dl.acm.org/doi/10.1145/2682862.2682863)*

一个常见的稀疏检索流程是这样的（来自 Kimi K2 的解释）：

1. 离线建索引（Index）

   - 分词 → 去停用词 → 统计词频、文档频率 → 倒排表（倒排索引）

     有关「倒排索引」是什么可以参考 [Kimi 之后的解释](https://www.kimi.com/share/d209gh18bjvil5r2i0g0) 和 [这篇博客](https://dongyifeng.github.io/2024/10/inverted-index/)

   - 同时预计算 BM25/TF-IDF 需要的常量（如 avgdl、IDF）

2. 在线查询解析（Query Parsing）

   - 用户输入一句话 → 同样分词 → 得到查询词项集合 q={t₁,t₂,…,tₖ}

3. 候选召回（Candidate Retrieval）

   - 用倒排表把**至少包含一个查询词**的文档捞回来（这一步叫“posting list 合并”）
   - 这一步只保证召回，不求精准排序，所以很快

4. 精准打分 & 排序（Scoring & Ranking）

   - 对召回的小集合（通常几百到几千篇）逐一算 BM25 或 TF-ITF 得分
   - 按分数降序排，Top-K 返回给用户

###### TF-IDF



###### BM25



###### 可学习的稀疏嵌入

*参考资料：*

1. *[详解如何通过稀疏向量优化信息检索 - Zilliz 向量数据库](https://zilliz.com.cn/blog/Optimizing-Information-Retrieval-through-Sparse-Vectors)*

2. [现代稀疏神经检索：从理论到实践 - Qdrant 向量数据库](https://qdrant.org.cn/articles/modern-sparse-neural-retrieval/)

这么一看，「稀疏向量」和「稀疏嵌入」并不是一回事

这部分的内容稍微了解一下就 OK 了

##### 什么是 ReRanker

*参考资料：*

1. [RAG：稠密 或 稀疏？混合才是版本答案！ - 知乎](https://zhuanlan.zhihu.com/p/696910211)
2. 





#### 多路召回

##### weoghted

##### rrf

##### 子模优化

[用子模优化做文本选择、段落重排和上下文工程](https://mp.weixin.qq.com/s/X5k5xVC65u0puIvfYOjdfA)

### 检索模块的设计与实现

#### 初始化函数设计

初始化函数的签名如下：

```python
def __init__(
            self,
            vector_store: Milvus,
            topk: int = 3,
            search_fields: Optional[List[str]] = None,  # Digg 因为将默认参数设置为可变对象导致一次事故
            search_type: Literal['sparse', 'dense', 'hybrid'] = 'dense',
            search_params: Optional[List[Dict]] = None,
            ranker_type: Literal['weighted', 'rrf'] = 'rrf',
            ranker_params: Optional[Dict] = None
    ):
```

参数的解释如下：

- `vector_store`: 要查询的向量数据库。会通过它使用一些 langchain_milvus 封装好的函数

- `topk`: 返回的文档数

- `search_fields` (List[str]): 要在哪些向量场上进行检索，默认仅在密集嵌入上进行检索

- `search_type` (Literal['sparse', 'dense', 'hybrid']): 搜索类型，默认为 dense，即仅在一个密集嵌入向量场上进行搜索

  *当需要在超过 2 个向量场上进行搜索时，应指定为 hybrid*

- `search_params` (List[Dict]): 调用 [client.search](https://milvus.io/api-reference/pymilvus/v2.5.x/ORM/Collection/search.md) 或 [client.hybrid_search](https://milvus.io/api-reference/pymilvus/v2.5.x/ORM/Collection/search.md) 所需要的搜索参数

  *需要为 `search_fields` 中的每个向量场都设置对应的搜索参数*

- ranker_type: 重排序多路召回结果的方式，现在支持

- ranker_params: 定义 reranker 所需的参数

  *当 `ranker_type` 为 `weighted` 时，应为 `search_fields` 中的每个向量场按照顺序传入对应的权重。权重在 [0, 1] 之间，不要求权重和为 1*

  *Weighted ReRanker 的默认参数为 `{'weights': [1.0] * len(sear)}`*

**你可以更改 Retriver 实例的一些属性值以更改搜索类型**

####  主检索函数设计

主检索函数为 `get_similar_chunk_with_score`，该方法主要负责：

1. 检查当前的搜索参数是否合法，包括：

   - `search_fields` 在 向量数据库中是否存在（`vector_store.vector_fields`）
   - 当在超过 2 个向量场上进行搜索时，`search_type` 是否指定为了 hybrid
   - 是否为 `search_fields` 中的每个向量场都设置了对应的搜索参数
   - 当使用 [Weighted ReRanker](https://milvus.io/docs/zh/weighted-ranker.md) 时是否为 `search_fields` 中的每个向量场都设置了权重

   这些功能在 `check_args` 方法中实现

2. 根据当前的搜索参数，路由到 `sparse_search`（仅在某个稀疏向量场上搜索）、 `dense_search`（仅在某个密集向量场上搜索）、 `hybrid_search` （在多个向量场上进行搜索）

3. 调用 langchain_milvus 的 `_parse_documents_from_search_results` 解析搜索结果

##### sparse_search

仅在单个稀疏向量场上进行搜索，实现逻辑为：

1. 取出这次的搜索参数

2. 去除当前稀疏向量场所对应的字段

   这是因为，在 Milvus 中 [BM25 函数生成的稀疏向量不能在全文检索中直接访问或输出](https://milvus.io/docs/zh/full-text-search.md#FAQ)

   langchain_milvus 中也实现了 `_remove_forbidden_fields` 方法来移除所有索引类型为 `BM25` 的字段

3. 调用 [client.search](https://milvus.io/api-reference/pymilvus/v2.5.x/ORM/Collection/search.md)

##### dense_search

仅在单个密集嵌入向量场上搜索，实现逻辑为：

1. 取出这次的搜索参数

2. 找到当前向量场建立时所使用的 embedding_func

   这里面用到了，Milvus 实例中的两个属性：

   - `embedding_func`：在创建 Milvus 实例时向 `embedding_function` 传入的嵌入函数 或 嵌入函数列表

   - `_vector_fields_from_embedding`：使用嵌入模型所创建的向量场的字段集合。可以说 `_vector_fields_from_embedding[i]` 是由 `embedding_func[i]` 生成的

     这也是为什么 _collection_hybrid_search 中可以依靠字段找到对应的嵌入函数

     ```python
     for field, param_dict in zip(self._vector_field, param_list):
         search_data: List[float] | Dict[int, float] | str
         if field in self._vector_fields_from_embedding:
             embedding_func: EmbeddingType = self._as_list(self.embedding_func)[  # type: ignore
                 self._vector_fields_from_embedding.index(field)
             ]
             search_data = embedding_func.embed_query(query)
         else:
             search_data = query
             request = AnnSearchRequest(
                 data=[search_data],
                 anns_field=field,
                 param=param_dict,
                 limit=fetch_k,
                 expr=expr,
             )
             search_requests.append(request)
     ```

3. 调用 [client.search](https://milvus.io/api-reference/pymilvus/v2.5.x/ORM/Collection/search.md)

##### hybrid_search

在多个向量场上进行搜索，实现逻辑为：

0. 如果使用 rerank model 进行重排，路由到 `search_with_rerank_model` 

1. 从 `search_fields`、`search_params` 中解析出各自向量场的 AnnSearchRequest

   这段代码的实现逻辑参考的是 _collection_hybrid_search

2. 创建 ReRanker

3. 去掉所有索引类型为 `BM25` 的稀疏向量场

4. 调用 [client.hybrid_search](https://milvus.io/api-reference/pymilvus/v2.5.x/ORM/Collection/search.md) 

这段代码是在 `check_args` 后执行的，不用担心参数间的长度不匹配

##### search_with_rerank_model

使用 rerank model 进行重排，具体逻辑为：

1. 重新创建一个 Retriever 实例用于粗排

   这就要求使用 `model` 作为 `ranker_type` 时，在传递 `ranker_params` 时要指定

   - 粗排时返回的文档数(`pre_topk`)，是新 Retriever 实例的 `topk`，默认为 10
   - 粗排时使用的重排器(`pre_ranker_type`)，是新 Retriever 实例的 `ranker_type`，默认为 rrf
   - 粗排时使用的重排器参数(`pre_ranker_params`)，是新 Retriever 实例的 `ranker_params`，默认为 {'k': 60}

   创建新 Retriever 实例时所需的其他参数**继承于**当前的 Retriever 实例

2. 将粗排的结果传递给 rerank model

   调用参考和返回值说明参考[这里](https://bailian.console.aliyun.com/?tab=api#/api/?type=model&url=https%3A%2F%2Fhelp.aliyun.com%2Fdocument_detail%2F2780056.html)

3. 解析出最后的结果

   使用了返回响应中的 `output.results.index` 字段索引到粗排结果中对应的文档

   ![image-20250731205812095](https://cdn.jsdelivr.net/gh/hahahajw/MyBlogImg@main/img/202509060948511.png)

### 实现

见代码

### Index 流程的图解

## 引入 LLM as a judge 模块

![image-20250527171449427](https://cdn.jsdelivr.net/gh/hahahajw/MyBlogImg@main/img/202509060948241.png)

这是好理解的图解，真正实现是通过递归调用带有 LLM as a Judge 模块的 Naive RAG

### 节点间的状态如何传递

在 LangGraph 中，节点间通过图所定义的「[状态通道](https://langchain-ai.github.io/langgraph/concepts/low_level/#state)」通信

一般的情况是，图中所有节点的输入、输出类型都是同一个状态。当前节点经过一些操作后在返回值处选择对应的通道来更新，更新的逻辑是由对应通道 reducer 决定的。当然，你可以自己定义某个通道 reducer 的逻辑

此外，如果某个通道的类型是「可变对象」的话，你可以利用可变对象的特性，通过某个引用修改通道的内容而不用显示在节点结束时利用返回值更新对应的通道。这样就避免了需要自定义 reducer 的麻烦

由于 LangChain 中的 Document 对象就是可变对象，因此在代码实现中就利用了「可变对象」的特性，在 LLM as a Judge 节点中就直接更改了对应 Document 对象的内容，而不是通过定义一个 reducer 来根据 Document 对象的 id 来将更新应用到相应的文档中

此外，节点间还可以[利用不同的私有状态进行通信](https://langchain-ai.github.io/langgraph/concepts/low_level/#multiple-schemas)

> 1. 「图状态」是指在利用 `StateGraph` 定义是所传入的所有状态的集合
> 2. 节点本身可以声明额外的状态，前提是这个被声明的状态在之前已经被定义了

在代码实现中，一共有 3 个状态：

1. NaiveRagState，包括 query、similar_chunks、answer 三个通道
2. JudgeState，包括 old_query、similar_chunk 两个通道
3. CallNewRagState，包括 query_for_next_rag、similar_chunk 两个通道

这三个状态间通过「[Send API](https://langchain-ai.github.io/langgraph/how-tos/graph-api/#map-reduce-and-the-send-api)」和「可变对象的特性」完成通信，具体来说：

1. NaiveRagState 通过 Send API 将 query 和某个 similar_chunk 传递给 JudgeState，之后在 LLM as a Judge 节点中利用可变对象的特性更改分配给自己的 similar_chunk，这两者间的通信就结束了

2. 在进入 call_new_rag 节点前，会处理 NaiveRagState 中的 similar_chunks 通道，搜集由 LLM as a Judge 节点生成的新问题，并用这些新问题和对应的 similar_chunk 通过 Send API 调用 call_new_rag 节点

   在 call_new_rag 节点中，同样利用可变对象的特点，将这轮 RAG 生成的答案添加到对应的 similar_chunk 中，使其更接近知识完备。至此，CallNewRagState 和 NaiveRagState 间的通信就算完成了

### 递归过程中的状态检索

出发点是这样的：在 LangGraph 中如果要实现[图的持久化](https://langchain-ai.github.io/langgraph/concepts/persistence/#persistence)，需要在编译图时在 `complie` 函数的 `checkpointer` 参数传入一个[检查点库实例](https://langchain-ai.github.io/langgraph/concepts/persistence/#checkpointer-libraries)，并之后在运行时配置中定义 `thread_id` 键值对来指定存储图状态的[线程](https://langchain-ai.github.io/langgraph/concepts/persistence/#threads)

如果不改变线程的值，则图所有的[检查点](https://langchain-ai.github.io/langgraph/concepts/persistence/#checkpoints)都将存储在同一个线程中，之后可以利用编译图下的方法 `get_state_history({'configurable': {'thread_id': '你要查询的线程'}})`  来[获得](https://langchain-ai.github.io/langgraph/concepts/persistence/#get-state-history)这个图在这个线程下的所有的执行历史

如果你在运行时配置中更改了 `thread_id` 键值对的值，图将忘记之前所有的记忆重新开始。但只要你知道之前某个使用过的 `thread_id` 值，重新传入，那么你将重新获得之前对应的记忆

也就是说，只要我们可以保存「何时递归调用了 RAG with Judge」和「调用时所使用的 `thread_id`」并保持使用同一个编译后的图，我们将可以恢复整个递归调用过程。具体的实现逻辑如下：

1. 每次检索到的文档都会被交给 Judeg 进行判断，Judge 的判断都会以键值对的形式被记录到对应 Document 对象的 metadata 属性中
2. 当所有的文档被并行判断完成后
   1. 如果有得分为 2 的文档，将直接转到增强模块。因为，如果有一篇文档被判断为「可以回答问题」，当前层的问题就是可回答的，RAG with Judge 的流程就不应当继续递归下去
   2. 之后，如果检索到文档的得分为 1，将进入 call_new_rag 节点，产生递归调用行为，并将这次递归调用行为所使用的 `thread_id` 以键值对的形式记录到对应 Document 对象的 metadata 属性中

这样设计是比较合理的，毕竟递归调用行为就是为了「补充当前检索到文档所缺失的信息，使其接近知识完备」。通过这样的设计，我们就可以比较合理的重建出来搜索树

这里有一个比较令人困惑的点：在递归调用的过程中，[不同运行时配置的定义方式将会导致递归调用过程中图的状态是不可追踪的](https://langchaincommunity.slack.com/archives/C06PTQF6MCK/p1750763585087499)，即使两种定义运行时配置的方法的逻辑和行为都是一致的。这被视为 LangGraph 框架本身的一个问题

### 可视化搜索树

*参考 API:*

1. *https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/time-travel/*
2. *https://langchain-ai.github.io/langgraph/reference/types/#langgraph.types.StateSnapshot*

根据 LangGraph 提供的 Time Travel 功能，调用 `get_state_history()` 函数得到图在当前 `thread_id` 下的执行记录，返回值的类型为 ` Iterator[StateSnapshot]`

1. `StateSnapshot` 中的 `value` 属性记录了当前图所有通道的值
2. 在之前的设计中，只有当真正发生了递归调用后，Document 对象的 metadata 中的键值对才会更改，否则是默认值 `''`。且由于使用了 LangChain 的 `_parse_documents_from_search_results` 解析搜索结果，该函数会为搜索将建立一个全新的 Document 实例，所以每次的搜索结果都是不同的实例，即使搜索到同一个文档，前后也不会互相干扰

综上，可以递归的重建搜索树

具体实现见代码

## RAG with Judge 可能的改进方向

1. [OneTab - Shared tabs](https://www.one-tab.com/page/OM65Vi9FS_6uYqpb9G4Kog)  ->  蒙特卡洛树，树搜索也有了，不过人家都是用的蒙特卡洛树搜索

   上面文章搜索都是因为看了 [这篇](https://mp.weixin.qq.com/s/GHlwbjIHgfBD2ftnLXfvhg) 微信公众号的文章







## 引入知识图谱

## 评估模块

- [ ] RAG 有什么评估指标
  - [ ] 有正确答案下的评估指标
  - [ ] 没有正确答案下的评估指标
- [ ] 传统问答系统有什么评估指标

### 评估指标

EM

Token Level F1 score









### RAGAS









### 评估模块的流程设计

run_qa 根据问题获得

mertics/em_f1 计算



































## 小知识



### typing 模块

#### Optional 和 Union

*参考资料：https://chat.qwen.ai/s/23e4807d-b8b7-419d-9ca4-cbf3258531ca?fev=0.0.97*

`Optional[X]` 只能接受一个参数，表示一个变量是「类型 X」或「None」，其是本质是 `Union[X, None]` 的简洁写法（语法糖）

`Union[a, b. c, ...]` 可以接受多个参数，表示一个变量是类型 a、b、c 等中的一种

### 获得一个文件夹下所有文件的路径

最终的代码：

```python
def get_all_file_path(data_path: str,
                      file_type_wanted: Set) -> List[str]:
    """
    得到 data_path 下，文件后缀在 file_type_wanted 列表中的所有文件的路径，包括子目录下的文件

    Args:
        data_path: 文件所处文件夹的名称
        file_type_wanted: 想要提取文件的类型

    Returns:
        List[str]: 所有想要文件的路径
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
```

#### python 中的 set 和 dict

*参考资料：https://chat.qwen.ai/s/eb468f0b-eb46-40cd-ad70-c33cdc504b17?fev=0.0.97*

在 Python 中，字典和集合都是可以通过 `{}` 来创建的，区别在于：如果 `{}` 中的元素是键值对，那么其将被解释为字典；反正，如果其中的元素只是单个数据，没有用 `:` 这个语法符号连接构成键值对，那么其将被解释为集合

因此，`{'.pdf', 'a', 1}` 是一个集合。此外，空集合要用 `set()` 来创建，`{}` 是空字典 

### try except else finally

*参考资料：*

1. *[Python——第五章：处理异常try、except、else、finally - Magiclala - 博客园](https://www.cnblogs.com/Magiclala/p/17899259.html)*
2. *https://chat.qwen.ai/s/cc8678fd-c630-4615-a229-f0a8b896f4d3?fev=0.0.167*

```python
try:
    # 可能出现异常的代码
    pass
except ExceptionType:
    # 处理特定异常的代码
    pass
except Exception as e:
    # 处理除系统级外的任何异常
except:
    # 处理任何类型异常的代码
    pass
else:
    # 没有异常时执行的代码
    pass
finally:
    # 无论是否有异常都会执行的代码
    # 即使上面的某一块的代码中包含 return 
    # 或者在上面的某一个 except or else 代码中又出现了异常
    # finally 块中的语句总是会被执行
    pass

# 剩余的代码
print(f'代码继续')
```

### 有关 LangChain 框架的一些小知识

#### 当 chain 中的某个节点所需要的输入不在前面节点的输出中

如果是在 LangGraph 中，可以通过[添加运行时配置](https://github.langchain.ac.cn/langgraph/how-tos/configuration/)来实现

在 LangChain 中也提供了对应的解决办法，使用 [`.bind()` 方法](https://python.langchain.ac.cn/docs/how_to/binding/) 来为 Runnable 添加默认参数、或使用 [`.configurable_fields` 方法](https://python.langchain.ac.cn/docs/how_to/configure/#configuring-fields-on-arbitrary-runnables) 来为 Runnable 在运行时指定参数



### 有关 LangGraph 框架的一些小知识

#### MessageState

[MessageState](https://github.langchain.ac.cn/langgraph/concepts/low_level/#messagesstate) 是 LangGraph 出于方便的目的封装的一个类，具体如下：

```python
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

这个类只有一个「通道（键）」就是 `messages`，其对应的 reducer（更新方式） 是 `add_messages` 用于将对话添加到之前的信息中

需要注意的是，在某个节点返回对话消息时要返回的类型是 **消息列表**

#### 为图添加运行时配置

*参考资料：[如何为你的图添加运行时配置 - LangChain 框架](https://github.langchain.ac.cn/langgraph/how-tos/configuration/)*

为图添加运行时配置是比较重要的，这样可以避免在设计图的 state 时引入过多通道。对于那些仅在图的某些节点会被使用到的参数，通过运行时配置来传递是很合适的

使用方式也很简单，只需要 2 步：

1. 在定义图的节点时传入参数 `config: RunnableConfig` 

   `RunnableConfig` 下的 `configurable` 的类型是字典，也是存储运行时配置的位置。之后便可在节点内部，利用字典取值的方式就可以获得对应的参数值

2. 之后定义好一个 `RunnableConfig`（重要的是其中的 `configurable` 属性，在里面通过键值对的方式给出你希望在运行时传递的参数）

   调用图时，在参数 `config` 中传入你所定义的运行时配置即可

```python
from langgraph.graph import MessagesState
from langchain_core.runnables import RunnableConfig

# 这里不能写 run_config！只能写 config，不然 LangGraph 解析不了
# 因为你在调用图时是想 config 形参传递的运行时参数
# def node1(state: MessageState, run_config: RunnableConfig) -> MessageState:
def node1(state: MessageState, config: RunnableConfig) -> MessageState:
    # 在函数中直接从字典中取出运行时的参数即可
    llm = config['configurable'].get('llm_name', openai)
    response = llm.invoke(state['messages'][-1])
    
    return {'messages': [response]}

run_config = {
    'configurable': {
        'llm_name': Qwen
    }
}
graph.invoke({"messages": [HumanMessage(content="你好，请介绍一下你自己")]},
             config=run_config  # 需要在调用图时在这里传入你希望的运行时参数)
```

#### 为图添加记忆

*参考资料：*

1. *[如何为你的图添加线程级持久性 - LangChain 框架](https://github.langchain.ac.cn/langgraph/how-tos/persistence/)*
2. *[如何将线程级别持久化添加到子图中 - LangChain 框架](https://github.langchain.ac.cn/langgraph/how-tos/subgraph-persistence/)*

> :fire:
>
> 线程级持久化是指：记住这一轮对话的全部内容
>
> [跨线程持久化](https://github.langchain.ac.cn/langgraph/how-tos/cross-thread-persistence/)是指：在新的对话中可以记住之前对话的内容，或者其他的一些你希望图要记住的内容

要为图添加线程级的持久化，我们只需要：*在编译图时传入一个 [checkpointer](https://github.langchain.ac.cn/langgraph/reference/checkpoints/#langgraph.checkpoint.base.BaseCheckpointSaver)*，并在之后定义运行时配置时添加一个 `thread_id: str` 键值对来规定有关图的内容存储在哪里

之后若要获取这次对话的所有内容，只需要运行时配置中添加对应的 `thread_id: str` 键值对

如果你想开启一轮新的对话，只需要在运行时配置中添加一个新的  `thread_id: str` 键值对

需要注意的是，如果你希望子图也可以有线程级的持久化，你只需要在 *编译父图时传入一个 chenkpointer* 即可，LangGraph 会将父图的 chenkpointer 自动传入子图。这对于调用子图的 [两种情况](https://github.langchain.ac.cn/langgraph/how-tos/subgraph/) 都适用

#### 如何流式传输运行结果

*参考资料：*

1. *[Streaming-Overview](https://langchain-ai.github.io/langgraph/concepts/streaming/)*
1. *[Stream outputs](https://langchain-ai.github.io/langgraph/how-tos/streaming/)*
1. *[操作指南 - 流式传输 - LangChain 框架](https://github.langchain.ac.cn/langgraph/how-tos/#streaming)*

流式传输对于用户体验来说很重要。LangGraph 对编译好的图暴露了 [.stream()](https://langchain-ai.github.io/langgraph/reference/pregel/#langgraph.pregel.Pregel.stream) 方法，所有图都可以使用该方法进行流式传输

##### 流式传输的类型

在使用 [.stream()](https://langchain-ai.github.io/langgraph/reference/pregel/#langgraph.pregel.Pregel.stream) 方法进行流式传输时，有 [这样几种流式传输类型](https://langchain-ai.github.io/langgraph/how-tos/streaming/#supported-stream-modes) 可供选择，接下来将重点介绍其中几种

###### messages mode

[messages streaming mode](https://langchain-ai.github.io/langgraph/how-tos/streaming/#messages) 的目的是「从图中任何包含 LLM 的部分（节点、工具等）流式传输其输出」。在 LangGraph 中，使用 messages mode 进行流式传输将会获得一个元组 `(message_chunk, metadata)`，其中：

- `message_chunk` : 来自 LLM 的 token 或消息片段。
- `metadata` : 一个包含有关图节点和 LLM 调用的详细信息的字典。

之后，你可以简单的使用 `print(message.content, end='', flush=True)` 来向终端流式传输 LLM 生成的内容

当然，「从图中任何包含 LLM 的部分流式传输其输出」未免也太过疯狂。因此，LangGraph 也提供了 2 种在流式传输时控制 LLM 输出的方法，分别是 [只流式传输特定 tags 的 LLM 输出](https://langchain-ai.github.io/langgraph/how-tos/streaming/#messages)、[只流式传输特定节点中的 LLM 输出](https://langchain-ai.github.io/langgraph/how-tos/streaming/#filter-by-node)

1. 方式一是通过在初始化 LLM 时为其打上不同的 tags，这些 tags 会再返回元组中的 `meatdata` 中体现，可以在输出时通过 if 条件过滤
2. 方式二则是利用了返回的 `meatdata` 中有 `langgraph_node` 这一属性（字段），通过在 if 条件中限定指定的节点才能输出来完成控制

一个简单的例子如下：

```python
query = input('请输入你的问题：')
# 指定 stream_mode="messages" 将会返回一个元组
for message, metadata in agentic_rag_graph.stream(
    input={'query': query, 'thread_ids': ['1']},  # 初始时传递给图状态中某些通道的值
    stream_mode="messages",
    config=config  # 运行时配置
):
    # 通过限定节点名字来控制 llm 输出
    if metadata["langgraph_node"] == "generate":
        print(message.content, end='', flush=True)
```

###### update mode 和 value mode

> :bulb:
>
> 需要注意的是，这里的「流式传输」和 LLM 以 token 形式流式传输其输出是不同的，返回的结果不是一个个蹦出来的
>
> 这里的「流式传输」侧重于强调：在图还没有执行完成前就可以获得图在执行过程中的状态，不必必须要等到图最终执行完成后才能获得图的状态
>
> 这么一看，其实 LLM 的流式输出也是一样的，也是「在 LLM 生成答案的过程中就可以实时获得其输出，而不是必须等到 LLM 生成所有回复后才能获得其输出」
>
> 因此，可以这样说：流式传输的本质是 *在程序完成前就可以实时获得其输出，不用必须等到程序完成后才能获得其输出*

[update mode 和 value mode](https://langchain-ai.github.io/langgraph/how-tos/streaming/#stream-graph-state) 都是用来流式传输图在运行时的状态（通道及其值），两者间的不同在于：

- `update` mode 会在图的每一步之后流式传输 刚刚被更新的状态（通道），即增量更新
- `value` mode 会在图的每一步之后流式传输 所有状态（通道）的值，即完整（有值的）的状态

由于存在这样的不同，所以两者的返回值也不相同，具体来说：

- 由于 `update` mode 是增量更新，所以其返回值是一个只有一个键值对的字典，其中的键是「发起更新的节点名称」，值是「该节点对图状态的增量更新」
- 由于 `value` mode 是体现完整（有值的）的状态，所以其返回值就是在执行完成当前节点后当前的状态，类型是一个字典

这可以从下面这个具体的例子中看到：

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
  topic: str
  joke: str
  a: str


def refine_topic(state: State):
    return {"topic": state["topic"] + " and cats"}


def generate_joke(state: State):
    return {"joke": f"This is a joke about {state['topic']}", "a": "a -> b"}

graph = (
  StateGraph(State)
  .add_node(refine_topic)
  .add_node(generate_joke)
  .add_edge(START, "refine_topic")
  .add_edge("refine_topic", "generate_joke")
  .add_edge("generate_joke", END)
  .compile()
)

for chunk in graph.stream(
    {"topic": "ice cream"},  # 这时对 topic 通道的更新并没有被算进去
    stream_mode="updates",
):
    print(chunk)
# 输出为：
# {'refine_topic': {'topic': 'ice cream and cats'}}
# {'generate_joke': {'joke': 'This is a joke about ice cream and cats', 'a': 'a -> b'}}
# 可以看到，updates mode 仅返回了被更新的状态（通道）及其值
# 并且，START 节点（初始化时）对 topic 通道的更新并没有算到其中

for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="values",
):
    print(chunk)
# 其输出为：
# {'topic': 'ice cream'}
# {'topic': 'ice cream and cats'}
# {'topic': 'ice cream and cats', 'joke': 'This is a joke about ice cream and cats', 'a': 'a -> b'}
# 可以看到，values mode 每次返回的都是当前图的所有状态（通道）的值，如果这个通道的值不是空的话
# 在这里，START 节点（初始化时）对 topic 通道的更新也被计算在内
```

###### 流式传输子图的输出

> :cherry_blossom:
>
> 流式传输子图的输出是指：在使用 values mode 或 updates mode 流式传输图的状态时，传输结果中也包含子图的运行时状态（有值的通道）

要在[流式输出中包含子图的输出](https://langchain-ai.github.io/langgraph/how-tos/streaming/#subgraphs)，只需在父图调用 `.stream()` 方法时，将参数 `subgraphs` 设置为 True，即 `subgraphs=True` ，这将同时流式传输父图和任何子图的输出

需要注意的是，流式传输的输出将是一个元组 `(namespace, data)`，其中的 `namespace` 也是一个元组，用以说明是由那个节点调用的当前子图，可以理解成「要到达当前节点所经过的所有节点（当前节点的调用路径）」

>:star2:
>
>之所以 `namespace` 是一个元组，是因为可能有嵌套调用子图的情况，这时 `namespace` 中的值将会是「从最上层的图开始，到当前节点的调用路径」
>
>- 对于不包含子图的图，其调用路径是一个空元组 `()`
>- 对于包含一个子图的图，子图的某个节点的调用路径将是 (父图的某个节点,)，例如 `('node_2:2908a094-f53a-da90-12d9-85dd863cc177',)`。node_2 是父图中调用当前子图的节点，后面 str 是节点的 task_id
>- 对于包含一个嵌套子图（即子图中仍包含一个子图）的图，最底层图中某个节点的调用路径将是 (祖父图中调用父图的节点, 父图中调用当前子图的节点)，例如：`('node_2:2908a094-f53a-da90-12d9-85dd863cc177', 'subgraph_node_3:6ee62665-bb6d-078d-e67b-1631d00f1758')`。同样，祖父图的 node_2 调用父图，父图中的 subgraph_node_3 调用当前的子图

下面是一个包含嵌套子图流式输出的例子：

```python
from langgraph.graph import START, StateGraph
from typing import TypedDict, List, Annotated
import operator


# Deffine sub sub graph
class SubSubgraphState(TypedDict):
    foo: str
    a: Annotated[List[int], operator.add]

def sub_subgraph_node_1(state: SubSubgraphState):
    return {'foo': state['foo'] + ' a', 'a': [1]}

def sub_subgraph_node_2(state: SubSubgraphState):
    return {'a': [2]}

sub_subgraph_builder = StateGraph(SubSubgraphState)
sub_subgraph_builder.add_node(sub_subgraph_node_1)
sub_subgraph_builder.add_node(sub_subgraph_node_2)
sub_subgraph_builder.add_edge(START, "sub_subgraph_node_1")
sub_subgraph_builder.add_edge("sub_subgraph_node_1", "sub_subgraph_node_2")
sub_subgraph = sub_subgraph_builder.compile()


# Define subgraph
class SubgraphState(TypedDict):
    foo: str  # note that this key is shared with the parent graph state
    bar: str

def subgraph_node_1(state: SubgraphState):
    return {"bar": "bar"}

def subgraph_node_2(state: SubgraphState):
    return {"foo": state["foo"] + state["bar"]}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_node(subgraph_node_2)
subgraph_builder.add_node('subgraph_node_3', sub_subgraph)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph_builder.add_edge("subgraph_node_2", "subgraph_node_3")
subgraph = subgraph_builder.compile()

# Define parent graph
class ParentState(TypedDict):
    foo: str

def node_1(state: ParentState):
    return {"foo": "hi! " + state["foo"]}

builder = StateGraph(ParentState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", subgraph)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
graph = builder.compile()

for chunk in graph.stream(
    {"foo": "foo"},
    # stream_mode="values",
    # stream_mode='updates',
    stream_mode=['values', 'updates'],
    subgraphs=True, 
):
    print(chunk)

# 输出如下：
((), 'values', {'foo': 'foo'})
((), 'updates', {'node_1': {'foo': 'hi! foo'}})
((), 'values', {'foo': 'hi! foo'})
(('node_2:c6ccc3f6-7f13-bde7-859d-9d8401fc88f0',), 'values', {'foo': 'hi! foo'})
(('node_2:c6ccc3f6-7f13-bde7-859d-9d8401fc88f0',), 'updates', {'subgraph_node_1': {'bar': 'bar'}})
(('node_2:c6ccc3f6-7f13-bde7-859d-9d8401fc88f0',), 'values', {'foo': 'hi! foo', 'bar': 'bar'})
(('node_2:c6ccc3f6-7f13-bde7-859d-9d8401fc88f0',), 'updates', {'subgraph_node_2': {'foo': 'hi! foobar'}})
(('node_2:c6ccc3f6-7f13-bde7-859d-9d8401fc88f0',), 'values', {'foo': 'hi! foobar', 'bar': 'bar'})
(('node_2:c6ccc3f6-7f13-bde7-859d-9d8401fc88f0', 'subgraph_node_3:67322fcd-7f9d-44aa-a1b9-d0260a6b9f7e'), 'values', {'foo': 'hi! foobar', 'a': []})
(('node_2:c6ccc3f6-7f13-bde7-859d-9d8401fc88f0', 'subgraph_node_3:67322fcd-7f9d-44aa-a1b9-d0260a6b9f7e'), 'updates', {'sub_subgraph_node_1': {'foo': 'hi! foobar a', 'a': [1]}})
(('node_2:c6ccc3f6-7f13-bde7-859d-9d8401fc88f0', 'subgraph_node_3:67322fcd-7f9d-44aa-a1b9-d0260a6b9f7e'), 'values', {'foo': 'hi! foobar a', 'a': [1]})
(('node_2:c6ccc3f6-7f13-bde7-859d-9d8401fc88f0', 'subgraph_node_3:67322fcd-7f9d-44aa-a1b9-d0260a6b9f7e'), 'updates', {'sub_subgraph_node_2': {'a': [2]}})
(('node_2:c6ccc3f6-7f13-bde7-859d-9d8401fc88f0', 'subgraph_node_3:67322fcd-7f9d-44aa-a1b9-d0260a6b9f7e'), 'values', {'foo': 'hi! foobar a', 'a': [1, 2]})
(('node_2:c6ccc3f6-7f13-bde7-859d-9d8401fc88f0',), 'updates', {'subgraph_node_3': {'foo': 'hi! foobar a'}})
(('node_2:c6ccc3f6-7f13-bde7-859d-9d8401fc88f0',), 'values', {'foo': 'hi! foobar a', 'bar': 'bar'})
((), 'updates', {'node_2': {'foo': 'hi! foobar a'}})
((), 'values', {'foo': 'hi! foobar a'})
```

###### custome

[custome mode](https://langchain-ai.github.io/langgraph/how-tos/streaming/#stream-custom-data) 的目的是为了从 node（或工具）中返回你自定义的数据，具体的操作步骤可以参考[这里](https://langchain-ai.github.io/langgraph/how-tos/streaming/#stream-custom-data)

> :full_moon:
>
> 在 [.stream()](https://langchain-ai.github.io/langgraph/reference/pregel/#langgraph.pregel.Pregel.stream) 方法中有这样一个参数 `output_keys`，其目的是「规定要流式传输的通道，默认为全部（有值）的通道」
>
> 参数 `output_keys` 与 custome mode 不同之处在于：custome mode 要传输内容不一定是图的某个通道，有可能是特定与节点中的一些内容，这些内容不在图状态的定义中，但你又需要；而 `output_keys` 参数只能限制存在于图状态下的通道参与流式传输与否

###### 组合不同的流式类型

此外，还可以[组合多种类型进行流式传输](https://langchain-ai.github.io/langgraph/how-tos/streaming/#stream-custom-data)。要实现这一点，只需在图调用 `.stream()` 方法时，向参数 `stream_mode` 传递一个列表，列表中的元素是所有你希望流式传输的类型

组合流式输出的输出将是 `(mode, chunk)` 元组，其中 `mode` 是流模式的名称， `chunk` 是该模式流式传输的数据



#### 快照     

​                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              

#### add_conditional_edges 的使用方法

*参考资料：*

1. *[Conditional branching](https://langchain-ai.github.io/langgraph/how-tos/graph-api/#conditional-branching)*
2. *[add_conditional_edges API](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph.add_conditional_edges)*

简单来说，`add_conditional_edges` 接受 3 个参数，分别是：

- source，条件边的「开始节点」，类型为 str

- path，确定下一个节点应当是什么的可调用对象（一般为函数）

  这个可调用对象应当返回一个或一些字符串，这些字符串可以是节点的名称，也可以是一些用户自定义的内容

  - 如果返回的是一个或一些的节点的名称，则这些节点就是条件边可以路由到的节点

  - 如果返回的是一些用户自定义的内容，则下一个参数是必填的，类型是一个字典

    其中，字典的键将是用户自定义的内容，键对应的值将是某个节点的名称

    这样，LangGraph 利用这个字典的键值对完成了「用户自定义内容」和「某个节点」间的映射关系。并且，用户自定义的内容在可视化时会体现在条件边上

  > :fire:
  >
  > 需要注意的是，在定义条件边时，如果
  >
  > 1. 没有为 path 函数添加返回值提供类型提示
  > 2. 并且在之后也没有在条件边中提供第三参数 `path_map` 来说明可能的终止节点是什么
  >
  > 那么在可视化时，条件边将被认为可以连接到图中的任何一个节点
  >
  > 不过只要提供了上述两个条件中的任何一个，可视化就是正常的，比如下面两个例子：
  >
  > - [提供返回值类型说明，为提供条件边的 path_map 参数](https://langchain-ai.github.io/langgraph/how-tos/graph-api/#conditional-branching)
  > - [提供了条件边的 path_map 参数，为提供返回值类型说明](https://github.langchain.ac.cn/langgraph/how-tos/async/#define-the-graph)

- path_map，其可以是 list 或 dict

  - 当 path_map 为 list 时，list 中的元素是条件边可能路由到的节点的名称

    > list 的作用是：在当条件边的 path 函数无法提供返回值的类型说明（比如要使用 Send API 并行执行时），如果也不提供 path_map 说明可能终止的节点是什么，那么在可视化图时将会造成混乱
    >
    > 这一点可以在下面的代码中看到：
    >
    > ```python
    > from langgraph.graph import START, StateGraph, END
    > from langgraph.types import Send
    > from langchain_core.documents import Document
    > from typing import TypedDict, List, Annotated
    > import operator
    > 
    > 
    > # Define subgraph
    > class SubgraphState(TypedDict):
    >     similar_chunk: List[Document]
    >     bar: str
    > 
    > def subgraph_node_1(state: SubgraphState):
    >     return {"bar": "bar"}
    > 
    > def subgraph_node_2(state: SubgraphState):
    >     state['similar_chunk'][0].page_content = 'hello again'
    >     return 
    > 
    > subgraph_builder = StateGraph(SubgraphState)
    > subgraph_builder.add_node(subgraph_node_1)
    > subgraph_builder.add_node(subgraph_node_2)
    > subgraph_builder.add_edge(START, "subgraph_node_1")
    > subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
    > subgraph_builder.add_edge("subgraph_node_2", END)
    > subgraph = subgraph_builder.compile()
    > 
    > # Define parent graph
    > class ParentState(TypedDict):
    >     similar_chunks: List[Document]
    > 
    > def node_1(state: ParentState):
    >     # 在这里，即使我没有通过字典显示返回 similar_chunks 这个通道的更新
    >     # 但是我仍通过引用修改了这个通道的值
    >     state['similar_chunks'][0].page_content = 'hello world'
    > 
    >     return 
    > 
    > # node_1 和 node_2 之间通过 Send API 传递数据，因为这里父子图间并没有共享键
    > def node1_to_node2(state: ParentState):
    >     return [
    >         Send(
    >             'node_2',
    >             {
    >                 'similar_chunk': [state['similar_chunks'][1]],
    >             }
    >         )
    >     ]
    > 
    > def node_3(state: ParentState):
    >     # 这里的 state['similar_chunks'] 仍然是父图的值
    >     state['similar_chunks'][0].page_content = 'hello world again'
    >     return
    > 
    > 
    > builder = StateGraph(ParentState)
    > builder.add_node("node_1", node_1)
    > builder.add_node("node_2", subgraph)
    > builder.add_node("node_3", node_3)
    > builder.add_edge(START, "node_1")
    > builder.add_conditional_edges(
    >     'node_1',
    >     node1_to_node2,
    >     # ['node_2']
    >     # ['node_2', 'node_3']  # 如果不指定可能终止的节点，那么在可视化时 node_1 将和所有可能的节点相连
    > )
    > builder.add_edge("node_2", "node_3")
    > builder.add_edge("node_3", END)
    > graph = builder.compile()
    > 
    > from IPython.display import Image, display
    > display(Image(graph.get_graph(xray=1).draw_mermaid_png()))
    > ```

  - 当 path_map 为 dict 时，dict 中的键值对是「用户自定义内容」:「对应希望链接到的节点名称」


