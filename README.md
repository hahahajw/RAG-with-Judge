在 Naive RAG 的基础上添加了 Judge 模块，Judge 模块中会调用 LLM 用于判断当前检索到的知识对回答问题的帮助程度，并在需要时在当前检索内容的基础上生成新问题以递归调用带有 Judge 模块的 RAG。递归调用将动态生成一颗搜索树，用于捕获回答问题所需的知识

![image-20250527171449427](https://cdn.jsdelivr.net/gh/hahahajw/MyBlogImg@main/img/202509060948241.png)

带有 Judge 模块的 RAG 是为了解决标准间的引用问题，对于一般的多跳问题也有一定的效果。在 HotpotQA 数据集上，EM 和 F1 指标分别为 52.40% 和 67.96%

![image-20250906110919287](https://cdn.jsdelivr.net/gh/hahahajw/MyBlogImg@main/img/202509061109280.png)

更多详细信息可以在[这里](https://github.com/hahahajw/RAG-with-Judge/blob/main/note.md)找到
