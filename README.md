# 秀秀训练营 RAG 平台

基于 LangChain 的轻量级 RAG（检索增强生成）平台，为训练营知识库提供智能问答能力。

---

## 系统架构

```
┌──────────────────────────────────────────────────┐
│                    用户层                          │
│  ┌──────────┐  ┌──────────┐                      │
│  │   API    │  │  CLI     │                      │
│  └────┬─────┘  └────┬─────┘                      │
└───────┼──────────────┼────────────────────────────┘
        │              │
┌───────┼──────────────┼────────────────────────────┐
│       │    LangChain 编排层                        │
│  ┌────▼──────────────▼────┐                       │
│  │  PromptTemplate        │                       │
│  │  Retriever             │                       │
│  │  ChatOpenAI (DeepSeek) │                       │
│  │  OutputParser          │                       │
│  └────────────────────────┘                       │
│                                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │             存储层                            │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐     │  │
│  │  │ Chroma   │ │ BM25     │ │ SQLite   │     │  │
│  │  │ 向量检索  │ │ 关键词   │ │ 元数据    │     │  │
│  │  └──────────┘ └──────────┘ └──────────┘     │  │
│  └──────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────┘
```

## 技术选型

| 层级 | LangChain 组件 | 说明 |
|------|---------------|------|
| **文档加载** | `TextLoader` / `PyMuPDFLoader` | 支持 md、pdf、docx |
| **文档切分** | `RecursiveCharacterTextSplitter` | 按标题 + 递归字符切分 |
| **嵌入模型** | `HuggingFaceEmbeddings` | 本地运行，完全免费 |
| **向量数据库** | `Chroma` (via `langchain-chroma`) | 嵌入式，零运维 |
| **混合检索** | `EnsembleRetriever` | 向量 + BM25 融合 |
| **LLM** | `ChatOpenAI` (DeepSeek) | 兼容 OpenAI 格式 |
| **Prompt** | `ChatPromptTemplate` | 结构化提示词模板 |
| **Chain** | LCEL (管道符) | 声明式链式编排 |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY 等配置

# 3. 导入本地文档
python main.py import ./data/documents

# 4. 导入飞书文档
python main.py import-feishu <folder_token>

# 5. 命令行提问
python main.py ask "秀秀训练营中如何讲解js事件循环？"

# 6. 启动 API 服务
python main.py
```

## 飞书云文档导入

### 前置准备

1. 访问 [飞书开放平台](https://open.feishu.cn) 创建企业自建应用
2. 获取 `App ID` 和 `App Secret`，填入 `.env`
3. 为应用添加权限：
   - `docx:document:readonly` — 读取文档内容
   - `drive:drive:readonly` — 读取云空间文件列表
4. 创建版本、发布应用并通过审核

### 获取 folder_token

在飞书中打开目标文件夹，URL 中的 `fldcnXXXXXX` 就是 folder_token：
```
https://xxx.feishu.cn/drive/folder/fldcnXXXXXXXX
                              ^^^^^^^^^^^^^^^^^^ 这就是 folder_token
```

### 导入命令

```bash
# 导入指定飞书文件夹下的所有文档（该文件夹需要是外部文件夹：分享文件夹 -> 链接分享 -> 互联网获得链接的人 -> 可阅读(至少需要可阅读的权限)）
python main.py import-feishu fldcnXXXXXXXX

# python main.py import-feishu AKxxxQid8Kxxxxg

# 也可以通过 API 导入
curl -X POST "http://localhost:8000/import/feishu?folder_token=fldcnXXXXXXXX"
```

### 导入流程

```
飞书文件夹 folder_token
    │
    ▼ FeishuClient.list_documents()
获取所有 docx 文档列表
    │
    ▼ FeishuClient.get_document_blocks()
逐个拉取文档 Block 结构
    │
    ▼ blocks_to_markdown()
Block → Markdown（保留标题、列表、代码块等格式）
    │
    ▼ split_documents()
LangChain RecursiveCharacterTextSplitter 切分
    │
    ▼ add_documents()
向量化 + 存入 Chroma
```

## 配置项

```env
# DeepSeek
DEEPSEEK_API_KEY=sk-xxx
LLM_MODEL=deepseek-chat

# 本地嵌入模型（免费）
EMBEDDING_MODEL=all-MiniLM-L6-v2

# 检索
RETRIEVAL_TOP_K=5
CHUNK_SIZE=512

# HuggingFace 国内镜像
HF_ENDPOINT=https://hf-mirror.com
```

## 项目结构

```
llm-demo/
├── main.py                          # 入口：CLI + API 服务
├── requirements.txt
├── .env
├── app/
│   ├── config.py                    # 配置管理
│   ├── ingestion/
│   │   └── loader.py               # LangChain 文档加载器
│   ├── chunking/
│   │   └── splitter.py             # LangChain 文本切分器
│   ├── embedding/
│   │   └── local_embed.py          # LangChain HuggingFace 嵌入
│   ├── storage/
│   │   ├── db.py                   # SQLite 元数据存储
│   │   └── vector_store.py         # LangChain Chroma 向量库
│   ├── retrieval/
│   │   ├── hybrid.py               # LangChain EnsembleRetriever
│   │   └── reranker.py             # LLM 重排序
│   ├── generation/
│   │   ├── llm.py                  # LangChain ChatOpenAI
│   │   └── prompt_templates.py     # LangChain PromptTemplate
│   ├── pipeline/
│   │   ├── ingestion_pipeline.py   # 文档导入流水线
│   │   └── rag_pipeline.py         # RAG 问答链
│   └── api/
│       └── routes.py               # FastAPI 接口
└── data/
    ├── documents/                   # 文档目录
    ├── chroma/                      # Chroma 持久化
    └── rag.db                       # SQLite 数据库
```

## LangChain 核心概念映射

| 概念 | 本项目实现 |
|------|-----------|
| **Document Loader** | `TextLoader` 加载 md 文件 |
| **Text Splitter** | `RecursiveCharacterTextSplitter` 按标题递归切分 |
| **Embeddings** | `HuggingFaceEmbeddings` (all-MiniLM-L6-v2) |
| **Vector Store** | `Chroma` 本地持久化 |
| **Retriever** | `vectorstore.as_retriever()` + `EnsembleRetriever` |
| **Prompt Template** | `ChatPromptTemplate` |
| **LLM** | `ChatOpenAI(base_url=DeepSeek)` |
| **Chain (LCEL)** | `retriever \| prompt \| llm \| parser` |
