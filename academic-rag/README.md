# ScholarLens · 学术论文智能问答系统

基于 **RAG + Agent** 的学术论文问答服务。上传 PDF 后即可对论文内容进行语义检索、LLM 问答和多轮对话。

## 项目简介

ScholarLens 将论文 PDF 解析、向量化、语义检索与大语言模型问答串联为一条完整的服务链路。支持标准 RAG 和 Agent 多轮工具调用两种问答模式，内置轻量前端工作台，无需额外构建即可直接使用。

**核心能力：**

- 上传 PDF，自动切块、向量化并持久化到 SQLite + FAISS
- FAISS 语义检索 + BM25 混合检索，可选 Reranker 二次排序
- 标准 RAG 模式与 Agent 模式（支持多轮工具调用）
- 按 `session_id` 隔离的短期会话记忆
- SSE 流式返回；SQLite TTL 缓存 embedding 与检索结果
- 兼容 OpenAI 接口协议（DeepSeek / OpenAI / Qwen 等均可接入）

## 架构

```
academic-rag/
├── main.py                    # FastAPI 服务入口
├── config.yaml                # LLM / embedding / 检索参数
├── src/
│   ├── rag/
│   │   ├── embedder.py        # 向量化（BAAI/bge-small-en-v1.5）
│   │   ├── retriever.py       # FAISS 语义检索
│   │   ├── bm25_retriever.py  # BM25 关键词检索
│   │   ├── reranker.py        # 交叉编码器重排序
│   │   ├── generator.py       # LLM 生成
│   │   └── pipeline.py        # RAG 流水线编排
│   ├── agent/
│   │   └── agent.py           # ReAct Agent + 工具调用
│   ├── storage/
│   │   ├── sqlite_store.py    # 论文与 chunk 元数据持久化
│   │   ├── app_store.py       # 会话消息与索引任务状态
│   │   └── sqlite_cache.py    # 本地持久化 TTL 缓存
│   ├── mcp/
│   │   ├── server.py          # MCP 服务端
│   │   └── tools.py           # MCP 工具定义
│   ├── shared/
│   │   └── context.py         # 跨模块共享状态
│   └── utils/
│       ├── config.py          # 配置加载
│       ├── pdf_parser.py      # PDF 解析 + 切块
│       └── cache.py           # 本地 LRU 缓存
├── scripts/                   # 批量索引、评测、压测脚本
├── data/
│   ├── faiss_index/           # FAISS 索引 + SQLite 文件
│   └── papers/                # 前端上传的 PDF 存放目录
├── frontend/                  # 内置前端（HTML + JS + CSS）
└── tests/
```

**技术栈：** FastAPI · FAISS · SQLite · sentence-transformers · OpenAI-compatible LLM

## 快速开始

```bash
pip install -r requirements.txt   # 安装依赖
$env:DS_API_KEY="your_api_key"    # 设置 API Key（PowerShell）
python main.py                    # 启动服务，访问 http://localhost:8011
```

PDF 上传后由 FastAPI 进程内后台任务完成索引，不需要启动外部缓存服务或独立 Worker。
论文和 chunk 元数据持久化到索引目录中的 SQLite，向量持久化到 FAISS。
`data/app.sqlite` 持久化完整会话消息、索引任务状态、query embedding 缓存和检索结果缓存；Agent 默认只读取每个会话最近 6 轮作为上下文。
