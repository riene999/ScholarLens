# ScholarLens · 学术论文智能问答系统

基于 **RAG + Agent** 的学术论文问答服务。上传 PDF 后即可对论文内容进行语义检索、LLM 问答和多轮对话。

## 项目简介

ScholarLens 将论文 PDF 解析、向量化、语义检索与大语言模型问答串联为一条完整的服务链路。支持标准 RAG 和 Agent 多轮工具调用两种问答模式，内置轻量前端工作台，无需额外构建即可直接使用。

**核心能力：**

- 上传 PDF，自动切块、向量化并持久化到 SQLite + FAISS
- FAISS 语义检索 + BM25 混合检索，可选 Reranker 二次排序
- 标准 RAG 模式与 Agent 模式（支持多轮工具调用）
- 按 `session_id` 隔离的持久化上下文记忆，支持 token 阈值压缩与证据回查
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
│   │   ├── agent.py           # ReAct Agent + 工具调用
│   │   └── context_memory.py  # 上下文预算、artifact 与累计摘要
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
`data/app.sqlite` 持久化完整会话、工具结果、RAG 召回证据、累计摘要、索引任务状态、query embedding 缓存和检索结果缓存。原始历史不会因为上下文压缩而删除。

上下文记忆默认在约 75k token 时进入软压缩：最近 3 个完整轮次保留工具和 RAG 证据原文，更早的证据改为 `artifact://sha256/<hash>`、摘要和回查提示。在约 100k token 时进入硬压缩：从最早的完整轮次开始累计超过 50k token，将“旧累计摘要 + 这批历史”重新总结。阈值和单项证据预算均可在 `config.yaml` 的 `memory` 节调整。

Agent 可通过 `get_context_artifact` 工具按 SHA256 恢复被引用的原始证据。Token 数使用离线保守估算并乘以安全系数，不依赖运行时下载 tokenizer。

## 用户研究画像

前端会为当前浏览器生成一个本地匿名 `user_id`，并向 `POST /events/batch` 批量上报现有界面中的研究行为：论文预览与有效停留、打开原始 PDF、加入检索范围、点击证据、文件库/证据搜索、上传论文、提问及回答结果。事件通过 `event_uid` 幂等去重，不记录鼠标轨迹或无关按键。

`data/app.sqlite` 中的个性化数据分为三层：

- `user_events`：不可变原始事件，可用于重新计算画像。
- `user_paper_stats`：每位用户对每篇论文的打开频率、最近时间、有效阅读时长、证据点击和提问次数等聚合值。
- `user_knowledge_gaps`：搜索无结果、回答无证据或失败形成的潜在知识缺口；同一问题后续获得证据时可标记为已解决。

`GET /users/{user_id}/profile` 可查看聚合画像，`DELETE /users/{user_id}/profile` 可删除该用户的事件和画像。标准 RAG 和 Agent 会读取精简后的高频/近期论文及未解决缺口作为个性化弱提示；系统不会因为用户打开过一篇论文就推断用户已经理解它。当前版本使用浏览器匿名 ID，若部署为真正的多用户系统，应将它替换为后端认证身份。

## 自适应文档范围

未在界面中选定论文时，系统先判断用户是否真的施加了论文来源约束，而不是按主题替问题推荐论文。明确论文名、当前预览中的“这篇论文”、上一轮唯一论文后的“它/该方法”，以及“前者/后者/刚才两篇”等可验证指代会在本地直接解析；一般概念、开放问题和宽泛综述直接使用全局检索。只有没有明显指代词、但可能延续上一轮唯一论文的省略式追问才交给轻量 LLM。路由器读取最近三轮实际召回的结构化论文来源，不依赖从回答文字中猜测论文。

前端/API 明确传入的 `source_names` 始终是硬约束。经过程序校验的明确标题、唯一上下文指代和当前论文指代，在置信度达到 `hard_confidence` 且用户没有要求外围材料时也会成为硬约束；依据无法验证、指代不唯一或需要外部背景时不会硬过滤。路由失败、超时或返回无效 JSON 时继续使用全局结果。Agent 不阻塞等待路由器；它执行检索工具时只读取当时已经完成的结果。`/query`、`/search` 和 SSE 的 `routing` 事件会返回约束依据、校验状态与候选数量。

相关参数位于 `config.yaml` 的 `source_routing`：`grace_ms` 是普通 RAG 在全局召回结束后最多补等的时间，`total_timeout_ms` 是路由调用总超时，`global_candidate_k` 是全局候选池大小，`scoped_candidate_min` 是触发定向补召回的最低候选数，`soft_confidence` 控制是否采纳已验证范围，`hard_confidence` 控制是否升级为严格来源约束。关闭 `enabled` 即恢复纯全局检索（显式 `source_names` 仍有效）。

## 检索评测

项目提供了一套基于当前 30 篇论文人工整理的 50 题测评集，其中包含 40 道单论文题和 10 道多论文综合题，并标注正确论文、文本片段、参考答案与关键词。运行方式、MRR、多来源覆盖率等指标定义以及实测结果见 [`eval/README.md`](eval/README.md)。
