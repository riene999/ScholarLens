# ScholarLens · 学术论文智能问答系统

ScholarLens 是一个面向论文阅读与科研检索的 RAG + Agent 应用。用户可以上传 PDF，围绕单篇或多篇论文提问，并查看答案对应的原始证据与论文来源。

项目内置 Web 工作台、FastAPI 接口和 MCP 工具，可直接作为本地论文助手使用，也可以接入其他 Agent 或应用。

## 主要功能

- **论文上传与管理**：上传、预览和打开 PDF，后台自动完成解析、切块与索引。
- **混合检索**：同时使用向量检索与 BM25，兼顾语义相关性、论文术语和精确关键词。
- **论文问答**：支持对单篇论文提问，也支持从整个论文库中检索并综合回答。
- **来源约束**：能够理解“这篇论文”“上一轮的方法”“前者和后者”等上下文，按用户指定的论文范围查找证据；一般知识问题仍会搜索完整论文库。
- **多轮 Agent 检索**：Agent 可以调用论文搜索、知识库检索和证据回取工具，根据中间结果继续搜索并补充信息。
- **复杂问题拆解**：可将跨论文或多步骤问题拆成多个子问题，分别检索后综合回答。
- **引用与证据追溯**：回答返回相关 chunk、论文来源和检索信息，便于回到原文核对。
- **长期会话记忆**：完整保存会话、检索证据和工具结果；长对话会自动压缩，历史证据仍可按需回取。
- **个性化知识库**：根据用户近期阅读、检索和提问记录，提供更符合个人研究方向的弱提示，并记录尚未解决的知识缺口。
- **本地持久化**：论文元数据、会话、任务状态和个性化数据保存在 SQLite，向量索引保存在 FAISS，不依赖 Redis 或外部任务队列。
- **MCP 接入**：提供论文搜索、论文问答和索引状态查询工具，可供支持 MCP 的客户端调用。

## 使用流程

1. 在文件库中上传一篇或多篇 PDF。
2. 等待后台完成论文解析和索引。
3. 选择指定论文进行定向提问，或取消选择后搜索整个论文库。
4. 在标准问答模式或 Agent 模式中输入问题。
5. 查看回答、引用来源和相关证据片段，必要时打开原始 PDF 核对。

连续追问时可以直接使用自然语言指代，例如：

```text
这篇论文提出了什么方法？
它在哪些数据集上做了实验？
前一篇和后一篇的结论有什么不同？
不要只看当前论文，从全部论文中总结常见做法。
```

## 快速开始

### 1. 创建并进入虚拟环境

PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置模型密钥

PowerShell：

```powershell
$env:DS_API_KEY="your_api_key"
```

macOS / Linux：

```bash
export DS_API_KEY="your_api_key"
```

模型地址、模型名称、Embedding、PDF 解析器和检索参数可在 `config.yaml` 中修改。项目通过 OpenAI 兼容接口调用大模型，默认配置为 DeepSeek。

### 4. 启动服务

```bash
python main.py
```

启动后访问：

- Web 工作台：<http://localhost:8011>
- 健康检查：<http://localhost:8011/health>
- OpenAPI 文档：<http://localhost:8011/docs>

PDF 索引任务由 FastAPI 进程在后台执行，不需要额外启动 Redis、Worker 或消息队列。

## 常用接口

| 接口 | 功能 |
|---|---|
| `POST /upload` | 上传 PDF 并创建索引任务 |
| `GET /jobs/{job_id}` | 查询索引任务状态 |
| `GET /documents` | 获取论文列表 |
| `GET /documents/{document_id}/preview` | 获取论文预览信息 |
| `POST /query` | 标准 RAG 问答；传入 `use_agent=true` 可启用 Agent |
| `POST /query/stream` | SSE 流式问答 |
| `POST /ask` | 返回答案、chunk 与 trace 的评测兼容接口 |
| `POST /search` | 仅检索相关证据 |
| `GET /users/{user_id}/profile` | 查看用户研究画像 |
| `DELETE /users/{user_id}/profile` | 删除用户画像与相关记录 |

## MCP 工具

启动 stdio MCP 服务：

```bash
python -m src.mcp.server
```

提供以下工具：

- `search_papers`：检索论文证据片段。
- `ask_papers`：基于已索引论文回答问题。
- `get_index_status`：查看索引和文档数量。

## 数据存储

- `data/papers/`：前端上传的 PDF。
- `data/faiss_indexes/`：FAISS 向量索引和论文 chunk 元数据。
- `data/app.sqlite`：会话、工具结果、用户画像、索引任务与本地缓存。

删除某个用户的画像不会删除论文索引。长会话压缩也不会删除原始历史和工具证据。

## 效果评测

项目在 30 篇论文上构建了包含单论文、多论文、定向和非定向问题的评测集：

- 扩大混合召回候选池后，证据段落 MRR@5 从 **0.594** 提升至 **0.641**，段落 Hit@5 从 **80%** 提升至 **92%**。
- 加入来源约束后，Top-5 正确来源段落占比从 **70.8%** 提升至 **81.6%**，多论文问题的完整来源覆盖率从 **70%** 提升至 **80%**。

数据集、指标说明和运行命令见 [`eval/README.md`](eval/README.md)。

## 技术栈

FastAPI · FAISS · SQLite · BM25 · sentence-transformers · OpenAI-compatible LLM · MCP
