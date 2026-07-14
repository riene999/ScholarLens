import asyncio
import json
import re
import time
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Optional
from uuid import uuid4

import uvicorn
import PyPDF2
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

from src.agent.agent import PaperAgent
from src.rag.pipeline import RAGPipeline, RAGResponse
from src.shared.context import create_pipeline
from src.storage.app_store import SQLiteAppStore
from src.utils.pdf_parser import extract_paper_title, normalize_title


rag_pipeline: Optional[RAGPipeline] = None
paper_agent: Optional[PaperAgent] = None
index_mtime: Optional[float] = None
app_store: Optional[SQLiteAppStore] = None

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
PAPER_DIR = BASE_DIR / "data" / "papers"
PDF_SEARCH_DIRS = [PAPER_DIR, BASE_DIR / "pdf", UPLOAD_DIR]
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline, paper_agent, index_mtime, app_store
    logger.info("服务启动，加载模型...")
    rag_pipeline = create_pipeline("config.yaml")
    app_store = SQLiteAppStore(rag_pipeline.config.storage.local_db_path)
    recovered_jobs = app_store.recover_interrupted_jobs()
    if recovered_jobs:
        logger.warning("Marked {} interrupted index job(s) as failed", recovered_jobs)
    paper_agent = PaperAgent(
        rag_pipeline,
        rag_pipeline.config.llm,
        conversation_store=app_store,
    )
    index_mtime = _get_index_mtime()
    logger.info("模型加载完成，服务就绪")
    yield
    logger.info("服务关闭")


app = FastAPI(
    title="学术论文 RAG 问答系统",
    description="基于 RAG 和 Agent 的学术论文智能问答，支持 PDF 后台索引和语义检索。",
    version="1.2.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str
    use_agent: bool = False
    session_id: str = "default"
    use_memory: bool = True
    source_names: Optional[list[str]] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list
    question: str
    session_id: str


class AskRequest(BaseModel):
    question: str
    stream: bool = False
    top_k: Optional[int] = None
    case_id: Optional[str] = None
    source_names: Optional[list[str]] = None


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    score_threshold: Optional[float] = None
    source_names: Optional[list[str]] = None


class IndexJobResponse(BaseModel):
    job_id: str
    status: str
    filename: str
    status_url: str


class ClearMemoryRequest(BaseModel):
    session_id: Optional[str] = None


class DocumentPreviewResponse(BaseModel):
    document_id: int
    source_name: str
    has_pdf: bool
    pdf_url: Optional[str] = None
    preview_type: str = "preview"
    preview_text: str


async def _run_sync(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))


async def _run_with_faiss_lock(func, *args, **kwargs):
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized")

    if not hasattr(rag_pipeline, "retriever") or not hasattr(rag_pipeline.retriever, "faiss_lock"):
        return await _run_sync(func, *args, **kwargs)

    async with rag_pipeline.retriever.faiss_lock:
        return await _run_sync(func, *args, **kwargs)


def _get_index_mtime() -> Optional[float]:
    if rag_pipeline is None or not hasattr(rag_pipeline, "config"):
        return None
    if hasattr(rag_pipeline.retriever, "document_store"):
        return float(rag_pipeline.retriever.document_store.current_version())

    index_path = Path(rag_pipeline.config.vector_store.index_path)
    version_file = index_path / ".index_version"
    marker_file = version_file if version_file.exists() else index_path / "index.faiss"
    if not marker_file.exists():
        return None
    return marker_file.stat().st_mtime


async def _reload_index_if_changed() -> None:
    global index_mtime
    if rag_pipeline is None:
        return

    current_mtime = _get_index_mtime()
    if current_mtime is None or current_mtime == index_mtime:
        return

    async with rag_pipeline.retriever.faiss_lock:
        current_mtime = _get_index_mtime()
        if current_mtime is None or current_mtime == index_mtime:
            return

        loaded = await _run_sync(rag_pipeline.retriever.load)
        if loaded and rag_pipeline.bm25_retriever is not None:
            rag_pipeline.bm25_retriever.load_documents(rag_pipeline.retriever.documents)
        index_mtime = current_mtime
        logger.info("检测到后台索引更新，已重新加载 FAISS/BM25")


def _create_index_job(job_id: str, filename: str) -> dict:
    if app_store is None:
        raise HTTPException(status_code=503, detail="Local app store is not initialized")
    return app_store.create_index_job(job_id, filename)


async def _run_index_job(job_id: str, pdf_path: Path, source_name: str) -> None:
    global index_mtime

    if app_store is None:
        logger.error("Cannot run index job {}: local app store is unavailable", job_id)
        return

    app_store.mark_index_job_started(job_id)
    logger.info("Starting local PDF index job: {}", source_name)

    try:
        chunks_added = await _run_with_faiss_lock(
            rag_pipeline.index_documents_from_pdf,
            str(pdf_path),
            source_name=source_name,
        )
        app_store.mark_index_job_finished(job_id, chunks_added)
        index_mtime = _get_index_mtime()
        logger.info("Finished local PDF index job: {}, chunks={}", source_name, chunks_added)
    except Exception as exc:
        app_store.mark_index_job_failed(job_id, f"{type(exc).__name__}: {exc}")
        logger.exception("Local PDF index job failed: {}", source_name)


def _serialize_job(job: dict) -> dict:
    return dict(job)


def _build_memory_aware_question(question: str, session_id: str, use_memory: bool) -> str:
    if not use_memory or not paper_agent:
        return question

    history = paper_agent.memory.get_messages(session_id)
    if not history:
        return question

    history_text = "\n".join(
        f"{'用户' if item['role'] == 'user' else '助手'}: {item['content']}"
        for item in history
    )
    return (
        "下面是同一会话的历史问答。请结合历史理解当前问题中的指代，"
        "但回答仍必须基于检索到的论文证据。\n\n"
        f"历史问答：\n{history_text}\n\n"
        f"当前问题：{question}"
    )


def _remember_turn(session_id: str, question: str, answer: str, use_memory: bool) -> None:
    if use_memory and paper_agent:
        paper_agent.memory.add_turn(session_id, question, answer)


def _format_retrieved_chunk(chunk) -> dict:
    metadata = chunk.document.metadata or {}
    return {
        "chunk_id": chunk.document.chunk_id,
        "id": chunk.document.chunk_id,
        "text": chunk.document.content,
        "content": chunk.document.content,
        "score": round(float(chunk.score), 6),
        "rank": chunk.rank,
        "source": metadata.get("source"),
        "paper_title": metadata.get("paper_title"),
        "page": metadata.get("page"),
        "metadata": metadata,
    }


def _clean_source_filter(source_names: Optional[list[str]]) -> list[str] | None:
    if not source_names:
        return None
    cleaned = []
    for item in source_names:
        source = Path(str(item)).name
        if source and source not in cleaned:
            cleaned.append(source)
    return cleaned or None


def _resolve_source_filter(question: str, source_names: Optional[list[str]]) -> list[str] | None:
    explicit = _clean_source_filter(source_names)
    if explicit:
        return explicit
    return _infer_source_filter_from_question(question)


def _infer_source_filter_from_question(question: str) -> list[str] | None:
    if rag_pipeline is None or not hasattr(rag_pipeline.retriever, "list_documents"):
        return None

    normalized_question = normalize_title(question)
    if len(normalized_question) < 8:
        return None

    matches = []
    for document in rag_pipeline.retriever.list_documents():
        source_name = str(document.get("source_name") or "")
        paper_title = str(document.get("paper_title") or Path(source_name).stem)
        keys = {
            normalize_title(source_name),
            normalize_title(paper_title),
        }
        for key in keys:
            if len(key) >= 12 and (key in normalized_question or normalized_question in key):
                matches.append(source_name)
                break

    unique_matches = []
    for match in matches:
        if match not in unique_matches:
            unique_matches.append(match)

    return unique_matches or None


def _resolve_pdf_path(source_name: str) -> Path | None:
    filename = Path(source_name).name
    if not filename.lower().endswith(".pdf"):
        return None

    for directory in PDF_SEARCH_DIRS:
        candidate = directory / filename
        if candidate.is_file():
            return candidate.resolve()

    for directory in PDF_SEARCH_DIRS:
        if not directory.exists():
            continue
        for candidate in sorted(directory.iterdir()):
            if candidate.is_file():
                candidate_name = candidate.name
                if candidate_name == filename or candidate_name.endswith(f"_{filename}"):
                    return candidate.resolve()

    return None


def _document_with_pdf_url(document: dict) -> dict:
    pdf_path = _resolve_pdf_path(str(document.get("source_name") or ""))
    enriched = dict(document)
    if not enriched.get("paper_title"):
        paper_title = None
        if pdf_path is not None:
            paper_title = _extract_title_from_pdf_file(pdf_path, str(enriched["source_name"]))
            if paper_title and rag_pipeline is not None and hasattr(rag_pipeline.retriever, "document_store"):
                rag_pipeline.retriever.document_store.update_document_title(
                    int(enriched["id"]),
                    paper_title,
                )
        enriched["paper_title"] = paper_title or Path(str(enriched.get("source_name") or "")).stem
    enriched["has_pdf"] = pdf_path is not None
    enriched["pdf_url"] = f"/documents/{document['id']}/pdf" if pdf_path else None
    return enriched


def _extract_title_from_pdf_file(pdf_path: Path, fallback: str) -> str | None:
    try:
        with pdf_path.open("rb") as file:
            reader = PyPDF2.PdfReader(file)
            return extract_paper_title(reader, fallback=fallback)
    except Exception as exc:
        logger.warning("Failed to extract paper title from {}: {}", pdf_path, exc)
        return None


def _read_pdf_pages_text(pdf_path: Path, max_pages: int = 3) -> str:
    pages: list[str] = []
    with pdf_path.open("rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages[:max_pages]:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _normalize_preview_text(text: str) -> str:
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _compact_preview_paragraph(text: str) -> str:
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" :-\n\t")


def _extract_abstract_from_text(text: str) -> str:
    normalized = _normalize_preview_text(text)
    start_match = re.search(
        r"(?im)(?:^|\n)\s*(?:abstract|摘要)\s*[:.\-]?\s*",
        normalized,
    )
    if not start_match:
        return ""

    body = normalized[start_match.end():]
    stop_match = re.search(
        r"(?im)"
        r"(?:^|\n)\s*(?:"
        r"keywords?|index terms?|"
        r"1\s*[\.\-]?\s*introduction|"
        r"i\s*[\.\-]?\s*introduction|"
        r"introduction|"
        r"background|related work|"
        r"摘要|关键词|引言"
        r")\b",
        body,
    )
    if stop_match:
        body = body[: stop_match.start()]

    abstract = _compact_preview_paragraph(body)
    if len(abstract) < 80:
        return ""
    return abstract[:2500]


FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def web_app():
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend assets are not available")
    return FileResponse(index_file)


@app.get("/health")
async def health_check():
    await _reload_index_if_changed()
    index_size = 0
    if rag_pipeline:
        async with rag_pipeline.retriever.faiss_lock:
            index_size = rag_pipeline.retriever.index.ntotal
    return {
        "status": "ok",
        "index_size": index_size,
        "memory_sessions": paper_agent.memory.session_count() if paper_agent else 0,
    }


@app.post("/upload", response_model=IndexJobResponse)
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持 PDF 文件")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="PDF 文件过大")

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    job_id = uuid4().hex
    filename = Path(file.filename).name
    pdf_path = PAPER_DIR / filename
    pdf_path.write_bytes(content)

    job = _create_index_job(job_id, filename)
    background_tasks.add_task(_run_index_job, job_id, pdf_path, filename)

    return IndexJobResponse(
        job_id=job_id,
        status=job["status"],
        filename=filename,
        status_url=f"/jobs/{job_id}",
    )


@app.get("/jobs/{job_id}")
async def get_index_job(job_id: str):
    if app_store is None:
        raise HTTPException(status_code=503, detail="Local app store is not initialized")
    job = app_store.get_index_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return _serialize_job(job)


@app.get("/documents")
async def list_documents():
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized")

    await _reload_index_if_changed()
    documents = [
        _document_with_pdf_url(document)
        for document in rag_pipeline.retriever.list_documents()
    ]
    return {"documents": documents}


@app.get("/documents/{document_id}/pdf")
async def get_document_pdf(document_id: int):
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized")
    if not hasattr(rag_pipeline.retriever, "document_store"):
        raise HTTPException(status_code=404, detail="Document store is not available")

    document = rag_pipeline.retriever.document_store.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    pdf_path = _resolve_pdf_path(str(document["source_name"]))
    if pdf_path is None:
        raise HTTPException(status_code=404, detail="PDF file is not available")

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=Path(document["source_name"]).name,
        content_disposition_type="inline",
    )


@app.get("/documents/{document_id}/preview", response_model=DocumentPreviewResponse)
async def get_document_preview(document_id: int):
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized")
    if not hasattr(rag_pipeline.retriever, "document_store"):
        raise HTTPException(status_code=404, detail="Document store is not available")

    document = rag_pipeline.retriever.document_store.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    source_name = str(document["source_name"])
    pdf_path = _resolve_pdf_path(source_name)
    pdf_url = f"/documents/{document_id}/pdf" if pdf_path else None
    preview_text = ""
    preview_type = "preview"

    if pdf_path is not None:
        try:
            raw_text = _read_pdf_pages_text(pdf_path)
            abstract = _extract_abstract_from_text(raw_text)
            if abstract:
                preview_text = abstract
                preview_type = "abstract"
            else:
                preview_text = _normalize_preview_text(raw_text)[:6000]
        except Exception as exc:
            logger.warning("Failed to parse PDF preview {}: {}", pdf_path, exc)

    if not preview_text:
        preview_chunks = [
            doc.content
            for doc in getattr(rag_pipeline.retriever, "documents", [])
            if (doc.metadata or {}).get("source") == source_name
        ][:3]
        preview_text = "\n\n".join(preview_chunks)

    return DocumentPreviewResponse(
        document_id=document_id,
        source_name=source_name,
        has_pdf=pdf_path is not None,
        pdf_url=pdf_url,
        preview_type=preview_type,
        preview_text=preview_text[:6000],
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    await _reload_index_if_changed()
    source_filter = _resolve_source_filter(request.question, request.source_names)

    if request.use_agent:
        answer = await _run_with_faiss_lock(
            paper_agent.run,
            request.question,
            session_id=request.session_id,
            use_memory=request.use_memory,
            source_names=source_filter,
        )
        return QueryResponse(
            answer=answer,
            sources=[],
            question=request.question,
            session_id=request.session_id,
        )

    effective_question = _build_memory_aware_question(
        request.question,
        request.session_id,
        request.use_memory,
    )
    chunks = await _run_with_faiss_lock(
        rag_pipeline.retrieve_chunks,
        request.question,
        source_filter=source_filter,
    )
    answer = await _run_sync(rag_pipeline.generator.generate, effective_question, chunks)
    response = RAGResponse(answer=answer, retrieved_chunks=chunks, query=request.question)
    _remember_turn(request.session_id, request.question, response.answer, request.use_memory)
    sources = [
        {
            "source": chunk.document.metadata.get("source"),
            "page": chunk.document.metadata.get("page"),
            "score": round(chunk.score, 4),
            "content_preview": chunk.document.content[:150] + "...",
        }
        for chunk in response.retrieved_chunks
    ]
    return QueryResponse(
        answer=response.answer,
        sources=sources,
        question=request.question,
        session_id=request.session_id,
    )


@app.post("/ask")
async def ask(request: AskRequest):
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized")
    if request.stream:
        logger.info("/ask received stream=true; returning non-streaming JSON for eval compatibility")

    await _reload_index_if_changed()
    source_filter = _resolve_source_filter(request.question, request.source_names)
    start = time.perf_counter()
    chunks = await _run_with_faiss_lock(
        rag_pipeline.retrieve_chunks,
        request.question,
        top_k=request.top_k,
        source_filter=source_filter,
    )
    answer = await _run_sync(rag_pipeline.generator.generate, request.question, chunks)
    response = RAGResponse(answer=answer, retrieved_chunks=chunks, query=request.question)
    retrieved_chunks = [_format_retrieved_chunk(chunk) for chunk in response.retrieved_chunks]
    citations = [chunk["chunk_id"] for chunk in retrieved_chunks]
    latency_ms = int((time.perf_counter() - start) * 1000)

    return {
        "answer": response.answer,
        "retrieved_chunks": retrieved_chunks,
        "citations": citations,
        "trace": {
            "tools": [],
            "rewrite_query": response.query,
            "steps": [
                {
                    "name": "rag_query",
                    "question": request.question,
                    "top_k": request.top_k,
                    "retrieved_count": len(retrieved_chunks),
                }
            ],
            "case_id": request.case_id,
        },
        "latency_ms": latency_ms,
    }


@app.post("/search")
async def search_papers(request: SearchRequest):
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized")

    await _reload_index_if_changed()
    source_filter = _resolve_source_filter(request.query, request.source_names)
    chunks = await _run_with_faiss_lock(
        rag_pipeline.retrieve_chunks,
        request.query,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        use_reranker=True,
        source_filter=source_filter,
    )
    return {
        "query": request.query,
        "retrieved_chunks": [_format_retrieved_chunk(chunk) for chunk in chunks],
    }


@app.post("/memory/clear")
async def clear_memory(request: ClearMemoryRequest):
    if not paper_agent:
        raise HTTPException(status_code=503, detail="Agent 尚未初始化")
    paper_agent.clear_memory(request.session_id)
    return {"status": "ok", "cleared_session_id": request.session_id}


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    await _reload_index_if_changed()
    source_filter = _resolve_source_filter(request.question, request.source_names)

    if request.use_agent:
        answer = await _run_with_faiss_lock(
            paper_agent.run,
            request.question,
            session_id=request.session_id,
            use_memory=request.use_memory,
            source_names=source_filter,
        )

        def generate_agent():
            yield f"data: {json.dumps({'type': 'sources', 'data': []}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'data': answer}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

        return StreamingResponse(generate_agent(), media_type="text/event-stream")

    effective_question = _build_memory_aware_question(
        request.question,
        request.session_id,
        request.use_memory,
    )
    chunks = await _run_with_faiss_lock(
        rag_pipeline.retrieve_chunks,
        request.question,
        source_filter=source_filter,
    )
    stream = rag_pipeline.generator.generate_stream(effective_question, chunks)

    def generate():
        sources = [
            {
                "source": c.document.metadata.get("source"),
                "page": c.document.metadata.get("page"),
                "score": round(c.score, 4),
                "chunk_id": c.document.chunk_id,
                "content_preview": c.document.content[:220] + (
                    "..." if len(c.document.content) > 220 else ""
                ),
            }
            for c in chunks
        ]
        yield f"data: {json.dumps({'type': 'sources', 'data': sources}, ensure_ascii=False)}\n\n"

        answer_parts = []
        for token in stream:
            answer_parts.append(token)
            yield f"data: {json.dumps({'type': 'token', 'data': token}, ensure_ascii=False)}\n\n"

        _remember_turn(
            request.session_id,
            request.question,
            "".join(answer_parts),
            request.use_memory,
        )
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/index/text")
async def index_text(text: str, source_name: str = "manual"):
    chunks_added = await _run_with_faiss_lock(rag_pipeline.index_text, text, source_name)
    return {"chunks_added": chunks_added}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8011, reload=True)
