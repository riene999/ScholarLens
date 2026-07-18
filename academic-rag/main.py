import asyncio
import json
import re
import time
from contextlib import asynccontextmanager, suppress
from functools import partial
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import uvicorn
import PyPDF2
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field

from src.agent.agent import PaperAgent
from src.rag.pipeline import RAGPipeline, RAGResponse
from src.rag.source_router import SourcePlan, SourcePlanHandle, SourceRouter
from src.shared.context import create_pipeline
from src.storage.app_store import SQLiteAppStore
from src.utils.pdf_parser import extract_paper_title, normalize_title


rag_pipeline: Optional[RAGPipeline] = None
paper_agent: Optional[PaperAgent] = None
index_mtime: Optional[float] = None
app_store: Optional[SQLiteAppStore] = None
source_router: Optional[SourceRouter] = None

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
PAPER_DIR = BASE_DIR / "data" / "papers"
PDF_SEARCH_DIRS = [PAPER_DIR, BASE_DIR / "pdf", UPLOAD_DIR]
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ALLOWED_USER_EVENT_TYPES = {
    "paper_preview_open",
    "paper_preview_close",
    "pdf_open",
    "evidence_clicked",
    "paper_scope_added",
    "paper_scope_removed",
    "library_search",
    "search_completed",
    "pdf_uploaded",
    "question_submitted",
    "answer_completed",
    "answer_failed",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline, paper_agent, index_mtime, app_store, source_router
    logger.info("服务启动，加载模型...")
    rag_pipeline = create_pipeline("config.yaml")
    app_store = SQLiteAppStore(rag_pipeline.config.storage.local_db_path)
    if rag_pipeline.config.source_routing.enabled:
        source_router = SourceRouter(
            rag_pipeline.config.llm,
            rag_pipeline.config.source_routing,
        )
    recovered_jobs = app_store.recover_interrupted_jobs()
    if recovered_jobs:
        logger.warning("Marked {} interrupted index job(s) as failed", recovered_jobs)
    paper_agent = PaperAgent(
        rag_pipeline,
        rag_pipeline.config.llm,
        conversation_store=app_store,
        memory_config=rag_pipeline.config.memory,
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
    user_id: Optional[str] = None
    use_memory: bool = True
    source_names: Optional[list[str]] = None
    current_document_id: Optional[int] = None
    current_source_name: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list
    question: str
    session_id: str
    routing: Optional[dict] = None


class AskRequest(BaseModel):
    question: str
    stream: bool = False
    top_k: Optional[int] = None
    case_id: Optional[str] = None
    source_names: Optional[list[str]] = None


class SearchRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    top_k: Optional[int] = None
    score_threshold: Optional[float] = None
    source_names: Optional[list[str]] = None
    current_document_id: Optional[int] = None
    current_source_name: Optional[str] = None


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


class UserEventRequest(BaseModel):
    event_uid: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    event_type: str = Field(min_length=1, max_length=64)
    session_id: Optional[str] = Field(default=None, max_length=128)
    document_id: Optional[int] = None
    source_name: Optional[str] = Field(default=None, max_length=500)
    occurred_at: Optional[str] = Field(default=None, max_length=64)
    properties: dict[str, Any] = Field(default_factory=dict)


class UserEventBatchRequest(BaseModel):
    events: list[UserEventRequest]


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


def _build_memory_aware_question(
    question: str,
    session_id: str,
    use_memory: bool,
    user_id: str | None = None,
) -> str:
    profile_context = (
        app_store.build_user_profile_context(user_id)
        if app_store is not None and user_id
        else ""
    )
    if not use_memory or not paper_agent:
        if not profile_context:
            return question
        return f"{profile_context}\n\nCurrent question:\n{question}"

    history = paper_agent.memory.get_messages(session_id, pending_content=question)
    if not history and not profile_context:
        return question

    history_text = "\n".join(
        f"{'用户' if item['role'] == 'user' else '助手'}: {item['content']}"
        for item in history
    )
    sections = [
        "请结合会话历史和用户研究画像理解当前问题，但回答仍必须基于检索到的论文证据。"
    ]
    if profile_context:
        sections.append(profile_context)
    if history_text:
        sections.append(f"历史问答：\n{history_text}")
    sections.append(f"当前问题：{question}")
    return "\n\n".join(sections)


def _remember_turn(
    session_id: str,
    question: str,
    answer: str,
    use_memory: bool,
    retrieved_chunks: list | None = None,
) -> None:
    if use_memory and paper_agent:
        artifacts = []
        if retrieved_chunks:
            payload = {
                "query": question,
                "results": [_format_retrieved_chunk(chunk) for chunk in retrieved_chunks],
            }
            sources = list(dict.fromkeys(
                str((chunk.document.metadata or {}).get("source"))
                for chunk in retrieved_chunks
                if (chunk.document.metadata or {}).get("source")
            ))
            artifact = paper_agent.memory.store_artifact(
                "rag_retrieval",
                payload,
                metadata={"result_count": len(retrieved_chunks), "sources": sources},
                digest=(
                    f"RAG retrieval for {question!r}: {len(retrieved_chunks)} passages"
                    + (f" from {', '.join(sources)}" if sources else "")
                ),
            )
            if artifact:
                artifacts.append(artifact)
        paper_agent.memory.add_turn(
            session_id,
            question,
            answer,
            artifacts=artifacts,
        )


def _format_retrieved_chunk(chunk) -> dict:
    metadata = chunk.document.metadata or {}
    return {
        "chunk_id": chunk.document.chunk_id,
        "id": chunk.document.chunk_id,
        "text": chunk.document.content,
        "content": chunk.document.content,
        "score": round(float(chunk.score), 6),
        "rank": int(chunk.rank),
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


def _resolve_source_filter(_question: str, source_names: Optional[list[str]]) -> list[str] | None:
    """Only an explicit UI/API selection is a hard retrieval constraint."""
    return _clean_source_filter(source_names)


def _source_router_context(session_id: str | None, user_id: str | None) -> dict:
    if app_store is None:
        return {
            "conversation_summary": "",
            "recent_messages": [],
            "recent_turns": [],
            "user_profile": "",
        }
    summary = app_store.get_active_memory_summary(session_id) if session_id else None
    recent_turns = (
        app_store.get_recent_conversation_rounds(session_id, limit=3)
        if session_id
        else []
    )
    structured_turns = []
    for position, turn in enumerate(recent_turns):
        structured_turns.append({
            "turn_index": position - len(recent_turns),
            "user": str(turn.get("user_content") or "")[:2000],
            "assistant": str(turn.get("assistant_content") or "")[:2000],
            "retrieved_sources": list(turn.get("retrieved_sources") or []),
            "artifact_digests": list(turn.get("artifact_digests") or [])[:4],
        })
    return {
        "conversation_summary": str((summary or {}).get("content") or ""),
        "recent_messages": (
            app_store.get_conversation_messages(session_id, limit=6)
            if session_id
            else []
        ),
        "recent_turns": structured_turns,
        "user_profile": app_store.build_user_profile_context(user_id) if user_id else "",
    }


_ROUTER_ALIAS_PATTERN = re.compile(
    r"\b(?:Fed[A-Z][A-Za-z0-9-]*|[A-Z]{2,}[A-Za-z0-9-]{0,20}|"
    r"[A-Z][a-z]+(?:[A-Z][A-Za-z0-9]*)+)\b"
)
_ROUTER_ALIAS_STOPWORDS = {
    "ABSTRACT", "ACM", "AI", "ANALYSES", "AND", "BACKGROUND", "CIFAR",
    "CNN", "CONVERGENCE", "CPU", "DATA", "DISTRIBUTIONS", "DNN", "FL",
    "FOR", "FROM", "GPU", "HFL", "HIERARCHICAL", "IEEE", "IID",
    "INTRODUCTION", "IOT", "LEARNING", "LLM", "ML", "MNIST",
    "MULTI-BRANCH", "NETWORKS", "NON-IID", "NONIID", "PDF", "PERIODICALLY",
    "RAG", "RELATED", "RNN", "SELECTION", "SFL", "SGD", "SHIFTING",
    "THE", "TOPOLOGY", "UNDER", "UNIFIED", "WORK",
}
_ROUTER_CANONICAL_ALIAS_OWNERS = {
    "FedAvg": "Communication-Efficient Learning of Deep Networks from Decentralized Data.pdf",
    "FedProx": "MLSys-2020-federated-optimization-in-heterogeneous-networks-Paper.pdf",
    "FedSeq": "FedSeq_A_Hybrid_Federated_Learning_Framework_Based_on_Sequential_In-Cluster_Training.pdf",
    "SPFL": "SPFL Sequential Updates with Parallel Aggregation for Enhanced Federated Learning Under Category and Domain Shifts.pdf",
    "TornadoAggregate": "TornadoAggregate Accurate and Scalable Federated Learning.pdf",
    "NbAFL": "Federated_Learning_With_Differential_Privacy_Algorithms_and_Performance_Analysis(1).pdf",
    "SHARE": "SHARE Shaping Data Distribution at Edge for Communication-Efficient Hierarchical Federated Learning.pdf",
    "HSFL": "A Joint Communication and Learning Framework for Hierarchical.pdf",
}


def _extract_router_aliases(
    text: str,
    *,
    title: str = "",
    source_name: str = "",
) -> list[str]:
    aliases = []
    for region in (title, Path(source_name).stem):
        first_token = re.split(r"[\s_:]+", region.strip(), maxsplit=1)[0]
        if not _ROUTER_ALIAS_PATTERN.fullmatch(first_token):
            continue
        if first_token.upper() in _ROUTER_ALIAS_STOPWORDS or len(first_token) < 3:
            continue
        if first_token not in aliases:
            aliases.append(first_token)
    # Abstracts often mention or even describe methods introduced by other papers.
    # Only title-leading names and explicitly owned corpus aliases are safe enough
    # to participate in source-constraint detection.
    for alias, owner in _ROUTER_CANONICAL_ALIAS_OWNERS.items():
        aliases = [item for item in aliases if item.casefold() != alias.casefold()]
        if source_name == owner:
            aliases.append(alias)
    return aliases[:20]


def _source_router_documents() -> list[dict]:
    if rag_pipeline is None:
        return []
    documents = [dict(item) for item in rag_pipeline.retriever.list_documents()]
    first_chunk_titles: dict[str, str] = {}
    aliases_by_source: dict[str, list[str]] = {}
    for indexed in getattr(rag_pipeline.retriever, "documents", []):
        metadata = indexed.metadata or {}
        source = str(metadata.get("source") or "")
        if not source or source in first_chunk_titles:
            continue
        if int(metadata.get("chunk_index") or 0) != 0:
            continue
        first_line = next(
            (line.strip() for line in indexed.content.splitlines() if line.strip()),
            "",
        )
        if first_line:
            first_chunk_titles[source] = first_line[:300]
        aliases_by_source[source] = _extract_router_aliases(
            indexed.content,
            title=str(metadata.get("paper_title") or ""),
            source_name=source,
        )
    for document in documents:
        source = str(document.get("source_name") or "")
        document["title_hint"] = first_chunk_titles.get(source, "")
        document["aliases"] = aliases_by_source.get(source, [])
    return documents


async def _resolve_source_plan(
    question: str,
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    current_document_id: int | None = None,
    current_source_name: str | None = None,
) -> SourcePlan | None:
    if source_router is None or rag_pipeline is None:
        return None
    context = await _run_sync(_source_router_context, session_id, user_id)
    documents = _source_router_documents()
    timeout = max(0.05, rag_pipeline.config.source_routing.total_timeout_ms / 1000)
    try:
        return await asyncio.wait_for(
            source_router.resolve(
                question,
                documents,
                current_document_id=current_document_id,
                current_source_name=current_source_name,
                **context,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.info("Source router timed out after {} ms", int(timeout * 1000))
    except Exception:
        logger.exception("Source router failed; continuing with global retrieval")
    return None


async def _take_source_plan(
    task: asyncio.Task[SourcePlan | None] | None,
    grace_ms: int,
) -> tuple[SourcePlan | None, bool]:
    """Take a completed plan or wait briefly after retrieval, then abandon it."""
    if task is None:
        return None, False
    timed_out = False
    try:
        if task.done():
            return await task, False
        return await asyncio.wait_for(
            asyncio.shield(task),
            timeout=max(0, grace_ms) / 1000,
        ), False
    except asyncio.TimeoutError:
        timed_out = True
    except Exception:
        logger.exception("Failed to collect source routing result")
    if not task.done():
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    return None, timed_out


async def _retrieve_with_source_routing(
    question: str,
    *,
    explicit_sources: list[str] | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    current_document_id: int | None = None,
    current_source_name: str | None = None,
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> tuple[list, dict]:
    """Speculatively retrieve globally while an LLM resolves document scope."""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized")
    final_top_k = top_k or rag_pipeline.config.source_routing.final_top_k
    if explicit_sources:
        chunks = await _run_with_faiss_lock(
            rag_pipeline.retrieve_chunks,
            question,
            top_k=final_top_k,
            score_threshold=score_threshold,
            source_filter=explicit_sources,
        )
        return chunks, {
            "mode": "hard",
            "origin": "explicit_selection",
            "source_names": explicit_sources,
            "final_count": len(chunks),
        }

    config = rag_pipeline.config.source_routing
    if source_router is None or not config.enabled:
        chunks = await _run_with_faiss_lock(
            rag_pipeline.retrieve_chunks,
            question,
            top_k=final_top_k,
            score_threshold=score_threshold,
        )
        return chunks, {"mode": "global", "origin": "router_disabled", "final_count": len(chunks)}

    router_task = asyncio.create_task(
        _resolve_source_plan(
            question,
            session_id=session_id,
            user_id=user_id,
            current_document_id=current_document_id,
            current_source_name=current_source_name,
        )
    )
    try:
        global_candidates = await _run_with_faiss_lock(
            rag_pipeline.retrieve_candidates,
            question,
            candidate_k=max(final_top_k, config.global_candidate_k),
            score_threshold=score_threshold,
        )
    except Exception:
        if not router_task.done():
            router_task.cancel()
            with suppress(asyncio.CancelledError):
                await router_task
        raise
    plan, timed_out = await _take_source_plan(router_task, config.grace_ms)
    trace = {
        "mode": "global",
        "origin": "speculative_router",
        "router_timed_out": timed_out,
        "global_candidate_count": len(global_candidates),
    }
    if plan is not None:
        trace.update(plan.to_trace())

    has_validated_scope = (
        plan is not None
        and plan.has_validated_scope
        and plan.confidence >= config.soft_confidence
    )
    if not has_validated_scope:
        chunks = await _run_sync(
            rag_pipeline.finalize_candidates,
            question,
            global_candidates,
            top_k=final_top_k,
        )
        trace["final_count"] = len(chunks)
        return chunks, trace

    if (
        plan.confidence >= config.hard_confidence
        and not plan.allow_global_evidence
    ):
        chunks = await _run_with_faiss_lock(
            rag_pipeline.retrieve_chunks,
            question,
            top_k=final_top_k,
            score_threshold=score_threshold,
            source_filter=plan.source_names,
        )
        trace.update({
            "mode": "hard_contextual",
            "origin": "validated_source_constraint",
            "final_count": len(chunks),
        })
        return chunks, trace

    scoped_sources = plan.source_names
    scoped_set = set(scoped_sources)
    scoped_candidates = [
        chunk
        for chunk in global_candidates
        if (chunk.document.metadata or {}).get("source") in scoped_set
    ]
    covered_sources = {
        (chunk.document.metadata or {}).get("source") for chunk in scoped_candidates
    }
    needs_coverage = plan.intent == "comparison" and not scoped_set.issubset(covered_sources)
    needs_more = len(scoped_candidates) < config.scoped_candidate_min or needs_coverage
    scoped_extra = []
    if needs_more:
        missing_sources = (
            [source for source in scoped_sources if source not in covered_sources]
            if needs_coverage
            else scoped_sources
        )
        per_source_k = max(
            4,
            (config.scoped_candidate_min + len(missing_sources) - 1)
            // max(1, len(missing_sources)),
        )
        for source_name in missing_sources:
            scoped_extra.extend(
                await _run_with_faiss_lock(
                    rag_pipeline.retrieve_candidates,
                    question,
                    candidate_k=per_source_k,
                    score_threshold=score_threshold,
                    source_filter=[source_name],
                )
            )

    # An inferred route is never a hard constraint: preserve global candidates
    # even when the router is confident. Only explicit UI selection is hard.
    candidate_pool = global_candidates + scoped_extra
    preferred_sources = scoped_sources
    required_sources = scoped_sources if plan.intent == "comparison" else None
    trace["mode"] = "soft_scoped"

    chunks = await _run_sync(
        rag_pipeline.finalize_candidates,
        question,
        candidate_pool,
        top_k=final_top_k,
        preferred_sources=preferred_sources,
        required_sources=required_sources,
    )
    trace.update({
        "scoped_candidate_count": len(scoped_candidates),
        "scoped_reretrieved_count": len(scoped_extra),
        "comparison_sources_reserved": bool(required_sources),
        "final_count": len(chunks),
    })
    logger.info("Adaptive source retrieval trace: {}", trace)
    return chunks, trace


async def _populate_source_plan_handle(
    handle: SourcePlanHandle,
    question: str,
    **kwargs,
) -> None:
    plan = await _resolve_source_plan(question, **kwargs)
    if plan is not None:
        handle.set(plan)


async def _run_agent_with_source_routing(
    request: QueryRequest,
    explicit_sources: list[str] | None,
) -> tuple[str, dict]:
    if paper_agent is None or rag_pipeline is None:
        raise HTTPException(status_code=503, detail="Agent is not initialized")
    if explicit_sources or source_router is None:
        answer = await _run_with_faiss_lock(
            paper_agent.run,
            request.question,
            session_id=request.session_id,
            use_memory=request.use_memory,
            source_names=explicit_sources,
            user_id=request.user_id,
        )
        return answer, {
            "mode": "hard" if explicit_sources else "global",
            "origin": "explicit_selection" if explicit_sources else "router_disabled",
            "source_names": explicit_sources or [],
        }

    handle = SourcePlanHandle()
    router_task = asyncio.create_task(
        _populate_source_plan_handle(
            handle,
            request.question,
            session_id=request.session_id,
            user_id=request.user_id,
            current_document_id=request.current_document_id,
            current_source_name=request.current_source_name,
        )
    )
    try:
        answer = await _run_with_faiss_lock(
            paper_agent.run,
            request.question,
            session_id=request.session_id,
            use_memory=request.use_memory,
            source_names=None,
            user_id=request.user_id,
            source_plan_handle=handle,
        )
    finally:
        if not router_task.done():
            router_task.cancel()
            with suppress(asyncio.CancelledError):
                await router_task
    plan = handle.get_if_ready()
    trace = {"mode": "agent_non_blocking", "origin": "speculative_router"}
    if plan is not None:
        trace.update(plan.to_trace())
    else:
        trace["router_timed_out"] = True
    return answer, trace


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


@app.post("/events/batch")
async def record_user_events(request: UserEventBatchRequest):
    if app_store is None:
        raise HTTPException(status_code=503, detail="Application storage is not initialized")
    if not request.events:
        return {"accepted": 0, "duplicates": 0}
    if len(request.events) > 100:
        raise HTTPException(status_code=413, detail="At most 100 events are accepted per batch")
    unknown = sorted({
        event.event_type
        for event in request.events
        if event.event_type not in ALLOWED_USER_EVENT_TYPES
    })
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown event type(s): {', '.join(unknown)}")
    payload = [event.model_dump() for event in request.events]
    inserted = await _run_sync(app_store.record_user_events, payload)
    return {"accepted": inserted, "duplicates": len(payload) - inserted}


@app.get("/users/{user_id}/profile")
async def get_user_profile(user_id: str, limit: int = 20):
    if app_store is None:
        raise HTTPException(status_code=503, detail="Application storage is not initialized")
    return await _run_sync(app_store.get_user_profile, user_id, limit)


@app.delete("/users/{user_id}/profile")
async def clear_user_profile(user_id: str):
    if app_store is None:
        raise HTTPException(status_code=503, detail="Application storage is not initialized")
    await _run_sync(app_store.clear_user_profile, user_id)
    return {"status": "ok", "cleared_user_id": user_id}


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
        answer, routing = await _run_agent_with_source_routing(request, source_filter)
        return QueryResponse(
            answer=answer,
            sources=[],
            question=request.question,
            session_id=request.session_id,
            routing=routing,
        )

    effective_question = await _run_sync(
        _build_memory_aware_question,
        request.question,
        request.session_id,
        request.use_memory,
        request.user_id,
    )
    chunks, routing = await _retrieve_with_source_routing(
        request.question,
        explicit_sources=source_filter,
        session_id=request.session_id,
        user_id=request.user_id,
        current_document_id=request.current_document_id,
        current_source_name=request.current_source_name,
    )
    answer = await _run_sync(rag_pipeline.generator.generate, effective_question, chunks)
    response = RAGResponse(answer=answer, retrieved_chunks=chunks, query=request.question)
    _remember_turn(
        request.session_id,
        request.question,
        response.answer,
        request.use_memory,
        response.retrieved_chunks,
    )
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
        routing=routing,
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
    chunks, routing = await _retrieve_with_source_routing(
        request.question,
        top_k=request.top_k,
        explicit_sources=source_filter,
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
            "source_routing": routing,
        },
        "latency_ms": latency_ms,
    }


@app.post("/search")
async def search_papers(request: SearchRequest):
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized")

    await _reload_index_if_changed()
    source_filter = _resolve_source_filter(request.query, request.source_names)
    chunks, routing = await _retrieve_with_source_routing(
        request.query,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        explicit_sources=source_filter,
        session_id=request.session_id,
        user_id=request.user_id,
        current_document_id=request.current_document_id,
        current_source_name=request.current_source_name,
    )
    return {
        "query": request.query,
        "retrieved_chunks": [_format_retrieved_chunk(chunk) for chunk in chunks],
        "routing": routing,
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
        answer, routing = await _run_agent_with_source_routing(request, source_filter)

        def generate_agent():
            yield f"data: {json.dumps({'type': 'routing', 'data': routing}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'data': []}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'data': answer}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

        return StreamingResponse(generate_agent(), media_type="text/event-stream")

    effective_question = await _run_sync(
        _build_memory_aware_question,
        request.question,
        request.session_id,
        request.use_memory,
        request.user_id,
    )
    chunks, routing = await _retrieve_with_source_routing(
        request.question,
        explicit_sources=source_filter,
        session_id=request.session_id,
        user_id=request.user_id,
        current_document_id=request.current_document_id,
        current_source_name=request.current_source_name,
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
        yield f"data: {json.dumps({'type': 'routing', 'data': routing}, ensure_ascii=False)}\n\n"
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
            chunks,
        )
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/index/text")
async def index_text(text: str, source_name: str = "manual"):
    chunks_added = await _run_with_faiss_lock(rag_pipeline.index_text, text, source_name)
    return {"chunks_added": chunks_added}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8011, reload=True)
