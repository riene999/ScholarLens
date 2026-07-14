"""RQ jobs for PDF indexing."""

from pathlib import Path
from threading import Lock
from typing import Any

from loguru import logger
from redis import Redis
from rq import Queue

from src.rag.pipeline import RAGPipeline
from src.shared.context import create_pipeline
from src.utils.config import RedisConfig, load_config


QUEUE_NAME = "pdf-indexing"
DEFAULT_CONFIG_PATH = "config.yaml"

# 进程级单例，由 init_worker_pipeline() 在 Worker 启动时初始化一次
_worker_pipeline: RAGPipeline | None = None
_index_write_lock = Lock()  # 保护 FAISS 并发写入


def init_worker_pipeline(config_path: str = DEFAULT_CONFIG_PATH) -> None:
    global _worker_pipeline
    if _worker_pipeline is None:
        logger.info("Worker: loading RAG pipeline (one-time)...")
        _worker_pipeline = create_pipeline(config_path)
        logger.info("Worker: pipeline ready")


def make_redis_connection(redis_config: RedisConfig) -> Redis:
    return Redis(
        host=redis_config.host,
        port=redis_config.port,
        db=redis_config.db,
        password=redis_config.password,
        socket_timeout=redis_config.socket_timeout,
        socket_connect_timeout=redis_config.socket_connect_timeout,
        decode_responses=False,
    )


def get_redis_connection(config_path: str = DEFAULT_CONFIG_PATH) -> Redis:
    config = load_config(config_path)
    return make_redis_connection(config.redis)


def get_index_queue(redis_connection: Redis) -> Queue:
    return Queue(QUEUE_NAME, connection=redis_connection)


def enqueue_pdf_index_job(
    queue: Queue,
    *,
    job_id: str,
    pdf_path: str,
    source_name: str,
    config_path: str = DEFAULT_CONFIG_PATH,
    delete_after: bool = True,
) -> Any:
    return queue.enqueue(
        index_pdf_job,
        pdf_path,
        source_name,
        config_path,
        delete_after,
        job_id=job_id,
        meta={"filename": source_name},
        job_timeout="30m",
        result_ttl=24 * 60 * 60,
        failure_ttl=24 * 60 * 60,
    )


def index_pdf_job(
    pdf_path: str,
    source_name: str,
    config_path: str = DEFAULT_CONFIG_PATH,
    delete_after: bool = True,
) -> dict:
    path = Path(pdf_path)
    logger.info("Starting PDF index job: {}", source_name)
    try:
        # 优先使用进程级单例，回退到每次新建（兼容不经 init_worker_pipeline 的调用方）
        pipeline = _worker_pipeline or create_pipeline(config_path)
        with _index_write_lock:
            chunks_added = pipeline.index_documents_from_pdf(str(path), source_name=source_name)
            version_file = Path(pipeline.config.vector_store.index_path) / ".index_version"
            version_file.touch()
        logger.info("Finished PDF index job: {}, chunks={}", source_name, chunks_added)
        return {
            "filename": source_name,
            "chunks_added": chunks_added,
            "index_path": pipeline.config.vector_store.index_path,
        }
    finally:
        if delete_after:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Failed to remove uploaded PDF {}: {}", path, exc)
