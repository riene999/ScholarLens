"""
配置加载模块
面试要点：统一配置管理，支持环境变量覆盖，方便部署
"""
import os
import yaml
from pathlib import Path
from dataclasses import dataclass


@dataclass
class LLMConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    citation_check_enabled: bool
    citation_check_retries: int
    citation_check_model: str


@dataclass
class EmbeddingConfig:
    model: str
    device: str
    batch_size: int
    query_cache_enabled: bool
    query_cache_max_size: int
    query_cache_ttl_seconds: int


@dataclass
class RetrievalConfig:
    top_k: int
    score_threshold: float
    chunk_size: int
    chunk_overlap: int
    query_decomposition_enabled: bool
    decomposition_max_subquestions: int
    result_cache_enabled: bool
    result_cache_max_size: int
    result_cache_ttl_seconds: int


@dataclass
class PDFParserConfig:
    engine: str
    chunk_size: int
    chunk_overlap: int
    summarize_tables: bool
    summarize_formulas: bool


@dataclass
class VectorStoreConfig:
    index_path: str
    dimension: int


@dataclass
class RerankerConfig:
    enabled: bool
    model: str
    device: str
    batch_size: int
    candidate_top_k: int


@dataclass
class BM25Config:
    enabled: bool
    top_k: int
    k1: float
    b: float
    rrf_k: int


@dataclass
class RedisConfig:
    host: str
    port: int
    db: int
    password: str | None
    socket_timeout: float
    socket_connect_timeout: float


@dataclass
class AppConfig:
    llm: LLMConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    pdf_parser: PDFParserConfig
    vector_store: VectorStoreConfig
    reranker: RerankerConfig
    bm25: BM25Config
    redis: RedisConfig


def load_config(config_path: str = "config.yaml") -> AppConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    api_key = os.environ.get("DS_API_KEY")
    if not api_key:
        raise RuntimeError("DS_API_KEY environment variable is required")

    embedding_cache = raw["embedding"].get(
        "cache",
        raw["embedding"].get("query_cache", {}),
    )
    retrieval_cache = raw["retrieval"].get(
        "cache",
        raw["retrieval"].get("result_cache", {}),
    )
    pdf_parser = raw.get("pdf_parser", {})
    parser_engine = str(pdf_parser.get("engine", "pypdf2")).lower()
    if parser_engine not in {"pypdf2", "marker"}:
        raise ValueError("pdf_parser.engine must be either 'pypdf2' or 'marker'")
    parser_engines = pdf_parser.get("engines", {})
    parser_settings = parser_engines.get(parser_engine, {})
    chunk_size = int(parser_settings.get("chunk_size", raw["retrieval"].get("chunk_size", 512)))
    chunk_overlap = int(
        parser_settings.get("chunk_overlap", raw["retrieval"].get("chunk_overlap", 64))
    )
    summarize_tables = bool(parser_settings.get("summarize_tables", True))
    summarize_formulas = bool(parser_settings.get("summarize_formulas", False))

    vector_store = raw["vector_store"]
    index_path = vector_store.get("index_path")
    if not index_path:
        index_root = Path(vector_store.get("index_root", "./data/faiss_indexes"))
        index_path = str(index_root / parser_engine)

    return AppConfig(
        llm=LLMConfig(
            provider=raw["llm"]["provider"],
            api_key=api_key,
            base_url=raw["llm"]["base_url"],
            model=raw["llm"]["model"],
            temperature=raw["llm"]["temperature"],
            max_tokens=raw["llm"]["max_tokens"],
            citation_check_enabled=raw["llm"].get("citation_check", {}).get("enabled", True),
            citation_check_retries=raw["llm"].get("citation_check", {}).get("retries", 2),
            citation_check_model=raw["llm"].get("citation_check", {}).get(
                "model",
                raw["llm"]["model"],
            ),
        ),
        embedding=EmbeddingConfig(
            model=raw["embedding"]["model"],
            device=raw["embedding"]["device"],
            batch_size=raw["embedding"]["batch_size"],
            query_cache_enabled=embedding_cache.get("enabled", True),
            query_cache_max_size=embedding_cache.get("max_size", 10000),
            query_cache_ttl_seconds=embedding_cache.get("ttl_seconds", 1800),
        ),
        retrieval=RetrievalConfig(
            top_k=raw["retrieval"]["top_k"],
            score_threshold=raw["retrieval"]["score_threshold"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            query_decomposition_enabled=raw["retrieval"].get("query_decomposition", {}).get(
                "enabled",
                True,
            ),
            decomposition_max_subquestions=raw["retrieval"]
            .get("query_decomposition", {})
            .get("max_subquestions", 3),
            result_cache_enabled=retrieval_cache.get("enabled", True),
            result_cache_max_size=retrieval_cache.get("max_size", 5000),
            result_cache_ttl_seconds=retrieval_cache.get("ttl_seconds", 900),
        ),
        pdf_parser=PDFParserConfig(
            engine=parser_engine,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            summarize_tables=summarize_tables,
            summarize_formulas=summarize_formulas,
        ),
        vector_store=VectorStoreConfig(
            index_path=index_path,
            dimension=vector_store["dimension"],
        ),
        reranker=RerankerConfig(
            enabled=raw.get("reranker", {}).get("enabled", False),
            model=raw.get("reranker", {}).get("model", "BAAI/bge-reranker-base"),
            device=raw.get("reranker", {}).get("device", "cpu"),
            batch_size=raw.get("reranker", {}).get("batch_size", 16),
            candidate_top_k=raw.get("reranker", {}).get("candidate_top_k", 20),
        ),
        bm25=BM25Config(
            enabled=raw.get("bm25", {}).get("enabled", True),
            top_k=int(raw.get("bm25", {}).get("top_k", 20)),
            k1=float(raw.get("bm25", {}).get("k1", 1.5)),
            b=float(raw.get("bm25", {}).get("b", 0.75)),
            rrf_k=int(raw.get("bm25", {}).get("rrf_k", 60)),
        ),
        redis=RedisConfig(
            host=raw.get("redis", {}).get("host", "localhost"),
            port=int(raw.get("redis", {}).get("port", 6379)),
            db=int(raw.get("redis", {}).get("db", 0)),
            password=raw.get("redis", {}).get("password"),
            socket_timeout=float(raw.get("redis", {}).get("socket_timeout", 1.0)),
            socket_connect_timeout=float(
                raw.get("redis", {}).get("socket_connect_timeout", 1.0)
            ),
        ),
    )
