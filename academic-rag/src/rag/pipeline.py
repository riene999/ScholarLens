"""RAG pipeline: retrieve -> optional rerank -> generate."""
from dataclasses import dataclass
from typing import Dict, List

from loguru import logger

from src.rag.bm25_retriever import BM25Retriever
from src.rag.generator import LLMGenerator
from src.rag.retriever import FAISSRetriever, RetrievedChunk
from src.utils.config import AppConfig


@dataclass
class RAGResponse:
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    query: str


class RAGPipeline:
    def __init__(self, config: AppConfig):
        from src.rag.embedder import Embedder
        from src.rag.reranker import CrossEncoderReranker

        self.config = config
        self.embedder = Embedder(
            model_name=config.embedding.model,
            device=config.embedding.device,
            batch_size=config.embedding.batch_size,
            query_cache_enabled=config.embedding.query_cache_enabled,
            query_cache_max_size=config.embedding.query_cache_max_size,
            query_cache_ttl_seconds=config.embedding.query_cache_ttl_seconds,
            redis_config=config.redis,
        )
        self.retriever = FAISSRetriever(
            embedder=self.embedder,
            dimension=config.vector_store.dimension,
            index_path=config.vector_store.index_path,
            result_cache_enabled=config.retrieval.result_cache_enabled,
            result_cache_max_size=config.retrieval.result_cache_max_size,
            result_cache_ttl_seconds=config.retrieval.result_cache_ttl_seconds,
            redis_config=config.redis,
        )
        self.generator = LLMGenerator(config.llm)
        self.bm25_retriever = (
            BM25Retriever(k1=config.bm25.k1, b=config.bm25.b)
            if config.bm25.enabled
            else None
        )

        self.reranker = None
        if config.reranker.enabled:
            self.reranker = CrossEncoderReranker(
                model_name=config.reranker.model,
                device=config.reranker.device,
                batch_size=config.reranker.batch_size,
            )

        self.retriever.load()
        if self.bm25_retriever is not None:
            self.bm25_retriever.load_documents(self.retriever.documents)

    def retrieve_chunks(
        self,
        question: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        use_reranker: bool = True,
        source_filter: List[str] | None = None,
    ) -> List[RetrievedChunk]:
        chunks = self._retrieve_chunks_with_optional_decomposition(
            question=question,
            top_k=top_k,
            score_threshold=score_threshold,
            use_reranker=use_reranker,
            source_filter=source_filter,
        )
        self._restore_raw_content(chunks)
        return chunks

    def _restore_raw_content(self, chunks: List[RetrievedChunk]) -> None:
        """For summary-indexed special chunks, swap content back to the original
        raw table/formula markdown so the LLM receives the full structured data."""
        store = self.retriever.document_store
        for chunk in chunks:
            raw_chunk_id = chunk.document.metadata.get("raw_chunk_id")
            if not raw_chunk_id:
                continue
            raw = store.get_raw_chunk(raw_chunk_id)
            if raw:
                caption = raw.get("caption", "")
                section = raw.get("section_title", "")
                header = ""
                if section:
                    header += f"[Section: {section}]\n"
                if caption:
                    header += f"[Caption: {caption}]\n"
                chunk.document.content = header + raw["raw_content"]

    def _retrieve_chunks_single(
        self,
        question: str,
        top_k: int,
        score_threshold: float,
        use_reranker: bool,
        source_filter: List[str] | None = None,
    ) -> List[RetrievedChunk]:
        retrieve_k = top_k
        if self.reranker and use_reranker:
            retrieve_k = max(top_k, self.config.reranker.candidate_top_k)

        dense_chunks = self.retriever.retrieve(
            query=question,
            top_k=retrieve_k,
            score_threshold=score_threshold,
            source_filter=source_filter,
        )
        chunks = self._hybrid_retrieve(
            question=question,
            dense_chunks=dense_chunks,
            retrieve_k=retrieve_k,
            source_filter=source_filter,
        )
        if not (self.reranker and use_reranker):
            return chunks[:top_k]

        reranked = self.reranker.rerank(question, chunks, top_k=top_k)
        logger.info(
            "Reranker enabled: reranked {} candidates to top {}",
            len(chunks),
            len(reranked),
        )
        return reranked

    def _hybrid_retrieve(
        self,
        question: str,
        dense_chunks: List[RetrievedChunk],
        retrieve_k: int,
        source_filter: List[str] | None = None,
    ) -> List[RetrievedChunk]:
        if self.bm25_retriever is None:
            return dense_chunks

        sparse_chunks = self.bm25_retriever.retrieve(
            query=question,
            top_k=max(retrieve_k, self.config.bm25.top_k),
        )
        if source_filter:
            source_set = set(source_filter)
            sparse_chunks = [
                chunk
                for chunk in sparse_chunks
                if (chunk.document.metadata or {}).get("source") in source_set
            ]
        if not sparse_chunks:
            return dense_chunks

        fused = self._rrf_fuse(
            [dense_chunks, sparse_chunks],
            rrf_k=self.config.bm25.rrf_k,
        )
        logger.info(
            "Hybrid retrieval: dense={}, bm25={}, fused={}",
            len(dense_chunks),
            len(sparse_chunks),
            len(fused),
        )
        return fused[: max(retrieve_k, self.config.bm25.top_k)]

    def _rrf_fuse(
        self,
        ranked_lists: List[List[RetrievedChunk]],
        rrf_k: int,
    ) -> List[RetrievedChunk]:
        scores: Dict[str, float] = {}
        chunks_by_id: Dict[str, RetrievedChunk] = {}

        for ranked_list in ranked_lists:
            for rank, chunk in enumerate(ranked_list, start=1):
                key = chunk.document.chunk_id
                chunks_by_id.setdefault(key, chunk)
                scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

        ranked_ids = sorted(scores, key=lambda key: scores[key], reverse=True)
        fused: List[RetrievedChunk] = []
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            chunk = chunks_by_id[chunk_id]
            fused.append(
                RetrievedChunk(
                    document=chunk.document,
                    score=float(scores[chunk_id]),
                    rank=rank,
                )
            )
        return fused

    def _merge_chunks(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        merged: Dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            key = chunk.document.chunk_id
            current = merged.get(key)
            if current is None or chunk.score > current.score:
                merged[key] = chunk

        ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        for idx, chunk in enumerate(ranked, start=1):
            chunk.rank = idx
        return ranked

    def _retrieve_chunks_with_optional_decomposition(
        self,
        question: str,
        top_k: int | None,
        score_threshold: float | None,
        use_reranker: bool,
        source_filter: List[str] | None = None,
    ) -> List[RetrievedChunk]:
        final_top_k = top_k if top_k is not None else self.config.retrieval.top_k
        threshold = (
            score_threshold
            if score_threshold is not None
            else self.config.retrieval.score_threshold
        )
        decompose_enabled = self.config.retrieval.query_decomposition_enabled

        if not decompose_enabled:
            return self._retrieve_chunks_single(
                question=question,
                top_k=final_top_k,
                score_threshold=threshold,
                use_reranker=use_reranker,
                source_filter=source_filter,
            )

        sub_questions = self.generator.decompose_query(
            question,
            max_subquestions=self.config.retrieval.decomposition_max_subquestions,
        )
        if len(sub_questions) <= 1:
            return self._retrieve_chunks_single(
                question=question,
                top_k=final_top_k,
                score_threshold=threshold,
                use_reranker=use_reranker,
                source_filter=source_filter,
            )

        logger.info("Complex query detected, sub-questions={}", len(sub_questions))
        all_chunks: List[RetrievedChunk] = []
        for sub_q in sub_questions:
            sub_chunks = self._retrieve_chunks_single(
                question=sub_q,
                top_k=final_top_k,
                score_threshold=threshold,
                use_reranker=False,  # Rerank once on merged set by original question.
                source_filter=source_filter,
            )
            all_chunks.extend(sub_chunks)

        merged = self._merge_chunks(all_chunks)
        if self.reranker and use_reranker:
            candidate_count = max(final_top_k, self.config.reranker.candidate_top_k)
            candidates = merged[:candidate_count]
            reranked = self.reranker.rerank(question, candidates, top_k=final_top_k)
            logger.info(
                "Decomposition retrieval merged {} chunks, reranked to {}",
                len(merged),
                len(reranked),
            )
            return reranked
        return merged[:final_top_k]

    def query(
        self,
        question: str,
        top_k: int | None = None,
        source_filter: List[str] | None = None,
    ) -> RAGResponse:
        logger.info("Received query: {}...", question[:50])
        chunks = self.retrieve_chunks(question, top_k=top_k, source_filter=source_filter)
        answer = self.generator.generate(question, chunks)
        return RAGResponse(answer=answer, retrieved_chunks=chunks, query=question)

    def query_stream(self, question: str, source_filter: List[str] | None = None):
        chunks = self.retrieve_chunks(question, source_filter=source_filter)
        return chunks, self.generator.generate_stream(question, chunks)

    def index_documents_from_pdf(self, pdf_path: str, source_name: str | None = None) -> int:
        from src.utils.pdf_parser import PDFParser

        parser = PDFParser(
            chunk_size=self.config.pdf_parser.chunk_size,
            chunk_overlap=self.config.pdf_parser.chunk_overlap,
            device=self.config.embedding.device,
            engine=self.config.pdf_parser.engine,
        )
        documents, special_blocks = parser.parse_pdf(pdf_path, source_name=source_name)

        # For each special block: generate LLM summary → use as the indexed content,
        # persist the raw table/formula content in raw_chunks for retrieval-time fetch.
        enabled_special_blocks = [
            block
            for block in special_blocks
            if (
                block.chunk_type == "table"
                and self.config.pdf_parser.summarize_tables
            )
            or (
                block.chunk_type == "formula"
                and self.config.pdf_parser.summarize_formulas
            )
        ]
        skipped_special_blocks = len(special_blocks) - len(enabled_special_blocks)
        if skipped_special_blocks:
            logger.info(
                "Skipping {} special block summaries by parser config",
                skipped_special_blocks,
            )

        if enabled_special_blocks:
            store = self.retriever.document_store
            from pathlib import Path
            stem = Path(pdf_path).stem
            store.delete_raw_chunks_by_prefix(stem)

            for block in enabled_special_blocks:
                try:
                    summary = self.generator.summarize_special_chunk(block)
                except Exception as exc:
                    logger.warning("Summary generation failed for {}: {}", block.raw_chunk_id, exc)
                    summary = block.table_markdown  # fallback: index raw content

                store.save_raw_chunk(
                    raw_chunk_id=block.raw_chunk_id,
                    raw_content=block.table_markdown,
                    chunk_type=block.chunk_type,
                    section_title=block.section_title,
                    caption=block.caption,
                )
                # Replace the document's content with the summary for embedding
                for doc in documents:
                    if doc.chunk_id == block.raw_chunk_id:
                        doc.content = summary
                        break

        self.retriever.add_documents(documents, replace_sources=True)
        if self.bm25_retriever is not None:
            self.bm25_retriever.add_documents(documents)
        self.retriever.save()
        return len(documents)

    def index_text(self, text: str, source_name: str = "manual") -> int:
        from src.utils.pdf_parser import PDFParser

        parser = PDFParser(
            chunk_size=self.config.pdf_parser.chunk_size,
            chunk_overlap=self.config.pdf_parser.chunk_overlap,
            engine=self.config.pdf_parser.engine,
        )
        documents = parser.parse_text(text, source_name)
        self.retriever.add_documents(documents, replace_sources=True)
        if self.bm25_retriever is not None:
            self.bm25_retriever.add_documents(documents)
        self.retriever.save()
        return len(documents)
