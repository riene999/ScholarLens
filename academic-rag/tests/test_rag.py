"""
单元测试
面试可讲：测试覆盖核心模块，保证代码质量
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.utils.pdf_parser import PDFParser, Document
from src.rag.retriever import FAISSRetriever
from src.agent.agent import ConversationMemory


class TestPDFParser:
    def setup_method(self):
        self.parser = PDFParser(chunk_size=100, chunk_overlap=20)

    def test_split_short_text(self):
        text = "这是一段短文本。"
        chunks = self.parser._split_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_long_text(self):
        text = "这是第一段。" * 50
        chunks = self.parser._split_text(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= self.parser.chunk_size + self.parser.chunk_overlap + 10

    def test_parse_text(self):
        text = "联邦学习是一种分布式机器学习框架。" * 20
        docs = self.parser.parse_text(text, "test")
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)
        assert all(d.metadata["source"] == "test" for d in docs)

    def test_formula_segments_are_packed_into_text_sized_chunks(self):
        parser = PDFParser(chunk_size=220, chunk_overlap=0)
        text = "\n\n".join(
            f"Paragraph {idx} explains the update rule and its convergence role.\n\n"
            f"$$x_{idx} = y_{idx} + z_{idx}$$"
            for idx in range(12)
        )

        segments = parser._split_into_segments(text)
        records = parser._merge_segments(segments)

        assert sum(1 for _, chunk_type in segments if chunk_type == "formula") == 12
        assert len(records) < len(segments)
        assert all(len(content) <= parser.chunk_size for content, _ in records)
        assert all(chunk_type == "formula" for _, chunk_type in records)


class TestFAISSRetriever:
    def setup_method(self):
        mock_embedder = MagicMock()
        mock_embedder.dimension = 64
        # 模拟embed返回固定维度向量
        mock_embedder.embed_documents.return_value = np.random.randn(3, 64).astype(np.float32)
        mock_embedder.embed_query.return_value = np.random.randn(64).astype(np.float32)

        self.retriever = FAISSRetriever(
            embedder=mock_embedder,
            dimension=64,
            index_path="/tmp/test_faiss"
        )

    def test_add_and_retrieve(self):
        docs = [
            Document(content=f"文档{i}", metadata={"source": "test"}, chunk_id=f"chunk_{i}")
            for i in range(3)
        ]
        self.retriever.add_documents(docs)
        assert self.retriever.index.ntotal == 3

    def test_retrieve_empty_index(self):
        results = self.retriever.retrieve("查询", top_k=3)
        assert results == []


class TestConversationMemory:
    def test_add_turn_keeps_recent_messages(self):
        memory = ConversationMemory(max_turns=2)
        memory.add_turn("s1", "q1", "a1")
        memory.add_turn("s1", "q2", "a2")
        memory.add_turn("s1", "q3", "a3")

        messages = memory.get_messages("s1")
        assert [m["content"] for m in messages] == ["q2", "a2", "q3", "a3"]

    def test_clear_session(self):
        memory = ConversationMemory()
        memory.add_turn("s1", "q1", "a1")
        memory.add_turn("s2", "q2", "a2")

        memory.clear("s1")

        assert memory.get_messages("s1") == []
        assert memory.get_messages("s2") != []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
