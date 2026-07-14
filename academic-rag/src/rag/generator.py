import json
from typing import Iterator, List, Tuple

from loguru import logger
from openai import OpenAI

from src.rag.retriever import RetrievedChunk
from src.utils.config import LLMConfig


SYSTEM_PROMPT = (
    "You are an academic QA assistant. "
    "Answer only with provided evidence. "
    "If evidence is insufficient, say so explicitly. "
    "Cite source/page when making key claims."
)

CONSISTENCY_CHECK_PROMPT = """You are a factual consistency checker.
Decide whether the answer is supported by the evidence.
Rules:
1. If answer contains unsupported or contradictory claims, return false.
2. If main claims are supported, return true.
3. Output strict JSON only:
{"is_consistent": true/false, "reason": "..."}
No extra text."""

QUERY_DECOMPOSITION_PROMPT = """You are a query planner.
Decompose a complex question into at most {max_subquestions} retrievable sub-questions.
Rules:
1. Cover major aspects of the original question.
2. Keep each sub-question specific for semantic retrieval.
3. For simple questions, return a single-item list containing the original question.
Output strict JSON array only, e.g. ["sub-question 1", "sub-question 2"]."""

TABLE_SUMMARY_PROMPT = """You are an academic research assistant helping build a retrieval index.
You will be shown a table from a scholarly paper along with its surrounding context.
Write a concise English summary (≤200 words) suitable for semantic search indexing.

The summary MUST include:
1. What the table compares or presents (methods / datasets / metrics / conditions).
2. Key findings: highest/lowest values, notable differences, trends.
3. All technical terms, method names, dataset names (include both English and Chinese if present).
4. The section this table belongs to and how it relates to the paper's argument.

Do NOT copy every number verbatim. Focus on meaning and searchability.

[Section]: {section_title}
[Caption]: {caption}
[Preceding context]:
{prev_context}

[Table]:
{table_markdown}

[Following context]:
{next_context}

Summary:"""

FORMULA_SUMMARY_PROMPT = """You are an academic research assistant helping build a retrieval index.
You will be shown a mathematical formula or equation block from a scholarly paper along with its surrounding context.
Write a concise English summary (≤150 words) suitable for semantic search indexing.

The summary MUST include:
1. What this formula defines or computes (loss function / objective / update rule / metric / theorem, etc.).
2. What each key symbol represents (in plain English).
3. The intuition or purpose behind the formula in the context of the paper.
4. All relevant technical terms, method names, and named quantities.

Do NOT re-state the LaTeX verbatim. Focus on meaning and searchability.

[Section]: {section_title}
[Preceding context]:
{prev_context}

[Formula]:
{table_markdown}

[Following context]:
{next_context}

Summary:"""


def _build_context(retrieved_chunks: List[RetrievedChunk]) -> str:
    if not retrieved_chunks:
        return "(No relevant documents retrieved.)"

    context_parts = []
    for chunk in retrieved_chunks:
        doc = chunk.document
        source_info = (
            f"[source: {doc.metadata.get('source', 'unknown')}, "
            f"page: {doc.metadata.get('page', '?')}, score: {chunk.score:.3f}]"
        )
        context_parts.append(f"{source_info}\n{doc.content}")
    return "\n\n---\n\n".join(context_parts)


def build_rag_prompt(
    query: str,
    retrieved_chunks: List[RetrievedChunk],
    extra_instruction: str = "",
) -> str:
    # 整合检索到的chunk，连带着问题和额外指令，生成一段prompt
    context = _build_context(retrieved_chunks)
    guidance = f"\n\nAdditional instruction:\n{extra_instruction}\n" if extra_instruction else ""
    return (
        "Retrieved evidence:\n\n"
        f"{context}\n\n"
        "---\n\n"
        f"Question: {query}{guidance}"
    )


class LLMGenerator:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def _chat_once(self, prompt: str, temperature: float = None) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=self.config.max_tokens,
        )
        answer = response.choices[0].message.content or ""
        logger.info("Generated answer, total_tokens={}", response.usage.total_tokens)
        return answer

    def is_complex_query(self, query: str) -> bool:
        # 判断是否拆解为子问题
        q = query.strip()
        if len(q) >= 40:
            return True

        markers = [
            "compare",
            "versus",
            "tradeoff",
            "pros and cons",
            "difference",
            "differences",
            "以及",
            "并且",
            "对比",
            "比较",
            "优缺点",
            "分别",
        ]
        lowered = q.lower()
        return any(m in lowered or m in q for m in markers)

    def decompose_query(self, query: str, max_subquestions: int = 3) -> List[str]:
        # 让LLM来拆解问题
        if not self.is_complex_query(query):
            return [query]

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {
                    "role": "system",
                    "content": QUERY_DECOMPOSITION_PROMPT.format(
                        max_subquestions=max_subquestions
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=300,
        )

        raw = (response.choices[0].message.content or "").strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                cleaned = [str(item).strip() for item in parsed if str(item).strip()]
                if cleaned:
                    logger.info("Decomposed query into {} sub-questions", len(cleaned))
                    return cleaned[: max(1, max_subquestions)]
        except json.JSONDecodeError:
            logger.warning("Failed to parse decomposition JSON, fallback to original query")

        return [query]

    def _check_consistency(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> Tuple[bool, str]:
        if not retrieved_chunks:
            return True, "no_evidence"

        payload = {
            "query": query,
            "answer": answer,
            "evidence": [
                {
                    "source": c.document.metadata.get("source", "unknown"),
                    "page": c.document.metadata.get("page", "?"),
                    "content": c.document.content,
                }
                for c in retrieved_chunks
            ],
        }

        review = self.client.chat.completions.create(
            model=self.config.citation_check_model,
            messages=[
                {"role": "system", "content": CONSISTENCY_CHECK_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.0,
            max_tokens=300,
        )

        raw = (review.choices[0].message.content or "").strip()
        try:
            result = json.loads(raw)
            return bool(result.get("is_consistent", False)), str(result.get("reason", ""))
        except json.JSONDecodeError:
            lowered = raw.lower()
            if '"is_consistent": true' in lowered:
                return True, "parsed_fallback_true"
            return False, f"checker_parse_failed: {raw[:180]}"

    def generate(self, query: str, retrieved_chunks: List[RetrievedChunk]) -> str:
        # 构建prompt->生成一次回答->启用一致性检查->不一致就在额外信息中告知->让LLM重答
        retries = max(0, int(self.config.citation_check_retries))
        check_enabled = bool(self.config.citation_check_enabled)

        extra_instruction = ""
        last_answer = ""
        last_reason = ""

        for attempt in range(retries + 1):
            prompt = build_rag_prompt(query, retrieved_chunks, extra_instruction=extra_instruction)
            answer = self._chat_once(prompt)
            last_answer = answer

            if not check_enabled:
                return answer

            is_consistent, reason = self._check_consistency(query, answer, retrieved_chunks)
            if is_consistent:
                if attempt > 0:
                    logger.info("Citation consistency check passed after retry #{}", attempt)
                return answer

            last_reason = reason
            logger.warning(
                "Citation consistency check failed at attempt {}/{}: {}",
                attempt + 1,
                retries + 1,
                reason,
            )
            extra_instruction = (
                "Your previous answer was not consistent with evidence. "
                f"Reason: {reason}. "
                "Rewrite strictly grounded in evidence and add [source,page] for key claims. "
                "If evidence is insufficient, clearly say so."
            )

        logger.warning("Consistency retries exhausted, returning last answer. last_reason={}", last_reason)
        return last_answer

    def summarize_special_chunk(self, block) -> str:
        """Generate an English indexing summary for a table/formula SpecialBlock."""
        template = FORMULA_SUMMARY_PROMPT if block.chunk_type == "formula" else TABLE_SUMMARY_PROMPT
        prompt = template.format(
            section_title=block.section_title or "(unknown section)",
            caption=block.caption or "(no caption)",
            prev_context=block.prev_context or "(none)",
            table_markdown=block.table_markdown,
            next_context=block.next_context or "(none)",
        )
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=400,
        )
        summary = (response.choices[0].message.content or "").strip()
        logger.info("Generated table summary ({} chars) for {}", len(summary), block.raw_chunk_id)
        return summary

    def generate_stream(self, query: str, retrieved_chunks: List[RetrievedChunk]) -> Iterator[str]:
        # 如果流式就没有一致性检查
        prompt = build_rag_prompt(query, retrieved_chunks)
        stream = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
