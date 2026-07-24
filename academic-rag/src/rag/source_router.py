"""Asynchronous LLM router for speculative document-scope resolution."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from threading import Lock

from loguru import logger
from openai import AsyncOpenAI

from src.utils.config import LLMConfig, SourceRoutingConfig
from src.utils.pdf_parser import normalize_title


ROUTER_SYSTEM_PROMPT = """You resolve user-imposed paper source constraints.
You are NOT a paper recommender and must not choose papers merely because they are
relevant to, or capable of answering, the question. The default is `global`.

Return one JSON object only. Documents must come from DOCUMENT_CATALOG.

Use `scoped` only when the source constraint is grounded by at least one of:
1. The current question explicitly names a catalog paper. A short method alias counts
   only when it is paired with a source cue such as "paper", "论文", or "according to".
2. A pronoun or deictic phrase (it, this paper, that method, 它, 这篇论文, 该方法,
   前者/后者, 第一篇/第二篇) has a unique antecedent in RECENT_TURNS.
3. The user explicitly says "this paper/study" while CURRENT_DOCUMENT is present.
4. A comparison explicitly refers to uniquely identifiable papers from recent turns.
5. The question is an elliptical immediate follow-up to a turn with exactly one
   retrieved paper, and is not a general definition, survey, or broad method question.

Use `global` for definitions, general concepts, open research questions, surveys, and
broad method questions even when the current document, conversation, profile, or
catalog contains topically relevant papers. Current-document presence, topic
similarity, retrieval history, and user preferences alone never create a constraint.
Mentioning a method such as FedAvg or SPFL does not by itself mean that the user wants
evidence restricted to its original paper; other papers may analyze that method.
An explicit request to search "all papers", "the full paper collection", "globally",
or "across the literature" always means `global`, even if a named method has an
originating paper in the catalog.

Use `ambiguous` when source-referential language exists but its antecedent is missing
or not unique. Do not guess. Conversation summaries and user profiles may help with
meaning but can never by themselves justify a hard paper constraint.

Examples:
- Recent turn used only paper A; "What does it propose?" -> scoped to A.
- Recent turn used paper A; "What is federated learning?" -> global.
- Recent turn used papers A and B; "What does it prove?" -> ambiguous.
- Paper A is open; "What are common non-IID methods?" -> global.
- Paper A is open; "What datasets does this paper use?" -> scoped to A.
- "What convergence rate is known for FedAvg?" -> global.
- "According to the original FedAvg paper, how is communication reduced?" -> scoped.

Schema:
{
  "decision": "scoped|global|ambiguous",
  "intent": "single_paper|comparison|open_research|follow_up",
  "documents": [{"document_id": 1, "title": "canonical title"}],
  "constraint_strength": "hard|none",
  "binding_basis": "explicit_name|recent_unique_antecedent|recent_elliptical_follow_up|current_document_reference|comparison_reference|none",
  "anchor_text": "the words in the current question that impose the constraint",
  "antecedent_turn": -1,
  "confidence": 0.0,
  "allow_global_evidence": false,
  "reason": "short explanation"
}

For `global` and `ambiguous`, return no documents, `constraint_strength=none`, and
`binding_basis=none`. For a scoped comparison, include every constrained paper.
Set allow_global_evidence=true only when the user explicitly asks for outside papers
or general background. Never invent titles or IDs.
"""


@dataclass(frozen=True)
class RoutedDocument:
    document_id: int
    source_name: str
    title: str


@dataclass
class SourcePlan:
    decision: str = "global"
    intent: str = "open_research"
    documents: list[RoutedDocument] = field(default_factory=list)
    confidence: float = 0.0
    allow_global_evidence: bool = True
    constraint_strength: str = "none"
    binding_basis: str = "none"
    anchor_text: str = ""
    antecedent_turn: int | None = None
    validated_hard_scope: bool = False
    reason: str = "router_not_used"
    latency_ms: int = 0

    @property
    def source_names(self) -> list[str]:
        return [document.source_name for document in self.documents]

    @property
    def document_ids(self) -> list[int]:
        return [document.document_id for document in self.documents]

    @property
    def has_validated_scope(self) -> bool:
        return (
            self.decision == "scoped"
            and self.constraint_strength == "hard"
            and self.validated_hard_scope
            and bool(self.documents)
        )

    def to_trace(self) -> dict:
        return {
            "decision": self.decision,
            "intent": self.intent,
            "document_ids": self.document_ids,
            "source_names": self.source_names,
            "confidence": round(self.confidence, 4),
            "allow_global_evidence": self.allow_global_evidence,
            "constraint_strength": self.constraint_strength,
            "binding_basis": self.binding_basis,
            "anchor_text": self.anchor_text,
            "antecedent_turn": self.antecedent_turn,
            "validated_hard_scope": self.validated_hard_scope,
            "reason": self.reason,
            "latency_ms": self.latency_ms,
        }


class SourcePlanHandle:
    """Thread-safe, non-blocking handoff from an async router to the sync Agent."""

    def __init__(self):
        self._lock = Lock()
        self._plan: SourcePlan | None = None

    def set(self, plan: SourcePlan) -> None:
        with self._lock:
            self._plan = plan

    def get_if_ready(self) -> SourcePlan | None:
        with self._lock:
            return self._plan


class SourceRouter:
    def __init__(self, llm: LLMConfig, config: SourceRoutingConfig):
        self.config = config
        self.client = AsyncOpenAI(api_key=llm.api_key, base_url=llm.base_url)

    async def resolve(
        self,
        question: str,
        documents: list[dict],
        *,
        conversation_summary: str = "",
        recent_messages: list[dict[str, str]] | None = None,
        recent_turns: list[dict] | None = None,
        current_document_id: int | None = None,
        current_source_name: str | None = None,
        user_profile: str = "",
    ) -> SourcePlan:
        started = time.perf_counter()
        catalog = self._prepare_catalog(documents)
        if not catalog:
            return SourcePlan(reason="empty_document_catalog")

        prepared_turns = self._prepare_recent_turns(recent_turns or [], catalog)
        payload = {
            "question": question,
            "conversation_summary": conversation_summary[:6000],
            "recent_messages": (recent_messages or [])[-6:],
            "RECENT_TURNS": prepared_turns,
            "current_document": {
                "document_id": current_document_id,
                "source_name": current_source_name,
            },
            "user_profile": user_profile[:4000],
            "DOCUMENT_CATALOG": catalog,
        }
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.0,
            max_tokens=280,
        )
        raw = (response.choices[0].message.content or "").strip()
        result = self._parse_and_validate(raw, catalog)
        result.latency_ms = int((time.perf_counter() - started) * 1000)
        logger.info("Source router plan: {}", result.to_trace())
        return result

    def _prepare_recent_turns(
        self,
        recent_turns: list[dict],
        catalog: list[dict],
    ) -> list[dict]:
        by_source = {str(item["source_name"]): item for item in catalog}
        prepared = []
        total = len(recent_turns[-3:])
        for position, turn in enumerate(recent_turns[-3:]):
            documents = []
            for source in turn.get("retrieved_sources") or []:
                item = by_source.get(str(source))
                if item is None or any(
                    document["document_id"] == int(item["document_id"])
                    for document in documents
                ):
                    continue
                documents.append({
                    "document_id": int(item["document_id"]),
                    "source_name": str(item["source_name"]),
                    "title": str(item["title"]),
                })
            try:
                turn_index = int(turn.get("turn_index", position - total))
            except (TypeError, ValueError):
                turn_index = position - total
            prepared.append({
                "turn_index": turn_index,
                "user": str(turn.get("user") or turn.get("user_content") or "")[:2000],
                "assistant": str(
                    turn.get("assistant") or turn.get("assistant_content") or ""
                )[:2000],
                "retrieved_documents": documents,
                "artifact_digests": list(turn.get("artifact_digests") or [])[:4],
            })
        return prepared

    def _prepare_catalog(
        self,
        documents: list[dict],
    ) -> list[dict]:
        catalog = []
        for document in documents:
            document_id = document.get("id")
            try:
                document_id = int(document_id)
            except (TypeError, ValueError):
                continue
            source_name = str(document.get("source_name") or "")
            title = str(document.get("paper_title") or source_name)
            title_hint = str(document.get("title_hint") or "")
            aliases = [str(item) for item in document.get("aliases") or []]
            catalog.append({
                "document_id": document_id,
                "source_name": source_name,
                "title": title,
                "title_hint": title_hint,
                "aliases": aliases,
            })
        return catalog[: self.config.catalog_max_documents]

    def _parse_and_validate(self, raw: str, catalog: list[dict]) -> SourcePlan:
        cleaned = raw
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            cleaned = cleaned.rsplit("```", 1)[0].strip()
        try:
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse source router JSON: {}", raw[:300])
            return SourcePlan(decision="ambiguous", reason="invalid_router_json")
        if not isinstance(parsed, dict):
            return SourcePlan(decision="ambiguous", reason="invalid_router_shape")

        by_id = {int(item["document_id"]): item for item in catalog}
        by_title = {
            normalize_title(str(item["title"])): item
            for item in catalog
            if normalize_title(str(item["title"]))
        }
        for item in catalog:
            normalized_hint = normalize_title(str(item.get("title_hint") or ""))
            if normalized_hint:
                by_title.setdefault(normalized_hint, item)
            for alias in item.get("aliases") or []:
                normalized_alias = normalize_title(str(alias))
                if normalized_alias:
                    by_title.setdefault(normalized_alias, item)
        document_items = parsed.get("documents", [])
        if not isinstance(document_items, list):
            document_items = []
        routed: list[RoutedDocument] = []
        for item in document_items[: self.config.max_documents]:
            if not isinstance(item, dict):
                continue
            candidate = None
            try:
                candidate = by_id.get(int(item.get("document_id")))
            except (TypeError, ValueError):
                pass
            if candidate is None:
                candidate = by_title.get(normalize_title(str(item.get("title") or "")))
            if candidate is None or any(
                existing.document_id == int(candidate["document_id"])
                for existing in routed
            ):
                continue
            routed.append(
                RoutedDocument(
                    document_id=int(candidate["document_id"]),
                    source_name=str(candidate["source_name"]),
                    title=str(candidate["title"]),
                )
            )

        decision = str(parsed.get("decision") or "global").lower()
        if decision not in {"scoped", "global", "ambiguous"}:
            decision = "ambiguous"
        if decision == "scoped" and not routed:
            decision = "ambiguous"
        intent = str(parsed.get("intent") or "open_research").lower()
        if intent not in {"single_paper", "comparison", "open_research", "follow_up"}:
            intent = "open_research"
        try:
            confidence = min(1.0, max(0.0, float(parsed.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        allow_global = parsed.get("allow_global_evidence", True)
        if not isinstance(allow_global, bool):
            allow_global = True
        constraint_strength = str(parsed.get("constraint_strength") or "none").lower()
        if constraint_strength not in {"hard", "none"}:
            constraint_strength = "none"
        binding_basis = str(parsed.get("binding_basis") or "none").lower()
        if binding_basis not in {
            "explicit_name",
            "recent_unique_antecedent",
            "recent_elliptical_follow_up",
            "current_document_reference",
            "comparison_reference",
            "none",
        }:
            binding_basis = "none"
        antecedent_turn = parsed.get("antecedent_turn")
        try:
            antecedent_turn = int(antecedent_turn) if antecedent_turn is not None else None
        except (TypeError, ValueError):
            antecedent_turn = None
        return SourcePlan(
            decision=decision,
            intent=intent,
            documents=routed,
            confidence=confidence,
            allow_global_evidence=allow_global,
            constraint_strength=constraint_strength,
            binding_basis=binding_basis,
            anchor_text=str(parsed.get("anchor_text") or "")[:200],
            antecedent_turn=antecedent_turn,
            # Routing intent is decided entirely by the LLM. Local code only
            # validates the returned shape and catalog document identifiers.
            validated_hard_scope=(
                decision == "scoped"
                and constraint_strength == "hard"
                and bool(routed)
            ),
            reason=str(parsed.get("reason") or "")[:500],
        )
