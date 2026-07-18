"""Asynchronous LLM router for speculative document-scope resolution."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
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


_REFERENCE_CUE_RE = re.compile(
    r"(?:它|他|它们|他们|那(?:篇|个|种)?|这篇(?:论文|文章)|该(?:论文|文章|方法|工作)|"
    r"这个(?:方法|工作|模型)|上述|刚才|前者|后者|第一篇|第二篇|两篇|"
    r"\bit\b|\bthey\b|\bthis paper\b|\bthat paper\b|\bthis study\b|"
    r"\bthat study\b|\bthis method\b|\bthat method\b|\bthe former\b|"
    r"\bthe latter\b|\bfirst paper\b|\bsecond paper\b|\bprevious(?: paper| work)?\b|"
    r"\btwo papers\b|\bboth papers\b|\bthese papers\b)",
    re.IGNORECASE,
)
_CURRENT_DOCUMENT_CUE_RE = re.compile(
    r"(?:这篇(?:论文|文章)|当前(?:论文|文章)|本文|\bthis paper\b|"
    r"\bthis study\b|\bthe current paper\b)",
    re.IGNORECASE,
)
_OUTSIDE_EVIDENCE_RE = re.compile(
    r"(?:其他(?:论文|方法|工作)|除此之外|还有哪些|更广泛|\bother (?:papers|methods|work)\b|"
    r"\bbeyond (?:this|these)\b|\bmore broadly\b)",
    re.IGNORECASE,
)
_GENERAL_QUESTION_RE = re.compile(
    r"(?:什么是|基本定义|通常有哪些|有哪些(?:常见)?方法|常见(?:的)?(?:方法|优缺点|做法)|"
    r"一般(?:有什么|有哪些|如何)|如何评价|\bwhat is\b|\bwhat are common\b|"
    r"\bin general\b|\bhow is .{0,80} measured\b)",
    re.IGNORECASE,
)
_FORMER_REFERENCE_RE = re.compile(
    r"(?:前者|第一篇|\bthe former\b|\bfirst paper\b)", re.IGNORECASE
)
_LATTER_REFERENCE_RE = re.compile(
    r"(?:后者|第二篇|\bthe latter\b|\bsecond paper\b)", re.IGNORECASE
)
_ALL_RECENT_REFERENCE_RE = re.compile(
    r"(?:两篇|第一篇和第二篇|第一篇与第二篇|\btwo papers\b|\bboth papers\b|"
    r"\bthese papers\b)",
    re.IGNORECASE,
)


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
        catalog = self._prepare_catalog(
            question,
            documents,
            current_document_id=current_document_id,
            current_source_name=current_source_name,
        )
        if not catalog:
            return SourcePlan(reason="empty_document_catalog")

        prepared_turns = self._prepare_recent_turns(recent_turns or [], catalog)
        deterministic = self._resolve_deterministically(
            question,
            catalog,
            prepared_turns,
            recent_messages=recent_messages or [],
            current_document_id=current_document_id,
            current_source_name=current_source_name,
        )
        if deterministic is not None:
            deterministic.latency_ms = int((time.perf_counter() - started) * 1000)
            logger.info("Deterministic source constraint: {}", deterministic.to_trace())
            return deterministic

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
        self._validate_hard_scope(
            result,
            question,
            catalog,
            prepared_turns,
            current_document_id=current_document_id,
            current_source_name=current_source_name,
        )
        result.latency_ms = int((time.perf_counter() - started) * 1000)
        logger.info("Source router plan: {}", result.to_trace())
        return result

    @staticmethod
    def _catalog_document(item: dict) -> RoutedDocument:
        return RoutedDocument(
            document_id=int(item["document_id"]),
            source_name=str(item["source_name"]),
            title=str(item["title"]),
        )

    @staticmethod
    def _alias_in_question(alias: str, question: str) -> bool:
        alias = alias.strip()
        if len(alias) < 3:
            return False
        return re.search(
            rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])",
            question,
            re.IGNORECASE,
        ) is not None

    @classmethod
    def _alias_has_source_cue(cls, alias: str, question: str) -> bool:
        if not cls._alias_in_question(alias, question):
            return False
        escaped = re.escape(alias.strip())
        return re.search(
            rf"(?:"
            rf"(?:根据|按照|在)\s*{escaped}\s*(?:论文|文章|工作)(?:中)?|"
            rf"{escaped}\s*(?:论文|文章|工作|paper|study|work)|"
            rf"(?:论文|文章|paper|study|work)\s*(?:on|about|for|by)?\s*{escaped}|"
            rf"according\s+to\s+(?:the\s+)?(?:original\s+)?{escaped}"
            rf")",
            question,
            re.IGNORECASE,
        ) is not None

    def _matching_documents(self, question: str, catalog: list[dict]) -> list[dict]:
        normalized_question = normalize_title(question)
        matched = []
        for item in catalog:
            source_stem = Path(str(item.get("source_name") or "")).stem
            long_names = [
                str(item.get("title") or ""),
                str(item.get("title_hint") or ""),
                source_stem,
            ]
            has_long_match = any(
                len(normalized) >= 8 and normalized in normalized_question
                for normalized in (normalize_title(value) for value in long_names)
                if normalized
            )
            has_alias_match = any(
                self._alias_has_source_cue(str(alias), question)
                for alias in item.get("aliases") or []
            )
            if has_long_match or has_alias_match:
                matched.append(item)
        return matched

    @staticmethod
    def _current_catalog_document(
        catalog: list[dict],
        current_document_id: int | None,
        current_source_name: str | None,
    ) -> dict | None:
        for item in catalog:
            if current_document_id is not None and int(item["document_id"]) == int(
                current_document_id
            ):
                return item
            if current_source_name and item["source_name"] == current_source_name:
                return item
        return None

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

    def _resolve_deterministically(
        self,
        question: str,
        catalog: list[dict],
        recent_turns: list[dict],
        *,
        recent_messages: list[dict[str, str]],
        current_document_id: int | None,
        current_source_name: str | None,
    ) -> SourcePlan | None:
        explicit = self._matching_documents(question, catalog)
        if explicit:
            return SourcePlan(
                decision="scoped",
                intent="comparison" if len(explicit) > 1 else "single_paper",
                documents=[self._catalog_document(item) for item in explicit],
                confidence=1.0,
                allow_global_evidence=bool(_OUTSIDE_EVIDENCE_RE.search(question)),
                constraint_strength="hard",
                binding_basis="explicit_name",
                anchor_text=question[:200],
                validated_hard_scope=True,
                reason="deterministic_explicit_document_match",
            )

        current = self._current_catalog_document(
            catalog,
            current_document_id,
            current_source_name,
        )
        if current is not None and _CURRENT_DOCUMENT_CUE_RE.search(question):
            return SourcePlan(
                decision="scoped",
                intent="follow_up",
                documents=[self._catalog_document(current)],
                confidence=1.0,
                allow_global_evidence=bool(_OUTSIDE_EVIDENCE_RE.search(question)),
                constraint_strength="hard",
                binding_basis="current_document_reference",
                anchor_text=_CURRENT_DOCUMENT_CUE_RE.search(question).group(0),
                validated_hard_scope=True,
                reason="deterministic_current_document_reference",
            )

        reference = _REFERENCE_CUE_RE.search(question)
        if reference is not None:
            if recent_turns:
                last_turn = recent_turns[-1]
                antecedents = last_turn.get("retrieved_documents") or []
                if len(antecedents) == 1:
                    return SourcePlan(
                        decision="scoped",
                        intent="follow_up",
                        documents=[self._catalog_document(antecedents[0])],
                        confidence=1.0,
                        allow_global_evidence=bool(_OUTSIDE_EVIDENCE_RE.search(question)),
                        constraint_strength="hard",
                        binding_basis="recent_unique_antecedent",
                        anchor_text=reference.group(0),
                        antecedent_turn=int(last_turn["turn_index"]),
                        validated_hard_scope=True,
                        reason="deterministic_unique_recent_antecedent",
                    )
                if antecedents:
                    selected = []
                    if _ALL_RECENT_REFERENCE_RE.search(question):
                        selected = antecedents
                    elif _FORMER_REFERENCE_RE.search(question):
                        selected = antecedents[:1]
                    elif _LATTER_REFERENCE_RE.search(question) and len(antecedents) >= 2:
                        selected = antecedents[1:2]
                    if selected:
                        return SourcePlan(
                            decision="scoped",
                            intent="comparison" if len(selected) > 1 else "follow_up",
                            documents=[self._catalog_document(item) for item in selected],
                            confidence=1.0,
                            allow_global_evidence=bool(_OUTSIDE_EVIDENCE_RE.search(question)),
                            constraint_strength="hard",
                            binding_basis="comparison_reference",
                            anchor_text=reference.group(0),
                            antecedent_turn=int(last_turn["turn_index"]),
                            validated_hard_scope=True,
                            reason="deterministic_multi_document_reference",
                        )
                    return SourcePlan(
                        decision="ambiguous",
                        intent="follow_up",
                        confidence=1.0,
                        allow_global_evidence=True,
                        reason="multiple_possible_antecedents",
                    )
            return SourcePlan(
                decision="ambiguous",
                intent="follow_up",
                confidence=1.0,
                allow_global_evidence=True,
                reason="reference_without_structured_antecedent",
            )

        if _GENERAL_QUESTION_RE.search(question):
            return SourcePlan(
                decision="global",
                intent="open_research",
                confidence=1.0,
                allow_global_evidence=True,
                reason="general_question_without_source_reference",
            )

        # A standalone general question has no discourse scope to resolve. Skip
        # the LLM entirely instead of asking it to recommend relevant papers.
        if not recent_turns and current is None and not recent_messages:
            return SourcePlan(
                decision="global",
                intent="open_research",
                confidence=1.0,
                allow_global_evidence=True,
                reason="no_source_constraint_signals",
            )
        return None

    def _prepare_catalog(
        self,
        question: str,
        documents: list[dict],
        *,
        current_document_id: int | None,
        current_source_name: str | None,
    ) -> list[dict]:
        normalized_question = normalize_title(question)
        ranked = []
        for position, document in enumerate(documents):
            document_id = document.get("id")
            try:
                document_id = int(document_id)
            except (TypeError, ValueError):
                continue
            source_name = str(document.get("source_name") or "")
            title = str(document.get("paper_title") or source_name)
            title_hint = str(document.get("title_hint") or "")
            aliases = [str(item) for item in document.get("aliases") or []]
            normalized_title = normalize_title(title)
            normalized_hint = normalize_title(title_hint)
            normalized_source = normalize_title(source_name)
            score = 0
            if document_id == current_document_id or (
                current_source_name and source_name == current_source_name
            ):
                score += 1000
            if normalized_title and normalized_title in normalized_question:
                score += 500
            if normalized_source and normalized_source in normalized_question:
                score += 400
            if normalized_hint and normalized_hint in normalized_question:
                score += 450
            if any(self._alias_in_question(alias, question) for alias in aliases):
                score += 600
            # list_documents is already recent-first; preserve that as a weak tie-break.
            ranked.append(
                (score, -position, document_id, source_name, title, title_hint, aliases)
            )
        ranked.sort(reverse=True)
        return [
            {
                "document_id": document_id,
                "source_name": source_name,
                "title": title,
                "title_hint": title_hint,
                "aliases": aliases,
            }
            for _, _, document_id, source_name, title, title_hint, aliases in ranked[
                : self.config.catalog_max_documents
            ]
        ]

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
            reason=str(parsed.get("reason") or "")[:500],
        )

    def _validate_hard_scope(
        self,
        plan: SourcePlan,
        question: str,
        catalog: list[dict],
        recent_turns: list[dict],
        *,
        current_document_id: int | None,
        current_source_name: str | None,
    ) -> None:
        if (
            plan.decision != "scoped"
            or plan.constraint_strength != "hard"
            or not plan.documents
        ):
            plan.validated_hard_scope = False
            return

        routed_ids = set(plan.document_ids)
        valid = False
        if plan.binding_basis == "explicit_name":
            matched_ids = {
                int(item["document_id"])
                for item in self._matching_documents(question, catalog)
            }
            valid = bool(routed_ids) and routed_ids.issubset(matched_ids)
        elif plan.binding_basis == "current_document_reference":
            current = self._current_catalog_document(
                catalog,
                current_document_id,
                current_source_name,
            )
            valid = (
                current is not None
                and routed_ids == {int(current["document_id"])}
                and _CURRENT_DOCUMENT_CUE_RE.search(question) is not None
            )
        elif plan.binding_basis in {
            "recent_unique_antecedent",
            "recent_elliptical_follow_up",
            "comparison_reference",
        }:
            turn_index = plan.antecedent_turn if plan.antecedent_turn is not None else -1
            antecedent = next(
                (
                    turn
                    for turn in recent_turns
                    if int(turn["turn_index"]) == int(turn_index)
                ),
                None,
            )
            antecedent_ids = {
                int(item["document_id"])
                for item in (antecedent or {}).get("retrieved_documents", [])
            }
            cue_present = _REFERENCE_CUE_RE.search(question) is not None
            if plan.binding_basis == "recent_unique_antecedent":
                valid = (
                    len(antecedent_ids) == 1
                    and routed_ids == antecedent_ids
                    and cue_present
                )
            elif plan.binding_basis == "recent_elliptical_follow_up":
                valid = (
                    len(antecedent_ids) == 1
                    and routed_ids == antecedent_ids
                    and not cue_present
                    and _GENERAL_QUESTION_RE.search(question) is None
                )
            else:
                valid = (
                    len(antecedent_ids) >= 2
                    and bool(routed_ids)
                    and routed_ids.issubset(antecedent_ids)
                    and cue_present
                )

        if valid:
            plan.validated_hard_scope = True
            return

        had_reference = _REFERENCE_CUE_RE.search(question) is not None
        plan.decision = "ambiguous" if had_reference else "global"
        plan.documents = []
        plan.constraint_strength = "none"
        plan.binding_basis = "none"
        plan.validated_hard_scope = False
        plan.allow_global_evidence = True
        plan.reason = f"unvalidated_constraint: {plan.reason}"[:500]
