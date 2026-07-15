"""Token-budgeted, persistent conversation context for the RAG agent."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Callable

from loguru import logger

from src.storage.app_store import SQLiteAppStore
from src.utils.config import MemoryConfig


SUMMARY_PROMPT = """You maintain a cumulative memory for an academic RAG agent.

Summarize the previous cumulative memory and the oldest conversation rounds below.
Preserve facts needed for future turns, especially:
- user goals, constraints, preferences, and unresolved questions;
- documents, paper names, methods, results, citations, and source relationships;
- decisions already made and actions already taken;
- what is known, uncertain, missing, or still needs retrieval;
- every artifact reference in the form artifact://sha256/<hash>.

Do not invent facts. Do not treat an artifact digest as the complete artifact. Write a
compact structured memory that can replace the supplied history in the active prompt.

CONTENT TO COMPACT:
{content}
"""


@dataclass(frozen=True)
class RenderedRound:
    round_id: int
    messages: list[dict[str, str]]
    token_count: int
    artifact_refs: list[str]


class TokenCounter:
    """Offline conservative estimator suitable for mixed Chinese/English text."""

    def __init__(self, model: str, safety_factor: float = 1.1):
        # DeepSeek does not expose a local exact tokenizer here. Estimating one token
        # per three UTF-8 bytes is deliberately conservative for English and close to
        # one token per CJK character, while never requiring a runtime download.
        self.model = model
        self.safety_factor = max(1.0, float(safety_factor))

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        estimated = math.ceil(len(text.encode("utf-8")) / 3)
        return max(1, math.ceil(estimated * self.safety_factor))

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        # The per-message allowance covers role markers and provider-specific framing.
        return sum(self.count_text(str(message.get("content", ""))) + 6 for message in messages) + 3

    def truncate(self, text: str, max_tokens: int) -> str:
        max_tokens = max(1, int(max_tokens))
        if self.count_text(text) <= max_tokens:
            return text
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if self.count_text(text[:middle]) <= max_tokens:
                low = middle
            else:
                high = middle - 1
        return text[:low] + "\n… [truncated; use artifact reference for full content]"


class ContextMemoryManager:
    """Builds a compact active prompt while retaining immutable raw history."""

    def __init__(
        self,
        store: SQLiteAppStore,
        settings: MemoryConfig,
        model: str,
        summarize: Callable[[str, int], str] | None = None,
        response_reserve_tokens: int = 0,
    ):
        self.store = store
        self.settings = settings
        self.counter = TokenCounter(model, settings.token_estimate_safety_factor)
        self.summarize = summarize
        self.response_reserve_tokens = max(0, int(response_reserve_tokens))

    def store_artifact(
        self,
        artifact_type: str,
        payload: object,
        *,
        metadata: dict | None = None,
        digest: str | None = None,
        tool_name: str | None = None,
    ) -> dict:
        canonical = self.store.canonical_artifact_payload(payload).decode("utf-8")
        token_count = self.counter.count_text(canonical)
        safe_digest = digest or self.counter.truncate(
            canonical,
            self.settings.artifact_digest_tokens,
        )
        sha256 = self.store.put_context_artifact(
            artifact_type=artifact_type,
            payload=payload,
            digest=safe_digest,
            token_count=token_count,
            metadata=metadata,
        )
        return {
            "sha256": sha256,
            "artifact_type": artifact_type,
            "tool_name": tool_name,
            "metadata": metadata or {},
        }

    def add_turn(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
        *,
        tool_summary: str | None = None,
        artifacts: list[dict] | None = None,
    ) -> int:
        base_messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        return self.store.add_conversation_turn(
            session_id,
            user_content,
            assistant_content,
            tool_summary=tool_summary,
            token_count=self.counter.count_messages(base_messages),
            artifacts=artifacts,
        )

    def get_messages(
        self,
        session_id: str,
        pending_content: str | None = None,
    ) -> list[dict[str, str]]:
        """Return the active prompt view, compacting only when thresholds require it."""
        summary = self.store.get_active_memory_summary(session_id)
        rounds = self.store.get_context_rounds(
            session_id,
            after_round_id=int(summary["covers_through_round_id"]) if summary else 0,
        )
        if not rounds:
            # Compatibility for histories created before conversation_rounds existed.
            if summary:
                return [
                    self._summary_message(
                        summary["content"],
                        summary.get("artifact_refs", []),
                    )
                ]
            return self.store.get_conversation_messages(session_id, limit=1_000_000)

        full = self._render_context(summary, rounds, soft_mode=False)
        if summary is None and self._active_tokens(full, pending_content) < self.settings.soft_threshold_tokens:
            return full

        active = self._render_context(summary, rounds, soft_mode=True)
        while (
            self._active_tokens(active, pending_content) >= self.settings.hard_threshold_tokens
        ):
            if not self._compact_oldest_batch(session_id, summary, rounds):
                break
            summary = self.store.get_active_memory_summary(session_id)
            rounds = self.store.get_context_rounds(
                session_id,
                after_round_id=int(summary["covers_through_round_id"]),
            )
            active = self._render_context(summary, rounds, soft_mode=True)
        return active

    def _active_tokens(
        self,
        messages: list[dict[str, str]],
        pending_content: str | None,
    ) -> int:
        return (
            self.counter.count_messages(messages)
            + self.counter.count_text(pending_content or "")
            + self.response_reserve_tokens
        )

    def get_artifact_text(self, sha256: str, max_tokens: int | None = None) -> str:
        sha256 = sha256.strip().lower().rsplit("/", 1)[-1]
        if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
            return f"Invalid SHA256 artifact identifier: {sha256}"
        artifact = self.store.get_context_artifact(sha256)
        if artifact is None:
            return f"Artifact not found: {sha256}"
        text = json.dumps(artifact["payload"], ensure_ascii=False, indent=2)
        limit = max_tokens or self._artifact_limit(artifact["artifact_type"])
        body = self.counter.truncate(text, limit)
        return (
            f"artifact://sha256/{artifact['sha256']}\n"
            f"type: {artifact['artifact_type']}\n"
            f"digest: {artifact['digest']}\n"
            f"content:\n{body}"
        )

    def _render_context(
        self,
        summary: dict | None,
        rounds: list[dict],
        *,
        soft_mode: bool,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if summary:
            messages.append(
                self._summary_message(
                    summary["content"],
                    summary.get("artifact_refs", []),
                )
            )
        recent_start = max(0, len(rounds) - self.settings.preserve_recent_rounds)
        for index, round_data in enumerate(rounds):
            hydrate = not soft_mode or index >= recent_start
            messages.extend(self._render_round(round_data, hydrate=hydrate).messages)
        return messages

    def _render_round(self, round_data: dict, *, hydrate: bool) -> RenderedRound:
        assistant = round_data["assistant_content"]
        if round_data.get("tool_summary"):
            assistant += f"\n\n[tool activity] {round_data['tool_summary']}"
        artifact_blocks: list[str] = []
        refs: list[str] = []
        for link in round_data.get("artifacts", []):
            sha256 = link["sha256"]
            ref = f"artifact://sha256/{sha256}"
            refs.append(sha256)
            if hydrate:
                artifact_blocks.append(self.get_artifact_text(sha256))
            else:
                label = link.get("tool_name") or link["artifact_type"]
                artifact_blocks.append(
                    f"[{label}] {ref}\n"
                    f"digest: {link['digest']}\n"
                    "Full raw content is stored locally and can be recovered with get_context_artifact."
                )
        if artifact_blocks:
            assistant += "\n\n[retrieval/tool artifacts]\n" + "\n\n".join(artifact_blocks)
        messages = [
            {"role": "user", "content": round_data["user_content"]},
            {"role": "assistant", "content": assistant},
        ]
        return RenderedRound(
            round_id=int(round_data["id"]),
            messages=messages,
            token_count=self.counter.count_messages(messages),
            artifact_refs=refs,
        )

    def _compact_oldest_batch(
        self,
        session_id: str,
        summary: dict | None,
        rounds: list[dict],
    ) -> bool:
        if self.summarize is None:
            logger.warning("Hard memory threshold reached but no summarizer is configured")
            return False
        if not rounds:
            logger.warning("Hard memory threshold reached with no complete rounds to compact")
            return False

        selected: list[RenderedRound] = []
        selected_tokens = 0
        recent_start = max(0, len(rounds) - self.settings.preserve_recent_rounds)
        for index, round_data in enumerate(rounds):
            # Hard compaction operates on the actual soft-mode view: older evidence
            # is referenced, while evidence in the recent window is still hydrated.
            rendered = self._render_round(round_data, hydrate=index >= recent_start)
            selected.append(rendered)
            selected_tokens += rendered.token_count
            if selected_tokens > self.settings.hard_compact_batch_tokens:
                break
        if not selected:
            return False

        compact_messages: list[dict[str, str]] = []
        if summary:
            compact_messages.append(
                self._summary_message(
                    summary["content"],
                    summary.get("artifact_refs", []),
                )
            )
        for rendered in selected:
            compact_messages.extend(rendered.messages)
        compact_text = "\n\n".join(
            f"{message['role'].upper()}:\n{message['content']}"
            for message in compact_messages
        )
        prompt = SUMMARY_PROMPT.format(content=compact_text)
        try:
            new_summary = self.summarize(prompt, self.settings.summary_max_tokens).strip()
        except Exception:
            logger.exception("Failed to compact conversation memory")
            return False
        if not new_summary:
            logger.warning("Memory summarizer returned an empty result")
            return False

        refs = [sha for rendered in selected for sha in rendered.artifact_refs]
        if summary:
            refs.extend(summary.get("artifact_refs", []))
        stored_summary = self.counter.truncate(new_summary, self.settings.summary_max_tokens)
        self.store.save_memory_summary(
            session_id=session_id,
            content=stored_summary,
            token_count=self.counter.count_text(stored_summary),
            covers_through_round_id=selected[-1].round_id,
            previous_summary_id=int(summary["id"]) if summary else None,
            artifact_refs=refs,
        )
        logger.info(
            "Compacted {} oldest rounds (estimated {} tokens) for session {}",
            len(selected),
            selected_tokens,
            session_id,
        )
        return True

    @staticmethod
    def _summary_message(
        content: str,
        artifact_refs: list[str] | None = None,
    ) -> dict[str, str]:
        artifact_index = ""
        if artifact_refs:
            artifact_index = "\n\n[ARTIFACT INDEX FOR SUMMARIZED HISTORY]\n" + "\n".join(
                f"artifact://sha256/{sha256}" for sha256 in sorted(set(artifact_refs))
            )
        return {
            "role": "assistant",
            "content": (
                "[CUMULATIVE MEMORY SUMMARY — not a new answer]\n"
                + content
                + artifact_index
            ),
        }

    def _artifact_limit(self, artifact_type: str) -> int:
        if artifact_type == "rag_retrieval":
            return self.settings.max_rag_evidence_tokens
        return self.settings.max_single_tool_result_tokens
