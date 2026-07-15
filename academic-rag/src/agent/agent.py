import json
from threading import RLock
from typing import List, Dict, Any
from loguru import logger
from openai import OpenAI

from src.agent.context_memory import ContextMemoryManager
from src.rag.pipeline import RAGPipeline
from src.storage.app_store import SQLiteAppStore
from src.utils.config import LLMConfig, MemoryConfig


# 工具定义（OpenAI Function Calling格式）
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "Semantically search indexed academic papers for content relevant to a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A specific academic question or concept to search for."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of passages to return (default 3).",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_paper_overview",
            "description": "Retrieve a high-level overview of a specific aspect of the paper.",
            "parameters": {
                "type": "object",
                "properties": {
                    "aspect": {
                        "type": "string",
                        "enum": ["research_question", "methodology", "results", "conclusion"],
                        "description": "The aspect of the paper to retrieve."
                    }
                },
                "required": ["aspect"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_context_artifact",
            "description": "Recover full tool or RAG evidence previously replaced by an artifact://sha256 reference.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sha256": {
                        "type": "string",
                        "description": "The 64-character SHA256 from an artifact reference."
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum recovered content tokens (default uses the configured artifact budget)."
                    }
                },
                "required": ["sha256"]
            }
        }
    }
]

AGENT_SYSTEM_PROMPT = """You are an academic paper analysis agent. Use the available tools to answer questions about indexed papers.

Tools:
1. search_papers: semantic search over paper content
2. get_paper_overview: retrieve a structured overview of a specific aspect
3. get_context_artifact: recover evidence referenced by artifact://sha256/<hash>

Guidelines:
- For complex questions, decompose them and retrieve step by step.
- After each tool call, decide whether further retrieval is needed.
- Synthesise results from multiple calls into a complete answer.
- If retrieved evidence is insufficient, say so explicitly.
- Artifact digests are indexes, not complete evidence. Recover an artifact when its full content is needed.
- For follow-up questions referencing prior context ('it', 'this method', 'the above'), use conversation history to resolve the reference before searching."""


class ConversationMemory:
    """按 session_id 保存最近多轮用户问题和最终回答。"""

    def __init__(
        self,
        max_turns: int = 6,
        store: SQLiteAppStore | None = None,
        memory_config: MemoryConfig | None = None,
        model: str = "gpt-4o-mini",
        summarize=None,
        response_reserve_tokens: int = 0,
    ):
        self.max_turns = max_turns
        self._store = store
        self._sessions: Dict[str, List[Dict[str, str]]] = {}
        self._lock = RLock()
        self._manager = None
        if store is not None:
            settings = memory_config or MemoryConfig(
                soft_threshold_tokens=75000,
                hard_threshold_tokens=100000,
                hard_compact_batch_tokens=50000,
                preserve_recent_rounds=3,
                summary_max_tokens=6000,
                max_single_tool_result_tokens=10000,
                max_rag_evidence_tokens=16000,
                artifact_digest_tokens=200,
                token_estimate_safety_factor=1.1,
            )
            self._manager = ContextMemoryManager(
                store,
                settings,
                model,
                summarize,
                response_reserve_tokens=response_reserve_tokens,
            )

    def get_messages(
        self,
        session_id: str,
        pending_content: str | None = None,
    ) -> List[Dict[str, str]]:
        with self._lock:
            if self._store is not None:
                return self._manager.get_messages(session_id, pending_content=pending_content)
            return list(self._sessions.get(session_id, []))

    def add_turn(
        self,
        session_id: str,
        user_query: str,
        assistant_answer: str,
        tool_summary: str | None = None,
        artifacts: list[dict] | None = None,
    ) -> None:
        with self._lock:
            if self._store is not None:
                self._manager.add_turn(
                    session_id=session_id,
                    user_content=user_query,
                    assistant_content=assistant_answer,
                    tool_summary=tool_summary,
                    artifacts=artifacts,
                )
                return

            assistant_content = (
                f"[上轮检索: {tool_summary}]\n{assistant_answer}"
                if tool_summary
                else assistant_answer
            )
            history = self._sessions.setdefault(session_id, [])
            history.extend([
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": assistant_content},
            ])
            max_messages = self.max_turns * 2
            if len(history) > max_messages:
                history = history[-max_messages:]
            self._sessions[session_id] = history

    def store_artifact(
        self,
        artifact_type: str,
        payload: object,
        **kwargs,
    ) -> dict | None:
        if self._manager is None:
            return None
        return self._manager.store_artifact(artifact_type, payload, **kwargs)

    def get_artifact_text(self, sha256: str, max_tokens: int | None = None) -> str:
        if self._manager is None:
            return "Persistent artifact storage is not configured."
        return self._manager.get_artifact_text(sha256, max_tokens=max_tokens)

    def clear(self, session_id: str | None = None) -> None:
        with self._lock:
            if self._store is not None:
                self._store.clear_conversation(session_id)
                return
            if session_id is None:
                self._sessions.clear()
                return
            self._sessions.pop(session_id, None)

    def session_count(self) -> int:
        with self._lock:
            if self._store is not None:
                return self._store.conversation_session_count()
            return len(self._sessions)


class PaperAgent:
    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        llm_config: LLMConfig,
        memory_max_turns: int = 6,
        conversation_store: SQLiteAppStore | None = None,
        memory_config: MemoryConfig | None = None,
    ):
        self.rag = rag_pipeline
        self.client = OpenAI(
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
        )
        self.llm_config = llm_config
        self.memory = ConversationMemory(
            max_turns=memory_max_turns,
            store=conversation_store,
            memory_config=memory_config,
            model=llm_config.model,
            summarize=self._summarize_memory,
            response_reserve_tokens=llm_config.max_tokens,
        )

    def _summarize_memory(self, prompt: str, max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.llm_config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def run(
        self,
        user_query: str,
        max_iterations: int = 5,
        session_id: str = "default",
        use_memory: bool = True,
        source_names: list[str] | None = None,
    ) -> str:
        """
        ReAct Agent主循环
        LLM决策 -> 工具调用 -> 观察结果 -> 继续决策 -> 最终回答
        """
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        ]
        if use_memory:
            messages.extend(
                self.memory.get_messages(
                    session_id,
                    pending_content=AGENT_SYSTEM_PROMPT + "\n" + user_query,
                )
            )
        messages.append({"role": "user", "content": user_query})

        tool_calls_log: list[str] = []
        round_artifacts: list[dict] = []

        for iteration in range(max_iterations):
            logger.info(f"Agent迭代 {iteration + 1}/{max_iterations}")

            response = self.client.chat.completions.create(
                model=self.llm_config.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,  # Agent决策用低温，保证稳定
            )

            message = response.choices[0].message
            messages.append(message)

            # 没有工具调用，直接返回最终回答
            if not message.tool_calls:
                logger.info("Agent完成，返回最终回答")
                answer = message.content or ""
                if use_memory:
                    tool_summary = "; ".join(tool_calls_log) if tool_calls_log else None
                    self.memory.add_turn(
                        session_id,
                        user_query,
                        answer,
                        tool_summary,
                        artifacts=round_artifacts,
                    )
                return answer

            # 执行工具调用
            for tool_call in message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                tool_result, result_count, artifact = self._execute_tool_with_artifact(
                    tool_call.function.name,
                    args,
                    source_names=source_names,
                )
                if artifact:
                    round_artifacts.append(artifact)
                # 记录本次工具调用摘要，格式：工具名("query") → N条
                query_hint = args.get("query") or args.get("aspect", "")
                tool_calls_log.append(
                    f'{tool_call.function.name}("{query_hint}") → {result_count}条'
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })

        # 超过最大迭代次数，强制生成回答（tool_choice="none" 防止再次调用工具）
        logger.warning("达到最大迭代次数，强制生成回答")
        final_response = self.client.chat.completions.create(
            model=self.llm_config.model,
            messages=messages,
            tools=TOOLS,
            tool_choice="none",
            temperature=self.llm_config.temperature,
        )
        answer = final_response.choices[0].message.content or ""
        if use_memory:
            tool_summary = "; ".join(tool_calls_log) if tool_calls_log else None
            self.memory.add_turn(
                session_id,
                user_query,
                answer,
                tool_summary,
                artifacts=round_artifacts,
            )
        return answer

    def clear_memory(self, session_id: str | None = None) -> None:
        """清空指定会话或全部会话记忆。"""
        self.memory.clear(session_id)

    def _execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        source_names: list[str] | None = None,
    ) -> tuple[str, int]:
        result, count, _ = self._execute_tool_with_artifact(
            tool_name,
            args,
            source_names=source_names,
        )
        return result, count

    def _execute_tool_with_artifact(
        self,
        tool_name: str,
        args: Dict[str, Any],
        source_names: list[str] | None = None,
    ) -> tuple[str, int, dict | None]:
        """执行工具调用，并将完整原始结果保存为 artifact。"""
        logger.info(f"调用工具: {tool_name}, 参数: {args}")

        if tool_name == "get_context_artifact":
            text = self.memory.get_artifact_text(
                args["sha256"],
                max_tokens=args.get("max_tokens"),
            )
            return text, 1, None

        if tool_name == "search_papers":
            query = args["query"]
            top_k = args.get("top_k", 3)
            chunks = self.rag.retrieve_chunks(
                query,
                top_k=top_k,
                source_filter=source_names,
            )

            if not chunks:
                return "No relevant content found.", 0, None

            results = []
            for chunk in chunks:
                results.append(
                    f"[score: {chunk.score:.3f}] {chunk.document.content[:300]}..."
                )
            payload = {
                "tool": tool_name,
                "arguments": args,
                "results": [self._serialize_chunk(chunk) for chunk in chunks],
            }
            artifact = self.memory.store_artifact(
                "tool_result",
                payload,
                tool_name=tool_name,
                metadata={"result_count": len(chunks), "query": query},
                digest=f"search_papers({query!r}) returned {len(chunks)} passages",
            )
            result_text = "\n\n".join(results)
            if artifact:
                result_text += f"\n\nFull result: artifact://sha256/{artifact['sha256']}"
            return result_text, len(chunks), artifact

        elif tool_name == "get_paper_overview":
            aspect = args["aspect"]
            aspect_queries = {
                "research_question": "research question motivation problem why",
                "methodology": "method model algorithm framework approach",
                "results": "experiment results performance accuracy evaluation",
                "conclusion": "conclusion summary contribution future work",
            }
            query = aspect_queries.get(aspect, aspect)
            chunks = self.rag.retrieve_chunks(query, top_k=3, source_filter=source_names)

            if not chunks:
                return f"No content found for aspect: {aspect}.", 0, None

            payload = {
                "tool": tool_name,
                "arguments": args,
                "results": [self._serialize_chunk(chunk) for chunk in chunks],
            }
            artifact = self.memory.store_artifact(
                "tool_result",
                payload,
                tool_name=tool_name,
                metadata={"result_count": len(chunks), "aspect": aspect},
                digest=f"get_paper_overview({aspect!r}) returned {len(chunks)} passages",
            )
            result_text = "\n\n".join([c.document.content[:400] for c in chunks])
            if artifact:
                result_text += f"\n\nFull result: artifact://sha256/{artifact['sha256']}"
            return result_text, len(chunks), artifact

        return f"Unknown tool: {tool_name}", 0, None

    @staticmethod
    def _serialize_chunk(chunk) -> dict:
        return {
            "chunk_id": chunk.document.chunk_id,
            "content": chunk.document.content,
            "score": float(chunk.score),
            "rank": int(chunk.rank),
            "metadata": chunk.document.metadata or {},
        }
