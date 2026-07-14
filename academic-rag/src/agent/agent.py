import json
from threading import RLock
from typing import List, Dict, Any
from loguru import logger
from openai import OpenAI

from src.rag.pipeline import RAGPipeline
from src.storage.app_store import SQLiteAppStore
from src.utils.config import LLMConfig


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
    }
]

AGENT_SYSTEM_PROMPT = """You are an academic paper analysis agent. Use the available tools to answer questions about indexed papers.

Tools:
1. search_papers: semantic search over paper content
2. get_paper_overview: retrieve a structured overview of a specific aspect

Guidelines:
- For complex questions, decompose them and retrieve step by step.
- After each tool call, decide whether further retrieval is needed.
- Synthesise results from multiple calls into a complete answer.
- If retrieved evidence is insufficient, say so explicitly.
- For follow-up questions referencing prior context ('it', 'this method', 'the above'), use conversation history to resolve the reference before searching."""


class ConversationMemory:
    """按 session_id 保存最近多轮用户问题和最终回答。"""

    def __init__(
        self,
        max_turns: int = 6,
        store: SQLiteAppStore | None = None,
    ):
        self.max_turns = max_turns
        self._store = store
        self._sessions: Dict[str, List[Dict[str, str]]] = {}
        self._lock = RLock()

    def get_messages(self, session_id: str) -> List[Dict[str, str]]:
        with self._lock:
            if self._store is not None:
                return self._store.get_conversation_messages(
                    session_id,
                    limit=self.max_turns * 2,
                )
            return list(self._sessions.get(session_id, []))

    def add_turn(
        self,
        session_id: str,
        user_query: str,
        assistant_answer: str,
        tool_summary: str | None = None,
    ) -> None:
        with self._lock:
            assistant_content = (
                f"[上轮检索: {tool_summary}]\n{assistant_answer}"
                if tool_summary
                else assistant_answer
            )
            if self._store is not None:
                self._store.add_conversation_turn(
                    session_id,
                    user_query,
                    assistant_content,
                )
                return

            history = self._sessions.setdefault(session_id, [])
            history.extend([
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": assistant_content},
            ])
            max_messages = self.max_turns * 2
            if len(history) > max_messages:
                history = history[-max_messages:]
            self._sessions[session_id] = history

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
        )

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
            messages.extend(self.memory.get_messages(session_id))
        messages.append({"role": "user", "content": user_query})

        tool_calls_log: list[str] = []

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
                    self.memory.add_turn(session_id, user_query, answer, tool_summary)
                return answer

            # 执行工具调用
            for tool_call in message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                tool_result, result_count = self._execute_tool(
                    tool_call.function.name,
                    args,
                    source_names=source_names,
                )
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
            self.memory.add_turn(session_id, user_query, answer, tool_summary)
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
        """执行工具调用，返回 (结果字符串, 结果条数)"""
        logger.info(f"调用工具: {tool_name}, 参数: {args}")

        if tool_name == "search_papers":
            query = args["query"]
            top_k = args.get("top_k", 3)
            chunks = self.rag.retrieve_chunks(
                query,
                top_k=top_k,
                source_filter=source_names,
            )

            if not chunks:
                return "No relevant content found.", 0

            results = []
            for chunk in chunks:
                results.append(
                    f"[score: {chunk.score:.3f}] {chunk.document.content[:300]}..."
                )
            return "\n\n".join(results), len(chunks)

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
                return f"No content found for aspect: {aspect}.", 0

            return "\n\n".join([c.document.content[:400] for c in chunks]), len(chunks)

        return f"Unknown tool: {tool_name}", 0
