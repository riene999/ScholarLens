from src.agent.context_memory import ContextMemoryManager
from src.storage.app_store import SQLiteAppStore
from src.utils.config import MemoryConfig


def _settings(**overrides):
    values = {
        "soft_threshold_tokens": 75000,
        "hard_threshold_tokens": 100000,
        "hard_compact_batch_tokens": 50000,
        "preserve_recent_rounds": 3,
        "summary_max_tokens": 6000,
        "max_single_tool_result_tokens": 10000,
        "max_rag_evidence_tokens": 16000,
        "artifact_digest_tokens": 200,
        "token_estimate_safety_factor": 1.0,
    }
    values.update(overrides)
    return MemoryConfig(**values)


def test_artifact_is_content_addressed_and_round_trips(tmp_path):
    store = SQLiteAppStore(tmp_path / "app.sqlite")
    manager = ContextMemoryManager(store, _settings(), "gpt-4o-mini")
    payload = {"results": [{"content": "complete evidence"}]}

    first = manager.store_artifact("tool_result", payload, digest="evidence")
    second = manager.store_artifact("tool_result", payload, digest="evidence")

    assert first["sha256"] == second["sha256"]
    assert store.get_context_artifact(first["sha256"])["payload"] == payload


def test_soft_threshold_keeps_only_recent_artifact_raw_content(tmp_path):
    store = SQLiteAppStore(tmp_path / "app.sqlite")
    manager = ContextMemoryManager(
        store,
        _settings(
            soft_threshold_tokens=30,
            hard_threshold_tokens=10000,
            preserve_recent_rounds=1,
        ),
        "gpt-4o-mini",
    )
    old = manager.store_artifact(
        "rag_retrieval",
        {"content": "OLD_RAW_MARKER " * 80},
        digest="old evidence digest",
    )
    recent = manager.store_artifact(
        "rag_retrieval",
        {"content": "RECENT_RAW_MARKER " * 80},
        digest="recent evidence digest",
    )
    manager.add_turn("s", "old question", "old answer", artifacts=[old])
    manager.add_turn("s", "recent question", "recent answer", artifacts=[recent])

    active = "\n".join(message["content"] for message in manager.get_messages("s"))

    assert "OLD_RAW_MARKER" not in active
    assert "old evidence digest" in active
    assert f"artifact://sha256/{old['sha256']}" in active
    assert "RECENT_RAW_MARKER" in active


def test_hard_threshold_summarizes_oldest_complete_rounds_only(tmp_path):
    store = SQLiteAppStore(tmp_path / "app.sqlite")
    summary_inputs = []

    def summarize(prompt, max_tokens):
        summary_inputs.append((prompt, max_tokens))
        return "Cumulative summary preserving known facts and artifact references."

    manager = ContextMemoryManager(
        store,
        _settings(
            soft_threshold_tokens=20,
            hard_threshold_tokens=150,
            hard_compact_batch_tokens=30,
            preserve_recent_rounds=1,
            summary_max_tokens=50,
        ),
        "gpt-4o-mini",
        summarize=summarize,
    )
    old_artifact = manager.store_artifact(
        "tool_result",
        {"content": "old raw tool output"},
        digest="old tool digest",
    )
    for index in range(4):
        manager.add_turn(
            "s",
            f"question {index} " + "question " * 40,
            f"answer {index} " + "answer " * 40,
            artifacts=[old_artifact] if index == 0 else None,
        )

    active = manager.get_messages("s")
    summary = store.get_active_memory_summary("s")

    assert summary_inputs
    assert summary is not None
    assert summary["covers_through_round_id"] >= 1
    assert active[0]["content"].startswith("[CUMULATIVE MEMORY SUMMARY")
    assert f"artifact://sha256/{old_artifact['sha256']}" in active[0]["content"]
    assert len(
        store.get_context_rounds("s", after_round_id=summary["covers_through_round_id"])
    ) < 4
    with store._connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) AS count FROM conversation_rounds WHERE session_id = 's'"
        ).fetchone()["count"] == 4
