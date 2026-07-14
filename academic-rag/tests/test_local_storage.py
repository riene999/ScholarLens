import numpy as np

from src.agent.agent import ConversationMemory
from src.storage.app_store import SQLiteAppStore
from src.storage.sqlite_cache import SQLiteTTLCache


def test_conversation_history_persists_while_context_is_bounded(tmp_path):
    db_path = tmp_path / "app.sqlite"
    memory = ConversationMemory(max_turns=2, store=SQLiteAppStore(db_path))
    memory.add_turn("session", "q1", "a1")
    memory.add_turn("session", "q2", "a2")
    memory.add_turn("session", "q3", "a3")

    reopened = ConversationMemory(max_turns=2, store=SQLiteAppStore(db_path))
    assert [item["content"] for item in reopened.get_messages("session")] == [
        "q2",
        "a2",
        "q3",
        "a3",
    ]

    with SQLiteAppStore(db_path)._connect() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS count FROM conversation_messages"
        ).fetchone()
    assert row["count"] == 6


def test_index_job_status_persists_and_interrupted_jobs_are_recovered(tmp_path):
    db_path = tmp_path / "app.sqlite"
    store = SQLiteAppStore(db_path)
    store.create_index_job("job-1", "paper.pdf")
    store.mark_index_job_started("job-1")

    reopened = SQLiteAppStore(db_path)
    assert reopened.get_index_job("job-1")["status"] == "started"
    assert reopened.recover_interrupted_jobs() == 1
    assert store.get_index_job("job-1")["status"] == "failed"


def test_sqlite_ttl_caches_survive_reopen(tmp_path):
    db_path = tmp_path / "app.sqlite"
    vector = np.array([[1.0, 2.0]], dtype=np.float32)

    SQLiteTTLCache(
        db_path,
        "vectors",
        value_codec="ndarray",
    ).set("question", vector)
    SQLiteTTLCache(
        db_path,
        "results",
        value_codec="json",
    ).set("question", [(1, 0.9, 1)])

    reopened_vectors = SQLiteTTLCache(
        db_path,
        "vectors",
        value_codec="ndarray",
    )
    reopened_results = SQLiteTTLCache(
        db_path,
        "results",
        value_codec="json",
    )
    assert np.array_equal(reopened_vectors.get("question"), vector)
    assert reopened_results.get("question") == [[1, 0.9, 1]]
