"""Evaluate source-constraint detection and multi-turn paper resolution."""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main as api
from src.rag.source_router import RoutedDocument, SourcePlan, SourceRouter
from src.utils.config import load_config
from src.utils.pdf_parser import normalize_title


def load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON on line {line_number}: {exc}") from exc
    if not cases:
        raise RuntimeError("Source-routing dataset is empty")
    return cases


def load_documents(database: Path) -> list[dict]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT d.id, d.source_name, d.paper_title, c.content
            FROM documents d
            LEFT JOIN chunks c
              ON c.document_id = d.id AND c.chunk_index = 0
            ORDER BY d.id ASC
            """
        ).fetchall()
    finally:
        connection.close()
    documents = []
    for row in rows:
        content = str(row["content"] or "")
        first_line = next(
            (line.strip() for line in content.splitlines() if line.strip()),
            "",
        )
        documents.append({
            "id": int(row["id"]),
            "source_name": str(row["source_name"]),
            "paper_title": str(row["paper_title"] or row["source_name"]),
            "title_hint": first_line[:300],
            "aliases": api._extract_router_aliases(
                content,
                title=str(row["paper_title"] or ""),
                source_name=str(row["source_name"]),
            ),
        })
    return documents


def recent_messages(turns: list[dict]) -> list[dict[str, str]]:
    messages = []
    for turn in turns:
        messages.extend([
            {"role": "user", "content": str(turn.get("user") or "")},
            {"role": "assistant", "content": str(turn.get("assistant") or "")},
        ])
    return messages


def resolve_with_legacy_keyword_match(question: str, documents: list[dict]) -> SourcePlan:
    """Reproduce the pre-router title/filename substring constraint logic."""
    normalized_question = normalize_title(question)
    matches = []
    if len(normalized_question) >= 8:
        for document in documents:
            keys = {
                normalize_title(str(document.get("source_name") or "")),
                normalize_title(str(document.get("paper_title") or "")),
            }
            if any(
                len(key) >= 12
                and (key in normalized_question or normalized_question in key)
                for key in keys
            ):
                matches.append(document)
    if not matches:
        return SourcePlan(
            decision="global",
            intent="open_research",
            confidence=1.0,
            allow_global_evidence=True,
            reason="legacy_no_keyword_match",
        )
    return SourcePlan(
        decision="scoped",
        intent="comparison" if len(matches) > 1 else "single_paper",
        documents=[
            RoutedDocument(
                document_id=int(item["id"]),
                source_name=str(item["source_name"]),
                title=str(item["paper_title"]),
            )
            for item in matches
        ],
        confidence=1.0,
        allow_global_evidence=False,
        constraint_strength="hard",
        binding_basis="explicit_name",
        anchor_text=question[:200],
        validated_hard_scope=True,
        reason="legacy_keyword_match",
    )


def score_case(
    case: dict,
    plan: SourcePlan | None,
    *,
    timed_out: bool,
    error: str | None,
    deferred_to_llm: bool,
    used_llm: bool,
    production_budget_ms: int,
) -> dict:
    predicted_decision = (
        plan.decision
        if plan is not None
        else ("needs_llm" if deferred_to_llm else ("timeout" if timed_out else "error"))
    )
    predicted_sources = plan.source_names if plan is not None else []
    expected_sources = set(case["expected_sources"])
    predicted_source_set = set(predicted_sources)
    expected_scoped = case["expected_decision"] == "scoped"
    predicted_scoped = predicted_decision == "scoped"
    source_recall = (
        len(expected_sources.intersection(predicted_source_set)) / len(expected_sources)
        if expected_sources
        else float(not predicted_source_set)
    )
    trace = plan.to_trace() if plan is not None else None
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "language": case["language"],
        "question": case["question"],
        "expected_decision": case["expected_decision"],
        "predicted_decision": predicted_decision,
        "decision_correct": predicted_decision == case["expected_decision"],
        "expected_sources": case["expected_sources"],
        "predicted_sources": predicted_sources,
        "source_exact_match": predicted_source_set == expected_sources,
        "source_recall": source_recall,
        "expected_binding_basis": case["expected_binding_basis"],
        "binding_basis_correct": (
            plan is not None
            and plan.binding_basis == case["expected_binding_basis"]
        ),
        "expected_hard_scope": bool(case["expected_hard_scope"]),
        "predicted_hard_scope": bool(plan and plan.has_validated_scope),
        "hard_scope_correct": bool(plan and plan.has_validated_scope) == bool(
            case["expected_hard_scope"]
        ),
        "expected_scoped": expected_scoped,
        "predicted_scoped": predicted_scoped,
        "timed_out": timed_out,
        "error": error,
        "deferred_to_llm": deferred_to_llm,
        "used_llm": used_llm,
        "execution_mode": (
            "llm"
            if used_llm
            else ("needs_llm" if deferred_to_llm else "deterministic")
        ),
        "within_production_budget": bool(
            plan is not None and plan.latency_ms <= production_budget_ms
        ),
        "routing": trace,
    }


def mean(values) -> float:
    values = list(values)
    return round(statistics.fmean(values), 6) if values else 0.0


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return float(ordered[index])


def summarize(results: list[dict]) -> dict:
    locally_resolved = [
        item
        for item in results
        if item["routing"] is not None
        and not item["deferred_to_llm"]
        and not item["used_llm"]
    ]
    llm_required = [
        item for item in results if item["deferred_to_llm"] or item["used_llm"]
    ]
    llm_executed = [item for item in results if item["used_llm"]]
    true_positive = sum(
        item["expected_scoped"] and item["predicted_scoped"] for item in results
    )
    false_positive = sum(
        not item["expected_scoped"] and item["predicted_scoped"] for item in results
    )
    false_negative = sum(
        item["expected_scoped"] and not item["predicted_scoped"] for item in results
    )
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    scoped = [item for item in results if item["expected_scoped"]]
    global_cases = [item for item in results if item["expected_decision"] == "global"]
    ambiguous = [item for item in results if item["expected_decision"] == "ambiguous"]
    latencies = [
        int(item["routing"]["latency_ms"])
        for item in results
        if item["routing"] is not None
    ]
    categories = {}
    for category in sorted({item["category"] for item in results}):
        items = [item for item in results if item["category"] == category]
        categories[category] = {
            "case_count": len(items),
            "decision_accuracy": mean(float(item["decision_correct"]) for item in items),
            "source_exact_match_rate": mean(
                float(item["source_exact_match"]) for item in items
            ),
        }
    return {
        "case_count": len(results),
        "decision_accuracy": mean(float(item["decision_correct"]) for item in results),
        "local_resolution_rate": mean(
            float(item in locally_resolved) for item in results
        ),
        "local_decision_accuracy_on_resolved": mean(
            float(item["decision_correct"]) for item in locally_resolved
        ),
        "llm_required_rate": mean(float(item in llm_required) for item in results),
        "llm_execution_rate": mean(float(item["used_llm"]) for item in results),
        "llm_decision_accuracy": mean(
            float(item["decision_correct"]) for item in llm_executed
        ),
        "constraint_precision": round(precision, 6),
        "constraint_recall": round(recall, 6),
        "constraint_f1": round(f1, 6),
        "scoped_source_exact_match_rate": mean(
            float(item["source_exact_match"]) for item in scoped
        ),
        "scoped_mean_source_recall": mean(item["source_recall"] for item in scoped),
        "binding_basis_accuracy": mean(
            float(item["binding_basis_correct"]) for item in results
        ),
        "hard_scope_accuracy": mean(float(item["hard_scope_correct"]) for item in results),
        "false_scope_rate_on_global": mean(
            float(item["predicted_scoped"]) for item in global_cases
        ),
        "ambiguous_recall": mean(
            float(item["predicted_decision"] == "ambiguous") for item in ambiguous
        ),
        "timeout_rate": mean(float(item["timed_out"]) for item in results),
        "error_rate": mean(float(bool(item["error"])) for item in results),
        "within_router_timeout_budget_rate": mean(
            float(item["within_production_budget"]) for item in results
        ),
        "deterministic_resolution_rate": mean(
            float(item in locally_resolved) for item in results
        ),
        "latency_ms": {
            "mean": mean(latencies),
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
        },
        "by_category": categories,
    }


def write_report(
    output: Path,
    *,
    dataset: Path,
    timeout_ms: int,
    production_budget_ms: int,
    router_mode: str,
    results: list[dict],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "dataset": str(dataset),
                "timeout_ms": timeout_ms,
                "production_budget_ms": production_budget_ms,
                "router_mode": router_mode,
                "metrics": summarize(results),
                "cases": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


async def evaluate(args) -> None:
    config = load_config(args.config)
    router = SourceRouter(config.llm, config.source_routing)
    documents = load_documents(args.database)
    cases = load_cases(args.dataset)
    results = []
    for index, case in enumerate(cases, start=1):
        current = case.get("current_document") or {}
        timed_out = False
        error = None
        deferred_to_llm = False
        used_llm = False
        plan = None
        if args.mode == "legacy-keyword":
            plan = resolve_with_legacy_keyword_match(case["question"], documents)
        else:
            try:
                plan = await asyncio.wait_for(
                    router.resolve(
                        case["question"],
                        documents,
                        recent_messages=recent_messages(case["recent_turns"]),
                        recent_turns=case["recent_turns"],
                        current_document_id=current.get("document_id"),
                        current_source_name=current.get("source_name"),
                    ),
                    timeout=max(0.05, args.timeout_ms / 1000),
                )
                used_llm = plan is not None
            except asyncio.TimeoutError:
                timed_out = True
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        result = score_case(
            case,
            plan,
            timed_out=timed_out,
            error=error,
            deferred_to_llm=deferred_to_llm,
            used_llm=used_llm,
            production_budget_ms=config.source_routing.total_timeout_ms,
        )
        results.append(result)
        write_report(
            args.output,
            dataset=args.dataset,
            timeout_ms=args.timeout_ms,
            production_budget_ms=config.source_routing.total_timeout_ms,
            router_mode=args.mode,
            results=results,
        )
        print(
            f"[{index:02d}/{len(cases)}] {case['case_id']} "
            f"expected={case['expected_decision']} predicted={result['predicted_decision']}"
        )
    print(json.dumps(summarize(results), ensure_ascii=False, indent=2))
    print(f"Full report: {args.output}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/faiss_indexes/marker/documents.sqlite"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("eval/datasets/source_routing_eval_50.jsonl"),
    )
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument(
        "--mode",
        choices=["current", "legacy-keyword"],
        default="current",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/results/source_routing_eval_50_llm_only.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(evaluate(parse_args()))
